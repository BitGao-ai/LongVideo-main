"""统一数据加载工厂：一处构造 loader，训练/评测/合成三态同口径。

替代 train_stage1/2 里各自内联的 `SyntheticVideoDataset + make_loader`，解决三件事：
  1) **真实 manifest 接入**：有 manifest → VideoTemporalDataset（feature/pixel），无 → SyntheticVideoDataset。
  2) **无限取批 `cycle`**：Trainer 按 max_steps 停，但真实 loader 有限；cycle 让 step 训练不会因
     max_steps>len(loader) 提前结束（合成路径原本靠 n=steps*2+4 侥幸够用，脆弱）。
  3) **吞吐/一致性**：num_workers/pin_memory/persistent_workers/drop_last + 特征维自动推断（防 feat_dim 错配）。
  4) **DDP 分布式**：自动检测分布式环境并注入 DistributedSampler，保证各 rank 数据不重叠。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .schema import DataConfig
from .bucketing import LengthGroupedBatchSampler
from .dataset import (VideoTemporalDataset, SyntheticVideoDataset,
                      SyntheticGroundingDataset, ByteTokenizer)
from .collate import collate_fn
from ..utils.distributed import dist_info


@dataclass
class LoaderConfig:
    # 真实数据（manifest 非空即启用真实路径）
    manifest: str | None = None
    data_root: str = ""
    mode: str = "feature"                 # feature | pixel
    max_frames: int = 8192                # 与 DataConfig/SamplerConfig 保持一致
    max_text_len: int = 8704              # ≥ max_frames + 文本，见 DataConfig 说明
    # 批与吞吐
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False
    pad_id: int = 256
    # 长度分桶（长视频显存利用率）：把长度相近的样本放进同一批，减少 collate 的 padding 浪费。
    # 数值中性——padding 位本就被 frame_mask 掩掉，分桶只改变样本的**分组方式**，不改变
    # 任何一个样本的内容。仅对真实 manifest 生效（合成数据等长，分桶无意义）。
    length_bucketing: bool = False
    bucket_multiplier: int = 32           # megabatch = batch_size × world_size × 该值
    seed: int = 0                         # 分桶采样器的洗牌种子（各 rank 必须一致）
    # 特征维（合成用；真实可 auto-infer 覆盖）
    feat_dim: int = 64
    feat_patches: int = 4
    # 特征张量 dtype，见 DataConfig.feat_dtype。"auto"（默认）= 沿用特征文件自身的 dtype，
    # 即 fp16 存就 fp16 用——省掉一份从来没有消费者的 fp32 副本（worker RSS / pinned
    # buffer / H2D / GPU 常驻 batch 全部减半）。只对真实 manifest 有效；合成数据恒 fp32。
    feat_dtype: str = "auto"
    # 合成兜底
    synth_n: int = 128
    synth_frames: int = 32
    synth_grounding: bool = False         # True：合成定位数据（带 gt_start/gt_end），供 grounding 训练


def _npz_member_shape(path: str, key: str = "features") -> tuple | None:
    """只读 .npz 里某个数组的 **header** 拿 shape，不解压数据体。失败返回 None。

    `np.load(npz)["features"]` 会把整个数组解压出来——L×P×d 的 fp16 特征动辄几百 MB，
    而我们只要 shape。逐样本探测若走那条路，等于每次启动把数据集解压一遍
    （与 _peek_n_frames 的注意事项同源）。这里用 zipfile 打开成员流，只读开头
    那一百来字节的 .npy header：deflate 是流式解压，读多少解多少。
    """
    import zipfile
    try:
        with zipfile.ZipFile(path) as zf:
            name = key + ".npy"
            if name not in zf.namelist():
                return None
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
                else:
                    return None
                return tuple(shape)
    except Exception:
        return None


def _feat_shape(path: str, ref: str) -> tuple | None:
    """统一取特征 shape（.npy 走 mmap 只读 header；.npz 只读成员 header）。失败返回 None。"""
    if ref.endswith(".npy"):
        return tuple(np.load(path, mmap_mode="r").shape)
    if ref.endswith(".npz"):
        shape = _npz_member_shape(path)
        if shape is not None:
            return shape
        # 兜底：header 读不出来（异常压缩方式/旧文件）才退回解压路径，保证功能不回退
        z = np.load(path)
        try:
            return tuple(z["features"].shape)
        finally:
            z.close()
    return None


def _probe_feat_shapes(manifest: str, data_root: str = "", max_probe: int = 8) -> list | None:
    """扫 manifest 前若干条可读特征，返回它们的 shape 列表。manifest 打不开返回 None。

    **逐行独立 try** 是这里的关键：此前 infer_feat_dim 整个循环共用一个 try，于是
    manifest 里**第一条** feature_ref 不可达（路径拼错、data_root 与 build_manifest 的
    --data-root 不一致、该视频抽取失败）就直接 return None，后面成百上千条可读的特征
    一条都不看——外层 align_feat_dim 只打一句警告并沿用 YAML 里那个给 CPU smoke 用的
    feat_dim=768，模型就按错误维度建起来了。改成逐行 try/continue（与
    validate_dataset._infer_feat_dim 同口径），只有**全部**不可读才算推断失败。

    探测前 max_probe 条而非只看第一条：同一 manifest 里混进不同底座抽的特征是致命的
    数据错误，早发现一次胜过在第 N 步崩。取 shape 全程只读 header，不解压数据体。
    """
    shapes = []
    try:
        with open(manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ref = json.loads(line).get("feature_ref")
                except Exception:
                    continue                       # 单行 JSON 损坏不该让整次推断失败
                if not ref:
                    continue
                try:
                    shape = _feat_shape(os.path.join(data_root, ref), ref)
                except Exception:
                    continue                       # 该条不可读 → 看下一条
                if shape:
                    shapes.append(shape)
                    if len(shapes) >= max(1, int(max_probe)):
                        break
    except Exception:
        return None                                # manifest 本身打不开
    return shapes


def _one_value(shapes, axis: int, manifest: str, what: str, hint: str,
               min_ndim: int = 1) -> int | None:
    """从探测到的 shape 列表里取某一轴的唯一值；不一致即 raise（异源特征混用是致命错误）。

    min_ndim 拦掉秩不够的 shape：取 P（axis=-2）必须是 [L,P,d] 三维，若某个特征退化成
    二维 [L,d]，s[-2] 拿到的是帧数 L 而不是 P——那是个看起来很正常的错值。
    """
    vals = [int(s[axis]) for s in shapes if len(s) >= min_ndim]
    if not vals:
        return None
    if len(set(vals)) > 1:
        raise ValueError(
            f"manifest {manifest} 内{what}不一致: {sorted(set(vals))}（前 {len(vals)} 条探测）。{hint}")
    return vals[0]


def infer_feat_dim(manifest: str, data_root: str = "", max_probe: int = 8) -> int | None:
    """从 manifest 的可读特征推断 d（防止 cfg.feat_dim 与真实特征错配）。全不可读返回 None。"""
    shapes = _probe_feat_shapes(manifest, data_root, max_probe)
    if not shapes:
        return None
    return _one_value(
        shapes, -1, manifest, "特征维 d",
        "这说明特征目录里混了不同视觉塔/不同 --model 抽的特征，必须用同一底座重抽后重建 "
        "manifest；先跑 data_pipeline.src.validate_dataset --strict 定位到具体行")


def infer_feat_patches(manifest: str, data_root: str = "", max_probe: int = 8) -> int | None:
    """同上，推断每帧 patch 数 P（特征 shape[-2]）。全不可读或形状非 [L,P,d] 返回 None。

    P 与 d 一样由抽特征时的 --patches 唯一确定，不是建模自由度：合成兜底的
    LoaderConfig.feat_patches=4 与真实特征的 P=9 不一致时，兜底样本会与正常样本对不齐。
    """
    shapes = _probe_feat_shapes(manifest, data_root, max_probe)
    if not shapes:
        return None
    return _one_value(shapes, -2, manifest, "每帧 patch 数 P",
                      "必须用同一 --patches 重抽特征后重建 manifest", min_ndim=3)


def align_feat_dim(cfg, manifest: str | None, data_root: str = "", tag: str = "cfg",
                   strict: bool = True):
    """按真实特征把 cfg.feat_dim 校正到位（feature 模式且给了 manifest 时）。返回 cfg。

    feat_dim **不是建模自由度**：它由抽特征时用的视觉塔唯一确定（如 Qwen3-VL-4B=2560），
    模型侧只能跟随。此前只有"没给 --config"的分支会调 infer_feat_dim，YAML 带 model 段时
    直接短路，于是 configs/default.yaml 那个为 CPU smoke 设的 feat_dim=768 会原样建模型，
    一路撞到第 0 步 FeatureAdapter 的第一个 Linear 才炸成一句看不出所以然的
    "mat1 and mat2 shapes cannot be multiplied (9216x2560 and 768x384)"。

    因为正确值唯一，这里直接**校正**而不是报错——但必须出声，让日志能自证走了哪条路。

    strict=True（默认）：给了真实 manifest 却**一条特征都读不出来**时直接退出，而不是
    沿用 YAML 的兜底值继续训。旧行为只打一句警告，后果是整个 stage-1 用 feat_dim=768
    训完、把这个错误维度固化进 checkpoints/stage1/final 的 FeatureAdapter，直到 stage-2
    才因为和底座的 2560 对不上而退出——白烧一轮训练。读不出特征在真实训练里只有一个
    意思：manifest 的 feature_ref 与 --data-root 拼不出存在的文件，此时唯一正确的动作
    是停下来修数据。设 strict=False 可退回旧的"警告并沿用"行为（仅供 smoke）。
    """
    from dataclasses import replace                   # 泛型 replace，不引入对 models 的依赖
    if not manifest or getattr(cfg, "input_mode", "feature") != "feature":
        return cfg
    d = infer_feat_dim(manifest, data_root)
    if not d:
        msg = (f"[{tag}] 无法从 {manifest} 推断特征维：其中没有一条 feature_ref 能读出来。"
               f"最常见原因是 manifest 的 feature_ref 与 --data-root='{data_root}' 拼不出"
               f"存在的文件（build_manifest.py 的 --data-root 必须与训练脚本的 --data-root 一致），"
               f"或该 split 的特征还没抽。请先跑：\n"
               f"  python -m data_pipeline.src.validate_dataset "
               f"--manifest {manifest} --data-root {data_root or '.'} --strict")
        if strict:
            raise SystemExit(msg + f"\n[{tag}] 已中止：继续下去会用兜底 feat_dim={cfg.feat_dim} "
                                   f"训出一个维度错误的模型。")
        print(msg + f"\n[{tag}] 警告: 沿用 feat_dim={cfg.feat_dim}（strict=False）")
        return cfg
    if d != cfg.feat_dim:
        print(f"[{tag}] feat_dim 与真实特征不符：配置 {cfg.feat_dim} → 按特征校正为 {d}"
              f"（来源 {manifest}）")
        return replace(cfg, feat_dim=d)
    print(f"[{tag}] feat_dim={d} 与真实特征一致")
    return cfg


def _peek_n_frames(path: str) -> int | None:
    """只读文件头/小数组拿帧数，**不解压 features**。失败返回 None。

    .npy → mmap 只读 header；.npz → 读 timestamps（长度 L 的 float32，几十 KB），
    绝不碰 features（L×P×d，上百 MB，逐样本读一遍等于把数据集全载一次）。
    """
    try:
        if path.endswith(".npy"):
            return int(np.load(path, mmap_mode="r").shape[0])
        if path.endswith(".npz"):
            with np.load(path) as z:
                key = "timestamps" if "timestamps" in z.files else None
                return int(z[key].shape[0]) if key else None
    except Exception:
        return None
    return None


def sample_lengths(ds, lc: LoaderConfig) -> list[int] | None:
    """取每个样本的帧数，供长度分桶用。无法可靠获取时返回 None（调用方退回普通采样）。

    优先读 manifest 的 n_frames 字段（build_manifest.py 会写），缺失才去探文件头。
    返回值只是**分桶提示**，不参与任何计算：collate 仍按批内实际最大长度 padding，
    所以即使某条长度不准（如 __getitem__ 失败替换了别的行）也只影响显存效率，不影响正确性。
    """
    rows = getattr(ds, "rows", None)
    if not rows:                                   # 合成数据集：等长，分桶无意义
        return None
    cap = int(lc.max_frames)
    out, n_probe = [], 0
    for r in rows:
        n = r.get("n_frames")
        if n is None:
            ref = r.get("feature_ref")
            n = _peek_n_frames(os.path.join(lc.data_root, ref)) if ref else None
            n_probe += 1
        if n is None:
            return None                            # 有一条拿不到就整体放弃，避免半吊子分桶
        out.append(min(int(n), cap))               # dataset._subsample 会把超出的截到 cap
    if n_probe:
        print(f"[loaders] 长度分桶：manifest 无 n_frames，探测了 {n_probe} 个特征文件头。"
              f"在 build_manifest.py 中已会写入 n_frames，重建 manifest 可免去这步。")
    return out


def build_dataset(lc: LoaderConfig, tokenizer=None):
    """按配置构造 dataset（真实 manifest 或合成）。"""
    if lc.manifest:
        # 真实路径下 P 由抽特征的 --patches 唯一确定，和 feat_dim 同理不是建模自由度。
        # 它只影响 _fallback_sample 的形状，但那正是"全部真实样本不可用"时唯一进 collate
        # 的张量——P 用合成兜底值 4、真实特征是 9 的话，占位样本与正常样本在 collate 里
        # 按 P 维对不齐，报出的错会指向 collate 而不是真正的病灶。能推断就按真实值来。
        patches = lc.feat_patches
        inferred_p = infer_feat_patches(lc.manifest, lc.data_root) if lc.mode == "feature" else None
        if inferred_p and inferred_p != patches:
            print(f"[loaders] feat_patches 与真实特征不符：配置 {patches} → 按特征校正为 {inferred_p}")
            patches = inferred_p
        dcfg = DataConfig(manifest=lc.manifest, mode=lc.mode, max_frames=lc.max_frames,
                          max_text_len=lc.max_text_len, feat_dim=lc.feat_dim,
                          feat_patches=patches, feat_dtype=lc.feat_dtype)
        return VideoTemporalDataset(dcfg, tokenizer=tokenizer or ByteTokenizer(),
                                    data_root=lc.data_root)
    if lc.synth_grounding:
        return SyntheticGroundingDataset(n=lc.synth_n, L=lc.synth_frames, P=lc.feat_patches,
                                         d=lc.feat_dim)
    return SyntheticVideoDataset(n=lc.synth_n, L=lc.synth_frames, P=lc.feat_patches,
                                 d=lc.feat_dim)


def build_dataloader(lc: LoaderConfig, tokenizer=None):
    """构造 DataLoader（完整吞吐参数 + DDP 自动分片）。返回 (loader, dataset)。

    若传入的 tokenizer 有 pad_id 属性（如 HFTokenizer），自动覆盖 lc.pad_id 以对齐 collate 填充。
    DDP 环境下自动注入 DistributedSampler（shuffle 由 sampler 接管）。
    """
    ds = build_dataset(lc, tokenizer)
    pixel = (lc.mode == "pixel")
    # 用 `is not None` 而非 `or`：pad_id 合法取 0，`or` 会把 0 当假值吞掉、静默回退到
    # lc.pad_id（ByteTokenizer 下是 256），于是真正的 padding 位被填成一个有效 token id。
    _tok_pad = getattr(tokenizer, "pad_id", None)
    pad_id = _tok_pad if _tok_pad is not None else lc.pad_id

    # DDP：自动注入 DistributedSampler，各 rank 数据不重叠
    sampler = None
    shuffle = lc.shuffle
    # 分片信息走 dist_info()，它在进程组建立**之前**也能从 torchrun 的环境变量拿到
    # 正确的 (world, rank)。此前这里查的是 dist.is_initialized()，而训练脚本是
    # 先建 loader 再建 Trainer（进程组在 Trainer 里才建），于是 8 个 rank 全部拿到
    # world=1/rank=0——配上长度分桶固定 seed，8 张卡跑的是逐位相同的数据与梯度。
    world, rank = dist_info()
    if world > 1:
        print(f"[loaders] 分布式分片: world_size={world}, rank={rank}")

    # 长度分桶优先：它自带 DDP 分片（各 rank 取不重叠切片且步数相同），
    # 因此与 DistributedSampler 互斥——两者叠加会把数据切两次。
    batch_sampler = None
    if lc.length_bucketing:
        lengths = sample_lengths(ds, lc)
        if lengths is None:
            print("[loaders] 长度分桶已请求但拿不到样本长度（合成数据集或特征不可读），"
                  "本次退回普通采样；这不影响正确性，只是 padding 浪费照旧。")
        else:
            batch_sampler = LengthGroupedBatchSampler(
                lengths, batch_size=lc.batch_size, num_replicas=world, rank=rank,
                shuffle=lc.shuffle, drop_last=lc.drop_last, seed=lc.seed,
                bucket_multiplier=lc.bucket_multiplier)
    if batch_sampler is None and world > 1:
        # num_replicas/rank 显式传入：DistributedSampler 的默认值来自活跃进程组，
        # 而这里可能还没建组（见上方 dist_info 的说明），不传就会 raise。
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=lc.shuffle)
        shuffle = False  # sampler 接管 shuffle

    if batch_sampler is not None:
        # batch_sampler 与 batch_size/shuffle/sampler/drop_last 互斥（DataLoader 的约束）
        loader = DataLoader(
            ds, batch_sampler=batch_sampler, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory,
            persistent_workers=(lc.persistent_workers and lc.num_workers > 0),
            collate_fn=partial(collate_fn, pad_id=pad_id, pixel=pixel),
        )
        return loader, ds

    loader = DataLoader(
        ds, batch_size=lc.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=lc.num_workers,
        pin_memory=lc.pin_memory, drop_last=lc.drop_last,
        persistent_workers=(lc.persistent_workers and lc.num_workers > 0),
        collate_fn=partial(collate_fn, pad_id=pad_id, pixel=pixel),
    )
    return loader, ds


def cycle(loader):
    """无限迭代 loader（每轮重新洗牌）；供按 step 计数的 Trainer.fit 使用，天然处理 max_steps>一个 epoch。

    每轮对采样器调 set_epoch，保证跨 epoch 洗牌正确。DistributedSampler 与
    LengthGroupedBatchSampler 都实现了该方法；用 batch_sampler 时 loader.sampler
    不是我们注入的那个对象，所以两处都要查——只查 loader.sampler 会让分桶采样器
    每轮洗出**完全相同**的顺序（静默退化成单轮重复训练）。
    """
    epoch = 0
    while True:
        for s in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)
                break
        for batch in loader:
            yield batch
        epoch += 1
