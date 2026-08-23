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
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .schema import DataConfig
from .bucketing import LengthGroupedBatchSampler
from .dataset import (VideoTemporalDataset, SyntheticVideoDataset,
                      SyntheticGroundingDataset, ByteTokenizer)
from .collate import collate_fn


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


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
    # 合成兜底
    synth_n: int = 128
    synth_frames: int = 32
    synth_grounding: bool = False         # True：合成定位数据（带 gt_start/gt_end），供 grounding 训练


def infer_feat_dim(manifest: str, data_root: str = "") -> int | None:
    """从 manifest 首个可读特征推断 d（防止 cfg.feat_dim 与真实特征错配）。失败返回 None。"""
    try:
        with open(manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                ref = json.loads(line).get("feature_ref")
                if not ref:
                    continue
                path = os.path.join(data_root, ref)
                if ref.endswith(".npy"):
                    return int(np.load(path, mmap_mode="r").shape[-1])
                if ref.endswith(".npz"):
                    z = np.load(path)
                    try:
                        return int(z["features"].shape[-1])
                    finally:
                        z.close()
    except Exception:
        return None
    return None


def align_feat_dim(cfg, manifest: str | None, data_root: str = "", tag: str = "cfg"):
    """按真实特征把 cfg.feat_dim 校正到位（feature 模式且给了 manifest 时）。返回 cfg。

    feat_dim **不是建模自由度**：它由抽特征时用的视觉塔唯一确定（如 Qwen3-VL-4B=2560），
    模型侧只能跟随。此前只有"没给 --config"的分支会调 infer_feat_dim，YAML 带 model 段时
    直接短路，于是 configs/default.yaml 那个为 CPU smoke 设的 feat_dim=768 会原样建模型，
    一路撞到第 0 步 FeatureAdapter 的第一个 Linear 才炸成一句看不出所以然的
    "mat1 and mat2 shapes cannot be multiplied (9216x2560 and 768x384)"。

    因为正确值唯一，这里直接**校正**而不是报错——但必须出声，让日志能自证走了哪条路。
    """
    from dataclasses import replace                   # 泛型 replace，不引入对 models 的依赖
    if not manifest or getattr(cfg, "input_mode", "feature") != "feature":
        return cfg
    d = infer_feat_dim(manifest, data_root)
    if not d:
        print(f"[{tag}] 警告: 无法从 {manifest} 推断特征维（文件不可读或无 feature_ref），"
              f"沿用 feat_dim={cfg.feat_dim}；若与真实特征不符会在首个前向报形状错误")
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
        dcfg = DataConfig(manifest=lc.manifest, mode=lc.mode, max_frames=lc.max_frames,
                          max_text_len=lc.max_text_len, feat_dim=lc.feat_dim,
                          feat_patches=lc.feat_patches)
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
    world, rank = (dist.get_world_size(), dist.get_rank()) if _is_distributed() else (1, 0)

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
    if batch_sampler is None and _is_distributed():
        sampler = DistributedSampler(ds, shuffle=lc.shuffle)
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
