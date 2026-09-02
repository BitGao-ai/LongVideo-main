"""Qwen3-VL 视觉塔离线抽帧级特征 → .npz 特征缓存（供 VideoTemporalDataset 消费）。

与 cst_ssm/integrations/qwen3_vl.py::Qwen3VLVisionFeatureExtractor 的关键区别：
**喂进视觉塔的帧来自 adaptive_sampler 的变步长采样结果**（真实 Δt），而非 processor
默认的均匀 fps——这是保证时间戳真实性的落地点（docs/02 命门①）。

缓存一致性：每个 .npz 内嵌采样/编码参数指纹 `sampler_fp`（theta/coarse_fps/dt_max/
max_frames/patches/model 等的哈希）与溯源 meta（strategy/duration/n_sampled/
event_density）。--skip-existing 会先比对旧缓存指纹，参数变化或缺指纹的旧缓存自动
失效重算，避免静默命中陈旧特征。

用法：
    python -m data_pipeline.src.extract_features \
        --manifest data/filtered.jsonl --model Qwen/Qwen3-VL-4B-Instruct \
        --out data/features --sampler adaptive --patches 9
无 GPU/transformers 时加 --dry-run：用合成特征替身跑通全链路（等价 scripts/prepare_features.py）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Optional

import numpy as np

from .adaptive_sampler import SamplerConfig, sample_video


# Qwen3VLConfig 的占位默认值。真实 Qwen3-VL-4B 视觉塔是 2560——这个常量**只**在
# --dry-run 且拿不到真实 config.json 时兜底，任何真实抽取路径都不会用到它。
_FALLBACK_OUT_HIDDEN = 3584


# --------------------------- 视觉塔封装 ---------------------------
class VisionTower:
    """启动即显式加载 Qwen3-VL 视觉塔（带进度日志），按给定帧索引抽 patch 特征 [n,P,d]。"""

    def __init__(self, model_name: str, patches: int, chunk_frames: int = 64,
                 dtype: str = "auto", device: str = "auto", max_frame_tokens: int = 256):
        self.model_name = model_name
        self.patches = patches
        self.chunk = chunk_frames
        self.dtype = dtype
        self.device = device
        self.max_frame_tokens = max_frame_tokens
        self._ex = None

    def load(self):
        """显式加载视觉塔并搬到合适设备；失败直接抛出（不进逐视频 try）。"""
        if self._ex is not None:
            return self._ex
        import torch
        from cst_ssm.integrations.qwen3_vl import Qwen3VLVisionFeatureExtractor

        if self.device != "auto":
            dev = self.device
        elif torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        _dt_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        if self.dtype != "auto":
            tdtype = _dt_map[self.dtype]
        else:
            tdtype = torch.float32 if dev == "cpu" else torch.bfloat16

        print(f"[extract] 开始加载视觉塔 {self.model_name}（device={dev}, dtype={tdtype}）；"
              f"若长时间无输出，多半在下载权重——可设 HF_ENDPOINT=https://hf-mirror.com "
              f"或 --model 指向本地目录", flush=True)
        t0 = time.time()
        # 复用主仓库集成实现，避免重复视觉塔代码
        self._ex = Qwen3VLVisionFeatureExtractor.from_pretrained(
            self.model_name, torch_dtype=tdtype)
        self._ex.max_frame_tokens = int(self.max_frame_tokens)
        self._ex.visual.to(dev)          # 抽特征只用视觉塔，只搬 visual 省内存
        print(f"[extract] 视觉塔加载完成，耗时 {time.time() - t0:.1f}s", flush=True)
        return self._ex

    def encode(self, video_path: str, frame_idx: np.ndarray, ts: np.ndarray):
        """真实抽取：按 frame_idx **精确解码那几帧**过 Qwen3-VL 视觉塔（decord 随机访问），
        每帧独立成一个时间戳（避免 temporal_patch_size 合并非均匀采样帧），并池化到目标 P。
        按 chunk 分块前向，避免长视频数百帧单批推理导致显存/内存峰值失控。
        返回 feats[L,P,d] float16 与 ts[L] float32，L==len(frame_idx)、ts 原样对齐采样时间戳。"""
        ex = self.load()
        frame_idx = np.asarray(frame_idx)
        ts = np.asarray(ts, dtype=np.float32)
        L = len(frame_idx)
        c = max(1, int(self.chunk))
        # 整段共用一个 decord.VideoReader：分块时每块重开一次会重复建帧索引（L=512/c=64 → 8 次）
        reader = ex.make_reader(video_path)
        if L <= c:                                     # 帧数少：单次前向即可
            feats, ts_out = ex.encode_video_at(
                video_path, frame_idx, ts, target_patches=self.patches, reader=reader)
        else:                                          # 分块：控制单批规模，逐块解码+前向再拼接
            parts = []
            n_chunk = (L + c - 1) // c
            t0 = time.time()
            for i, s in enumerate(range(0, L, c), 1):
                fi, tsi = frame_idx[s:s + c], ts[s:s + c]
                f, _ = ex.encode_video_at(
                    video_path, fi, tsi, target_patches=self.patches, reader=reader)
                parts.append(f)
                print(f"[extract]   分块编码 {i}/{n_chunk} 完成（已处理 {min(s + c, L)}/{L} 帧，"
                      f"累计 {time.time() - t0:.1f}s）", flush=True)
            feats, ts_out = np.concatenate(parts, 0), ts
        return feats.astype(np.float16), ts_out.astype(np.float32)


# --------------------------- dry-run 替身 ---------------------------
def _synthetic(frame_idx: np.ndarray, ts: np.ndarray, patches: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    L = len(frame_idx)
    return rng.standard_normal((L, patches, d)).astype(np.float16), ts.astype(np.float32)


def _out_hidden_from_config(model: str) -> Optional[int]:
    """不加载权重、不依赖 transformers，直接从本地模型目录的 config.json 读视觉塔输出维。

    给 --dry-run 用：合成特征的 d 必须与**真实底座**一致，否则 dry-run 跑通的那条链路
    和真实训练不是同一条（d=3584 的假特征训出的 stage-1，换成真实 2560 的特征就作废）。
    --model 指向本地目录时这一步是免费的；指向 HF 仓库名则拿不到，返回 None。
    """
    cfg_path = os.path.join(model, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        v = (cfg.get("vision_config") or {}).get("out_hidden_size")
        return int(v) if v else None
    except Exception:
        return None


def _resolve_out_hidden(args) -> int:
    """确定 --dry-run 合成特征的 d：显式 --out-hidden > 本地 config.json > 兜底常量。

    此前 --out-hidden 默认写死 3584（Qwen3VLConfig 的占位默认值），而 Qwen3-VL-4B 实测
    是 2560。于是 dry-run 产出的 .npz 是 3584 维，manifest 也按 3584 推断，stage-1 照着
    训完，直到 stage-2 用真底座建模型（2560）才对不上——白烧一轮。
    """
    if args.out_hidden is not None:
        return int(args.out_hidden)
    d = _out_hidden_from_config(args.model)
    if d:
        print(f"[extract] --dry-run: 未指定 --out-hidden，已从 {args.model}/config.json "
              f"读到视觉塔输出维 out_hidden_size={d}，合成特征按此维生成（与真实抽取一致）")
        return d
    print(f"[extract] ⚠ --dry-run: 未指定 --out-hidden，且 --model='{args.model}' 不是"
          f"能读到 config.json 的本地目录，只能用兜底 d={_FALLBACK_OUT_HIDDEN}。"
          f"这个值**未必**等于真实视觉塔的输出维（Qwen3-VL-4B 实测 2560），"
          f"用它跑出的 stage-1 检查点在换真特征后不可用。"
          f"如需与真实链路一致，请显式 --out-hidden <真实维度> 或让 --model 指向本地权重目录")
    return _FALLBACK_OUT_HIDDEN


# --------------------------- 缓存一致性 ---------------------------
def _sampler_fingerprint(scfg: SamplerConfig, patches: int, model: str, dry_run: bool,
                         max_frame_tokens: int = 0) -> str:
    """把所有影响特征内容的采样/编码参数压成稳定短哈希。
    任一参数变化 → 指纹变化 → --skip-existing 不再命中旧缓存。"""
    payload = {
        "strategy": scfg.strategy,
        "saliency": scfg.saliency,
        "saliency_accum": scfg.saliency_accum,   # 累积口径变了 → 采样结果变 → 缓存必须失效
        # 粗扫下采样算法版本。v2 = 块平均（抗混叠）；v1 = 最近邻点采样。
        # 算法本身不是配置项，但它会改变显著性估计进而改变选帧，所以必须进指纹，
        # 否则 --skip-existing 会把上一版算法产出的 .npz 当成有效缓存继续用。
        "downsample": "v2-block-mean",
        "theta": scfg.theta,
        "coarse_fps": scfg.coarse_fps,
        "dt_min_s": scfg.dt_min_s,
        "dt_max_s": scfg.dt_max_s,
        "max_frames": scfg.max_frames,
        "patches": patches,
        # 空间池化算法版本。v2 = 按 (h',w') 网格做 2D 自适应平均池化（P=9 → 真 3×3）；
        # v1 = 在展平 token 序列上 tensor_split 切条（切点不落行边界，无空间语义）。
        # 与 downsample 同理：算法变了但配置没变，不进指纹就会静默命中 v1 的旧缓存。
        "pool": "v2-spatial2d",
        # 每帧编码 token 上限：改变输入分辨率 → 改变特征内容
        "max_frame_tokens": 0 if dry_run else int(max_frame_tokens),
        "model": "dry-run" if dry_run else model,
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _seed_from_id(vid: str) -> int:
    """由 video_id 派生稳定随机种子（dry-run 合成数据用）。

    不用 abs(hash(vid))：CPython 对 str 的 hash 每个进程带一个随机盐（PYTHONHASHSEED），
    同一个 vid 在两次运行里得到不同种子，dry-run 产出的合成特征就不可复现——而 dry-run
    正是给外部复现者跑通流水线用的。sha1 与同文件的 _sampler_fingerprint 口径一致。
    """
    return int(hashlib.sha1(vid.encode()).hexdigest()[:8], 16)


def _cached_fingerprint(npz_path: str) -> Optional[str]:
    """读旧缓存内嵌指纹；无指纹键或文件损坏返回 None（视为失效、需重算）。"""
    try:
        with np.load(npz_path) as z:
            if "sampler_fp" in z.files:
                return str(z["sampler_fp"])
    except Exception:
        pass
    return None


def _save_npz(out_npz: str, feats, ts, fp: str, meta: dict):
    """落盘特征 + 时间戳 + 参数指纹 + 溯源 meta（meta 以扁平标量键存储，
    下游只按 features/timestamps 取键，额外键不破坏兼容）。

    meta_model / meta_out_hidden / meta_patches 是**溯源三件套**：sampler_fp 只能回答
    "参数变没变"，回答不了"这批特征是谁抽的"。把底座名与维度直接写进文件，
    train_stage2 才能在热启前指着具体文件说清楚"你的特征是 X 抽的、底座是 Y"。
    """
    np.savez(out_npz, features=feats, timestamps=ts,
             sampler_fp=np.array(fp),
             meta_model=np.array(str(meta.get("model", ""))),
             meta_out_hidden=np.int64(feats.shape[-1] if getattr(feats, "ndim", 0) >= 1 else 0),
             meta_patches=np.int64(feats.shape[-2] if getattr(feats, "ndim", 0) >= 3 else 0),
             meta_strategy=np.array(str(meta.get("strategy", ""))),
             meta_duration=np.float32(meta.get("duration", 0.0)),
             meta_n_sampled=np.int64(meta.get("n_sampled", len(ts))),
             meta_event_density=np.float32(meta.get("event_density", 0.0)),
             # 抽稀审计：n_raw > n_sampled 即 max_frames 咬过人、dt_max 契约已失效。
             # 存进 npz 才能事后逐视频复查，而不是只在当时的 stdout 里一闪而过。
             meta_n_selected_raw=np.int64(meta.get("n_selected_raw", len(ts))),
             meta_decimate_ratio=np.float32(meta.get("decimate_ratio", 1.0)),
             meta_dt_p50=np.float32(meta.get("dt_p50", 0.0)),
             meta_dt_p90=np.float32(meta.get("dt_p90", 0.0)),
             meta_dt_p99=np.float32(meta.get("dt_p99", 0.0)))


# ------------------------------ 主流程 ------------------------------
def run(args):
    scfg = SamplerConfig(strategy=args.sampler, theta=args.theta,
                         coarse_fps=args.coarse_fps, dt_max_s=args.dt_max,
                         saliency_accum=args.saliency_accum,
                         max_frames=args.max_frames)
    os.makedirs(args.out, exist_ok=True)
    out_hidden = _resolve_out_hidden(args) if args.dry_run else 0
    tower = None if args.dry_run else VisionTower(args.model, args.patches, args.chunk_frames,
                                                   dtype=args.dtype, device=args.device,
                                                   max_frame_tokens=args.max_frame_tokens)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.shard:                                             # "i/N" 多机分片
        i, n = map(int, args.shard.split("/"))
        rows = rows[i::n]

    if tower is not None:
        tower.load()                     # 进循环前显式加载：失败直接报错，不静默吞进逐视频异常
    if not rows:
        print("[extract] manifest 为空（或分片后为空），无视频可处理")
        return
    print(f"[extract] manifest 共 {len(rows)} 个视频，开始逐个抽取（首个视频含 decord 采样+视觉塔前向，可能耗时较久）", flush=True)

    fp = _sampler_fingerprint(scfg, args.patches, args.model, args.dry_run,
                              args.max_frame_tokens)
    print(f"[extract] 参数指纹 sampler_fp={fp}（strategy={scfg.strategy}, theta={scfg.theta}, "
          f"coarse_fps={scfg.coarse_fps}, dt_max={scfg.dt_max_s}, max_frames={scfg.max_frames}, "
          f"patches={args.patches}, max_frame_tokens={args.max_frame_tokens}）；旧缓存指纹不一致将被重算")

    ok, fail, stale = 0, 0, 0
    fail_log = open(os.path.join(args.out, "extract_failed.jsonl"), "a")
    try:
        for idx, r in enumerate(rows, 1):
            vid = r["video_id"]
            out_npz = os.path.join(args.out, f"{vid}.npz")
            if args.skip_existing and os.path.exists(out_npz):
                old_fp = _cached_fingerprint(out_npz)
                if old_fp == fp:
                    continue
                stale += 1
                print(f"[extract] ({idx}/{len(rows)}) {vid}: 旧缓存指纹 "
                      f"{old_fp or '缺失/损坏'} != 当前 {fp}，重新抽取", flush=True)
            try:
                t0 = time.time()
                if args.dry_run:
                    # 无视频时合成"变步长时间戳"：直接造可变 Δt
                    rng = np.random.default_rng(_seed_from_id(vid))
                    L = int(rng.integers(20, args.max_frames))
                    ts = np.cumsum(rng.uniform(scfg.dt_min_s, scfg.dt_max_s, L)).astype(np.float32)
                    feats, ts = _synthetic(np.arange(L), ts, args.patches, out_hidden, seed=L)
                    meta = dict(duration=float(ts[-1]), n_sampled=L, strategy=args.sampler,
                                event_density=round(L/float(ts[-1]), 4), model="dry-run")
                else:
                    video_path = r.get("video_path") or os.path.join(args.video_root, r.get("rel_path", vid + ".mp4"))
                    print(f"[extract] ({idx}/{len(rows)}) {vid}: 开始变步长采样 → {video_path}", flush=True)
                    frame_idx, ts, meta = sample_video(video_path, scfg)
                    print(f"[extract] ({idx}/{len(rows)}) {vid}: 采样完成 L={len(frame_idx)}（{time.time()-t0:.1f}s），开始解码取帧+视觉塔编码", flush=True)
                    feats, ts = tower.encode(video_path, frame_idx, ts)
                    meta["model"] = args.model
                    print(f"[extract] ({idx}/{len(rows)}) {vid}: 编码完成（累计 {time.time()-t0:.1f}s）", flush=True)
                _save_npz(out_npz, feats, ts, fp, meta)
                ok += 1
                print(f"[extract] ({idx}/{len(rows)}) {vid}: 已保存 → {out_npz}（L={len(ts)}, ρ={meta['event_density']}，总耗时 {time.time()-t0:.1f}s）", flush=True)
                if ok % 50 == 0:
                    print(f"[extract] 进度 {ok}/{len(rows)}（失败 {fail}）", flush=True)
            except Exception as e:                                 # 单视频失败隔离
                fail += 1
                fail_log.write(json.dumps({"video_id": vid, "error": str(e)}, ensure_ascii=False) + "\n")
                fail_log.flush()
                if ok == 0 and fail <= 3:
                    print(f"[extract] 警告：前几个视频连续失败（最近：{vid}: {e}）；"
                          f"请检查 extract_failed.jsonl，确认非环境/路径问题", flush=True)
    finally:
        # 用 finally：KeyboardInterrupt / 磁盘写满等非 Exception 退出路径下，
        # 上面 flush 过的行虽已落盘，但句柄不关会在长跑批处理里累积。
        fail_log.close()
    print(f"[extract] 完成 {ok} 个 .npz → {args.out}；失败 {fail}（见 extract_failed.jsonl）"
          + (f"；其中 {stale} 个因参数指纹变化重算" if stale else ""))
    print(f"[extract] 提醒：规模化训练前用 npz_to_npy.py 转 .npy+.ts.npy 以启用真 mmap")


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL 视觉塔离线抽特征（变步长采样）")
    ap.add_argument("--manifest", required=True, help="筛选通过清单 jsonl（含 video_id / video_path）")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out-hidden", type=int, default=None,
                    help="仅 --dry-run 合成特征用的维度 d；真实抽取维度由视觉塔决定。"
                         "缺省时自动从 --model 指向的本地目录 config.json 读 "
                         "vision_config.out_hidden_size（Qwen3-VL-4B=2560），读不到才退回 "
                         f"{_FALLBACK_OUT_HIDDEN} 并告警——那个值只是 Qwen3VLConfig 的占位默认值")
    ap.add_argument("--patches", type=int, default=9, help="空间池化目标 P：1/9/64")
    ap.add_argument("--max-frame-tokens", type=int, default=256,
                    help="每帧送进视觉塔的合并 token 数上限（约束输入分辨率）。默认 256≈512×512，"
                         "对 P<=64 的池化目标绰绰有余；0=不限制（按原分辨率编码，慢数倍且几乎无收益）。"
                         "改动会使参数指纹变化、旧缓存自动失效重算")
    ap.add_argument("--sampler", default="adaptive", choices=["uniform", "adaptive", "scene"])
    ap.add_argument("--saliency-accum", default="frame_diff", choices=["frame_diff", "anchor_diff"],
                    help="显著性累积口径（见 adaptive_sampler.SamplerConfig）；改动会使参数指纹变化、"
                         "旧缓存自动失效重算")
    ap.add_argument("--theta", type=float, default=0.12)
    ap.add_argument("--coarse-fps", type=float, default=4.0)
    ap.add_argument("--dt-max", type=float, default=2.0,
                    help="静止段兜底的最大帧间隔（秒）。直接决定定位精度地板 δ/4："
                         "2.0→0.5s，4.0→1.0s。1 小时视频 @2.0 约需 1550 帧")
    ap.add_argument("--max-frames", type=int, default=8192,
                    help="采样帧数**上限**（不是目标：实际帧数由 theta/dt-max 决定）。"
                         "定位为防御异常输入的保险丝：dt-max 是速率契约、它是数量契约，"
                         "二者仅在 时长 ≤ max-frames×dt-max 时可同时成立（8192×2.0=4.5 小时）。"
                         "必须与 DataConfig.max_frames 一致，否则训练侧会再抽稀一次。"
                         "上限一旦生效，dt-max 的间隔契约即失效，届时采样器会显式告警")
    ap.add_argument("--chunk-frames", type=int, default=64,
                    help="视觉塔单次前向的帧数上限（分块编码）；MPS/CPU 建议调小到 16~32")
    ap.add_argument("--video-root", default="", help="rel_path 的根")
    ap.add_argument("--shard", default=None, help='多机分片 "i/N"')
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"],
                    help="视觉塔权重精度；auto=GPU/MPS 用 bf16、CPU 用 fp32")
    ap.add_argument("--device", default="auto", help="cuda / mps / cpu；auto 自动探测")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False,
                    help="已存在且参数指纹一致的 .npz 跳过重抽（默认关闭，即强制全量重抽；"
                         "大规模增量运行可显式加 --skip-existing；--no-skip-existing 可显式关闭）")
    ap.add_argument("--dry-run", action="store_true", help="无 GPU/transformers 时用合成特征跑通")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
