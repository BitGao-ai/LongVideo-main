"""Qwen3-VL 视觉塔离线抽帧级特征 → .npz 特征缓存（供 VideoTemporalDataset 消费）。

与 cst_ssm/integrations/qwen3_vl.py::Qwen3VLVisionFeatureExtractor 的关键区别：
**喂进视觉塔的帧来自 adaptive_sampler 的变步长采样结果**（真实 Δt），而非 processor
默认的均匀 fps——这是保证时间戳真实性的落地点（docs/02 命门①）。

用法：
    python -m data_pipeline.src.extract_features \
        --manifest data/filtered.jsonl --model Qwen/Qwen3-VL-4B-Instruct \
        --out data/features --sampler adaptive --patches 9
无 GPU/transformers 时加 --dry-run：用合成特征替身跑通全链路（等价 scripts/prepare_features.py）。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np

from .adaptive_sampler import SamplerConfig, sample_video


# --------------------------- 视觉塔封装 ---------------------------
class VisionTower:
    """惰性加载 Qwen3-VL 视觉塔，按给定帧索引抽 patch 特征 [n,P,d]。"""

    def __init__(self, model_name: str, patches: int, chunk_frames: int = 64):
        self.model_name = model_name
        self.patches = patches
        self.chunk = chunk_frames
        self._ex = None

    def _lazy(self):
        if self._ex is None:
            # 复用主仓库集成实现，避免重复视觉塔代码
            from cst_ssm.integrations.qwen3_vl import Qwen3VLVisionFeatureExtractor
            self._ex = Qwen3VLVisionFeatureExtractor.from_pretrained(self.model_name)
        return self._ex

    def encode(self, video_path: str, frame_idx: np.ndarray, ts: np.ndarray):
        """真实抽取：按 frame_idx **精确解码那几帧**过 Qwen3-VL 视觉塔（decord 随机访问），
        每帧独立成一个时间戳（避免 temporal_patch_size 合并非均匀采样帧），并池化到目标 P。
        返回 feats[L,P,d] float16 与 ts[L] float32，L==len(frame_idx)、ts 原样对齐采样时间戳。"""
        ex = self._lazy()
        feats, ts_out = ex.encode_video_at(
            video_path, frame_idx, ts, target_patches=self.patches)   # [L,P,d], [L]
        return feats.astype(np.float16), ts_out.astype(np.float32)


# --------------------------- dry-run 替身 ---------------------------
def _synthetic(frame_idx: np.ndarray, ts: np.ndarray, patches: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    L = len(frame_idx)
    return rng.standard_normal((L, patches, d)).astype(np.float16), ts.astype(np.float32)


# ------------------------------ 主流程 ------------------------------
def run(args):
    scfg = SamplerConfig(strategy=args.sampler, theta=args.theta,
                         coarse_fps=args.coarse_fps, dt_max_s=args.dt_max,
                         max_frames=args.max_frames)
    os.makedirs(args.out, exist_ok=True)
    tower = None if args.dry_run else VisionTower(args.model, args.patches, args.chunk_frames)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.shard:                                             # "i/N" 多机分片
        i, n = map(int, args.shard.split("/"))
        rows = rows[i::n]

    ok, fail = 0, 0
    fail_log = open(os.path.join(args.out, "extract_failed.jsonl"), "a")
    for r in rows:
        vid = r["video_id"]
        out_npz = os.path.join(args.out, f"{vid}.npz")
        if args.skip_existing and os.path.exists(out_npz):
            continue
        try:
            if args.dry_run:
                # 无视频时合成"变步长时间戳"：直接造可变 Δt
                rng = np.random.default_rng(abs(hash(vid)) % (2**32))
                L = int(rng.integers(20, args.max_frames))
                ts = np.cumsum(rng.uniform(scfg.dt_min_s, scfg.dt_max_s, L)).astype(np.float32)
                feats, ts = _synthetic(np.arange(L), ts, args.patches, args.out_hidden, seed=L)
                meta = dict(duration=float(ts[-1]), n_sampled=L, event_density=round(L/float(ts[-1]), 4))
            else:
                video_path = r.get("video_path") or os.path.join(args.video_root, r.get("rel_path", vid + ".mp4"))
                frame_idx, ts, meta = sample_video(video_path, scfg)
                feats, ts = tower.encode(video_path, frame_idx, ts)
            np.savez(out_npz, features=feats, timestamps=ts)
            ok += 1
            if ok % 200 == 0:
                print(f"[extract] 已完成 {ok}（最近 {vid}: L={len(ts)}, ρ={meta['event_density']}）")
        except Exception as e:                                 # 单视频失败隔离
            fail += 1
            fail_log.write(json.dumps({"video_id": vid, "error": str(e)}, ensure_ascii=False) + "\n")
    fail_log.close()
    print(f"[extract] 完成 {ok} 个 .npz → {args.out}；失败 {fail}（见 extract_failed.jsonl）")
    print(f"[extract] 提醒：规模化训练前用 npz_to_npy.py 转 .npy+.ts.npy 以启用真 mmap")


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL 视觉塔离线抽特征（变步长采样）")
    ap.add_argument("--manifest", required=True, help="筛选通过清单 jsonl（含 video_id / video_path）")
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out-hidden", type=int, default=3584, help="视觉塔输出维 == feat_dim")
    ap.add_argument("--patches", type=int, default=9, help="空间池化目标 P：1/9/64")
    ap.add_argument("--sampler", default="adaptive", choices=["uniform", "adaptive", "scene"])
    ap.add_argument("--theta", type=float, default=0.12)
    ap.add_argument("--coarse-fps", type=float, default=4.0)
    ap.add_argument("--dt-max", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=512)
    ap.add_argument("--chunk-frames", type=int, default=64)
    ap.add_argument("--video-root", default="", help="rel_path 的根")
    ap.add_argument("--shard", default=None, help='多机分片 "i/N"')
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--dry-run", action="store_true", help="无 GPU/transformers 时用合成特征跑通")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
