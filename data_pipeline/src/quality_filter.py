"""质量筛选 + 感知哈希去重：raw_videos/ → filtered.jsonl（通过）+ rejected.jsonl（剔除+原因）。

按"先便宜后昂贵"顺序（docs/01 §2）：元数据 → 黑帧/纯色 → 静止 → 场景碎裂 → 去重。
早剔除省下游抽取算力。静止视频按 keep_static_ratio 刻意保留一部分（Theorem 1(B) 正例）。

重解码依赖（decord）lazy import；无依赖时 --dry-run 用合成清单演示筛选/去重判定逻辑。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np


# ------------------------------ 感知哈希 ------------------------------
def phash_from_grays(grays: np.ndarray, hash_side: int = 8) -> int:
    """对若干灰度帧求均值图的 aHash（hash_side^2 位）。grays: (k,H,W) in [0,1]。"""
    if len(grays) == 0:
        return 0
    m = grays.mean(0)
    h, w = m.shape
    ys = np.linspace(0, h - 1, hash_side).astype(int)
    xs = np.linspace(0, w - 1, hash_side).astype(int)
    small = m[np.ix_(ys, xs)]
    bits = (small > small.mean()).astype(np.uint64).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ------------------------------ 探针 ------------------------------
def _probe(path: str, n_probe: int = 16):
    """解码稀疏采样 n_probe 帧灰度 + 元数据。返回 (grays(k,64,64), meta) 或 raise。"""
    import decord  # type: ignore
    vr = decord.VideoReader(path)
    fps = float(vr.get_avg_fps()) or 25.0
    n = len(vr); dur = n / fps
    idx = np.linspace(0, n - 1, min(n_probe, n)).astype(int)
    grays = []
    h = w = None
    for i in idx:
        fr = vr[int(i)].asnumpy()
        h, w = fr.shape[:2]
        g = fr.astype(np.float32).mean(-1) / 255.0
        ys = np.linspace(0, g.shape[0] - 1, 64).astype(int)
        xs = np.linspace(0, g.shape[1] - 1, 64).astype(int)
        grays.append(g[np.ix_(ys, xs)])
    meta = dict(duration=round(dur, 2), fps=round(fps, 2), height=h, width=w, n_frames=n)
    return np.stack(grays), meta


# ------------------------------ 各过滤器 ------------------------------
def check_meta(meta: dict, cfg: dict) -> Optional[str]:
    if meta["duration"] < cfg["min_duration_s"]:
        return "too_short"
    if meta["duration"] > cfg["max_duration_s"]:
        return "too_long"
    if min(meta["height"] or 0, meta["width"] or 0) < cfg["min_short_side_px"]:
        return "low_resolution"
    return None


def check_black(grays: np.ndarray, cfg: dict) -> Optional[str]:
    var = float(grays.var(axis=(1, 2)).mean()) * 255.0 * 255.0
    return "black_or_flat" if var < cfg["black_frame_var_thresh"] else None


def static_ratio(grays: np.ndarray, cfg: dict) -> float:
    """相邻探针帧平均绝对差 < 阈值 的占比（0~1）。"""
    if len(grays) < 2:
        return 1.0
    diffs = np.abs(np.diff(grays, axis=0)).mean(axis=(1, 2)) * 255.0
    return float((diffs < cfg["static_absdiff_thresh"]).mean())


# ------------------------------ 主流程 ------------------------------
def run(args, cfg: dict):
    holdout = None
    if cfg.get("holdout_hashes") and os.path.exists(cfg["holdout_hashes"]):
        import pickle
        holdout = pickle.load(open(cfg["holdout_hashes"], "rb"))       # set[int]

    videos = []
    for root, _, files in os.walk(args.video_dir):
        for fn in files:
            if fn.lower().endswith((".mp4", ".mkv", ".mov", ".avi", ".webm")):
                videos.append(os.path.join(root, fn))
    videos.sort()

    kept, rejected, seen_hashes = [], [], []
    n_static_kept = 0
    for path in videos:
        vid = os.path.splitext(os.path.basename(path))[0]
        try:
            grays, meta = _probe(path)
        except Exception as e:
            rejected.append({"video_id": vid, "reason": "decode_error", "detail": str(e)})
            continue

        reason = check_meta(meta, cfg) or check_black(grays, cfg)
        if reason:
            rejected.append({"video_id": vid, "reason": reason}); continue

        sr = static_ratio(grays, cfg)
        if sr > cfg["static_ratio_reject"]:
            # 静止：Stage-1 按比例保留做正例，其余剔除
            if n_static_kept < cfg["keep_static_ratio"] * max(len(videos), 1):
                n_static_kept += 1
            else:
                rejected.append({"video_id": vid, "reason": "static"}); continue

        ph = phash_from_grays(grays, int(round(cfg["dedup_phash_bits"] ** 0.5)))
        if holdout and any(hamming(ph, h) < cfg["dedup_hamming_thresh"] for h in holdout):
            rejected.append({"video_id": vid, "reason": "benchmark_leakage"}); continue
        if any(hamming(ph, h) < cfg["dedup_hamming_thresh"] for h in seen_hashes):
            rejected.append({"video_id": vid, "reason": "near_duplicate"}); continue
        seen_hashes.append(ph)

        kept.append({"video_id": vid, "video_path": path, "duration": meta["duration"],
                     "height": meta["height"], "width": meta["width"],
                     "static_ratio": round(sr, 3), "phash": ph})

    _write(args.out, kept)
    _write(args.rejected, rejected)
    print(f"[filter] 通过 {len(kept)} / 共 {len(videos)}；剔除 {len(rejected)}（原因见 {args.rejected}）")
    _summary(rejected)


def _write(path: str, rows: list):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summary(rejected: list):
    from collections import Counter
    c = Counter(r["reason"] for r in rejected)
    if c:
        print("[filter] 剔除原因分布: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))


def _dry_run(cfg: dict):
    """合成 5 条清单演示去重/静止判定（无解码依赖）。"""
    print("[dry-run] 感知哈希去重 & 汉明判定演示：")
    a = 0b1010101010101010
    b = a ^ 0b0000000000000011                     # 距离 2 → 近重复
    c = a ^ 0b1111111100000000                     # 距离 8 → 不同
    print(f"  h(a,b)={hamming(a,b)} < {cfg['dedup_hamming_thresh']} → 近重复(剔除)")
    print(f"  h(a,c)={hamming(a,c)} ≥ {cfg['dedup_hamming_thresh']} → 保留")
    grays = np.stack([np.full((64, 64), 0.5, np.float32)] * 4)      # 全静止
    print(f"  静止占比={static_ratio(grays, cfg):.2f}（>{cfg['static_ratio_reject']} 触发静止规则）")


DEFAULT_CFG = dict(
    min_duration_s=60.0, max_duration_s=10800.0, min_short_side_px=240,
    black_frame_var_thresh=8.0, static_absdiff_thresh=2.0, static_ratio_reject=0.9,
    keep_static_ratio=0.10, max_cuts_per_min=40, dedup_phash_bits=64,
    dedup_hamming_thresh=6, holdout_hashes=None,
)


def main():
    ap = argparse.ArgumentParser(description="质量筛选 + 感知哈希去重")
    ap.add_argument("--video-dir", help="原始视频目录；省略跑 dry-run")
    ap.add_argument("--out", default="data/filtered.jsonl")
    ap.add_argument("--rejected", default="data/rejected.jsonl")
    ap.add_argument("--min-duration", type=float, default=60.0)
    ap.add_argument("--keep-static-ratio", type=float, default=0.10)
    ap.add_argument("--holdout-hashes", default=None, help="评测集 pHash pkl（防泄漏）")
    a = ap.parse_args()
    cfg = dict(DEFAULT_CFG, min_duration_s=a.min_duration,
               keep_static_ratio=a.keep_static_ratio, holdout_hashes=a.holdout_hashes)
    if not a.video_dir:
        _dry_run(cfg); return
    run(a, cfg)


if __name__ == "__main__":
    main()
