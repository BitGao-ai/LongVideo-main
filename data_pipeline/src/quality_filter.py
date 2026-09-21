"""Quality filtering with perceptual-hash dedup: raw videos to filtered/rejected lists."""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np


def phash_from_grays(grays: np.ndarray, hash_side: int = 8) -> int:
    """Average-hash over mean gray frame; grays (k,H,W) in [0,1]."""
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


def _probe(path: str, n_probe: int = 16):
    """Decode sparse probe frames plus metadata; raises on failure."""
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
    """Fraction of adjacent probe pairs below the static threshold (0-1)."""
    if len(grays) < 2:
        return 1.0
    diffs = np.abs(np.diff(grays, axis=0)).mean(axis=(1, 2)) * 255.0
    return float((diffs < cfg["static_absdiff_thresh"]).mean())


def run(args, cfg: dict):
    holdout = None
    if cfg.get("holdout_hashes") and os.path.exists(cfg["holdout_hashes"]):
        import pickle
        holdout = pickle.load(open(cfg["holdout_hashes"], "rb"))

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
    print(f"[filter] kept {len(kept)} / {len(videos)}; rejected {len(rejected)} (see {args.rejected})")
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
        print("[filter] rejection reasons: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))


def _dry_run(cfg: dict):
    print("[dry-run] perceptual-hash dedup demo:")
    a = 0b1010101010101010
    b = a ^ 0b0000000000000011
    c = a ^ 0b1111111100000000
    print(f"  h(a,b)={hamming(a,b)} < {cfg['dedup_hamming_thresh']} -> near-duplicate (reject)")
    print(f"  h(a,c)={hamming(a,c)} >= {cfg['dedup_hamming_thresh']} -> keep")
    grays = np.stack([np.full((64, 64), 0.5, np.float32)] * 4)
    print(f"  static_ratio={static_ratio(grays, cfg):.2f} (>{cfg['static_ratio_reject']} triggers static rule)")


DEFAULT_CFG = dict(
    min_duration_s=60.0, max_duration_s=10800.0, min_short_side_px=240,
    black_frame_var_thresh=8.0, static_absdiff_thresh=2.0, static_ratio_reject=0.9,
    keep_static_ratio=0.10, max_cuts_per_min=40, dedup_phash_bits=64,
    dedup_hamming_thresh=6, holdout_hashes=None,
)


def main():
    ap = argparse.ArgumentParser(description="Quality filtering + perceptual-hash dedup")
    ap.add_argument("--video-dir")
    ap.add_argument("--out", default="data/filtered.jsonl")
    ap.add_argument("--rejected", default="data/rejected.jsonl")
    ap.add_argument("--min-duration", type=float, default=60.0)
    ap.add_argument("--keep-static-ratio", type=float, default=0.10)
    ap.add_argument("--holdout-hashes", default=None)
    a = ap.parse_args()
    cfg = dict(DEFAULT_CFG, min_duration_s=a.min_duration,
               keep_static_ratio=a.keep_static_ratio, holdout_hashes=a.holdout_hashes)
    if not a.video_dir:
        _dry_run(cfg); return
    run(a, cfg)


if __name__ == "__main__":
    main()
