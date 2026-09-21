#!/usr/bin/env python3
"""Demo data preparation: feature caches plus a manifest.

Usage:
    python scripts/prepare_features.py --out data_demo --n 8 --frames 40
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_demo")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--patches", type=int, default=4)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    feat_dir = os.path.join(args.out, "features")
    man_dir = os.path.join(args.out, "manifests")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(man_dir, exist_ok=True)

    rows = []
    for i in range(args.n):
        vid = f"vid{i:04d}"
        L = args.frames + int(rng.integers(-5, 6))
        feats = rng.standard_normal((L, args.patches, args.dim)).astype(np.float16)
        gaps = rng.uniform(0.1, 0.5, size=L).astype(np.float32)
        ts = np.cumsum(gaps)
        np.savez(os.path.join(feat_dir, f"{vid}.npz"), features=feats, timestamps=ts)
        dur = float(ts[-1])
        gs = float(rng.uniform(0, dur - 2)); ge = min(dur, gs + float(rng.uniform(1, 8)))
        rows.append(dict(
            video_id=vid, feature_ref=f"features/{vid}.npz",
            prompt="<video> When does the key event happen?",
            answer=f"from {gs:.2f}s to {ge:.2f}s",
            task_type="grounding", gt_start=round(gs, 3), gt_end=round(ge, 3),
            duration=round(dur, 3), split="train",
        ))

    man_path = os.path.join(man_dir, "train.jsonl")
    with open(man_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[prepare] {args.n} .npz -> {feat_dir}")
    print(f"[prepare] manifest {len(rows)} rows -> {man_path}")


if __name__ == "__main__":
    main()
