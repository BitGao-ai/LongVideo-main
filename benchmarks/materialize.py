#!/usr/bin/env python3
"""Materialize CSG/VFR input grids into feature caches plus an inference manifest.

Usage:
    python materialize.py --manifest csg_delta2.jsonl --video-root /data/charades \
        --model Qwen/Qwen3-VL-4B-Instruct --patches 9 --out csg_feats --out-manifest csg_infer.jsonl
    python materialize.py --manifest csg_demo.jsonl --dry-run --out /tmp/f --out-manifest /tmp/csg_infer.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from common import read_jsonl, write_jsonl             # noqa: E402


def safe_id(query_id: str) -> str:
    """Map a query id to a filename-safe string."""
    return re.sub(r"[^0-9A-Za-z._-]", "_", str(query_id))


def grid_to_frame_idx(grid, fps: float, n_frames: int | None = None) -> list:
    """Grid timestamps in seconds to frame indices round(t*fps), clamped to range."""
    idx = [int(round(float(t) * float(fps))) for t in grid]
    if n_frames is not None:
        idx = [min(max(0, i), n_frames - 1) for i in idx]
    else:
        idx = [max(0, i) for i in idx]
    return idx


def infer_row(r: dict, feature_ref: str) -> dict:
    """Build an inference manifest row from a CSG/VFR row plus a feature reference."""
    row = dict(
        video_id=r["video_id"], query_id=r["query_id"], feature_ref=feature_ref,
        prompt=f'<video> {r.get("query", "")}'.strip(),
        answer=f'from {float(r["gt_start"]):.3f}s to {float(r["gt_end"]):.3f}s',
        task_type="grounding",
        gt_start=round(float(r["gt_start"]), 3), gt_end=round(float(r["gt_end"]), 3),
        duration=round(float(r["duration"]), 3), split=r.get("split", "test"),
    )
    for k in ("delta", "floor", "level", "cv"):
        if k in r:
            row[k] = r[k]
    return row


def resolve_video_path(video_id: str, root: str, ext: str, vmap: dict | None):
    if vmap and video_id in vmap:
        return vmap[video_id]
    if root:
        return os.path.join(root, f"{video_id}{ext}")
    return None


def make_decord_backend():
    """Shared decord reader plus metadata accessor with a VideoReader cache."""
    cache: dict = {}

    def _vr(path):
        vr = cache.get(path)
        if vr is None:
            import decord  # type: ignore
            decord.bridge.set_bridge("native")
            vr = cache[path] = decord.VideoReader(path)
        return vr

    def reader(path, frame_idx):
        vr = _vr(path)
        batch = vr.get_batch([int(i) for i in frame_idx]).asnumpy()
        return [batch[i] for i in range(batch.shape[0])]

    def meta(path):
        vr = _vr(path)
        return len(vr), float(vr.get_avg_fps() or 0.0)

    return reader, meta


def save_features(out_dir: str, sid: str, feats: np.ndarray, ts: np.ndarray, fmt: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = np.asarray(ts, dtype=np.float32)
    if fmt == "npy":
        np.save(os.path.join(out_dir, sid + ".npy"), feats)
        np.save(os.path.join(out_dir, sid + ".ts.npy"), ts)
        return sid + ".npy"
    np.savez(os.path.join(out_dir, sid + ".npz"), features=feats, timestamps=ts)
    return sid + ".npz"


def run(args):
    rows = read_jsonl(args.manifest)
    vmap = None
    if args.video_map:
        vmap = {r["video_id"]: r["path"] for r in read_jsonl(args.video_map)}

    ex = None
    reader = meta = None
    if not args.dry_run:
        sys.path.insert(0, os.path.dirname(_HERE))
        from cst_ssm.integrations.qwen3_vl import Qwen3VLVisionFeatureExtractor
        ex = Qwen3VLVisionFeatureExtractor.from_pretrained(args.model)
        reader, meta = make_decord_backend()

    os.makedirs(os.path.dirname(args.out_manifest) or ".", exist_ok=True)
    grid_cache: dict = {}
    out_rows, n_feat, n_share, n_fail = [], 0, 0, 0
    fail_log = []

    for r in rows:
        grid = [float(t) for t in r["input_grid"]]
        key = (r["video_id"], tuple(grid))
        if key in grid_cache:
            out_rows.append(infer_row(r, grid_cache[key])); n_share += 1
            continue
        sid = safe_id(r["query_id"])
        try:
            if args.dry_run:
                fps = float(r.get("fps", 30.0))
                _ = grid_to_frame_idx(grid, fps)
                L = len(grid)
                seed = int(hashlib.sha1(sid.encode()).hexdigest()[:8], 16)
                feats = np.random.default_rng(seed).standard_normal(
                    (L, args.patches, args.out_hidden)).astype(np.float16)
                ts = np.asarray(grid, dtype=np.float32)
            else:
                path = resolve_video_path(r["video_id"], args.video_root, args.video_ext, vmap)
                if not path or not os.path.exists(path):
                    raise FileNotFoundError(f"video not found: {r['video_id']} -> {path}")
                n_frames, real_fps = meta(path)
                fps = real_fps or float(r.get("fps", 30.0))
                frame_idx = grid_to_frame_idx(grid, fps, n_frames)
                feats, ts = ex.encode_video_at(path, frame_idx, np.asarray(grid, np.float32),
                                               target_patches=args.patches, reader=reader)
                feats = feats.astype(np.float16)
            ref_name = save_features(args.out, sid, feats, ts, args.format)
            feature_ref = os.path.join(os.path.basename(args.out), ref_name) if args.ref_prefix else ref_name
            grid_cache[key] = feature_ref
            out_rows.append(infer_row(r, feature_ref)); n_feat += 1
            if n_feat % 200 == 0:
                print(f"[materialize] materialized {n_feat} (latest {sid}: L={len(ts)})")
        except Exception as e:
            n_fail += 1
            fail_log.append({"query_id": r.get("query_id"), "error": str(e)})

    write_jsonl(args.out_manifest, out_rows)
    if fail_log:
        write_jsonl(args.out_manifest + ".failed", fail_log)
    print(f"[materialize] features {n_feat} + shared {n_share} -> {args.out}")
    print(f"[materialize] inference manifest {len(out_rows)} rows -> {args.out_manifest}"
          f"{' (%d failed)' % n_fail if n_fail else ''}")


def main():
    ap = argparse.ArgumentParser(description="CSG/VFR grid to features + inference manifest")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="bench_feats")
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--video-root", default="")
    ap.add_argument("--video-ext", default=".mp4")
    ap.add_argument("--video-map", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--patches", type=int, default=9)
    ap.add_argument("--out-hidden", type=int, default=3584)
    ap.add_argument("--format", default="npz", choices=["npz", "npy"])
    ap.add_argument("--ref-prefix", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
