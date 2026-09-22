"""Convert .npz archives to .npy + .ts.npy for true memory mapping."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .dist_utils import get_dist_info, log_prefix, part_path, resolve_shard, shard_list


def _read_meta(z, vid: str) -> dict:
    """Carry over sampler_fp and meta_* scalars so provenance survives the conversion."""
    meta = {"video_id": vid}
    for k in z.files:
        if k in ("features", "timestamps"):
            continue
        v = z[k]
        try:
            v = v.item() if getattr(v, "ndim", 1) == 0 else v.tolist()
        except (ValueError, AttributeError):
            v = str(v)
        meta[k] = v
    return meta


def convert_one(npz_path: str, out_dir: str, shard_dirs: bool) -> tuple:
    vid = os.path.splitext(os.path.basename(npz_path))[0]
    sub = os.path.join(out_dir, vid[:2]) if shard_dirs else out_dir
    os.makedirs(sub, exist_ok=True)
    z = np.load(npz_path)
    try:
        feats = np.ascontiguousarray(z["features"])
        ts = np.ascontiguousarray(z["timestamps"]).astype(np.float32)
        meta = _read_meta(z, vid)
    finally:
        z.close()
    if feats.ndim < 1 or int(feats.shape[0]) != int(ts.shape[0]):
        raise ValueError(f"{vid}: feature frames {feats.shape} != timestamps {ts.shape[0]}")
    npy = os.path.join(sub, f"{vid}.npy")
    np.save(npy, feats)
    np.save(os.path.join(sub, f"{vid}.ts.npy"), ts)
    return vid, feats.shape, feats.dtype, npy, meta


def run(args):
    files = []
    for root, _, fns in os.walk(args.inp):
        files += [os.path.join(root, f) for f in fns if f.endswith(".npz")]
    files.sort()
    si, sn, shard_desc, shard_src = resolve_shard(args.shard)
    n_all = len(files)
    files = shard_list(files, si, sn)
    if shard_desc is not None:
        print(f"{log_prefix()}[npz->npy] shard {shard_desc} (source={shard_src}): "
              f"this rank handles {len(files)}/{n_all} files")

    total_bytes = 0
    meta_rows = []
    for k, p in enumerate(files):
        vid, shape, dt, npy, meta = convert_one(p, args.out, args.shard_dirs)
        total_bytes += int(np.prod(shape)) * np.dtype(dt).itemsize
        meta_rows.append(meta)
        if args.delete_src:
            chk = np.load(npy, mmap_mode="r")
            ok = tuple(chk.shape) == tuple(shape)
            if hasattr(chk, "_mmap") and chk._mmap is not None:
                chk._mmap.close()
            if not ok:
                print(f"[npz->npy] read-back mismatch for {vid}; keeping source {p}")
            else:
                os.remove(p)
        if (k + 1) % 500 == 0:
            print(f"[npz->npy] {k+1}/{len(files)} latest {vid} shape={shape}")
    os.makedirs(args.out, exist_ok=True)
    meta_base = os.path.join(args.out, "meta.jsonl")
    meta_path = part_path(meta_base, si, sn) if sn > 1 else meta_base
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    gb = total_bytes / 1e9
    print(f"{log_prefix()}[npz->npy] done {len(files)} files -> {args.out} (~{gb:.1f} GB); "
          f"meta -> {meta_path}")
    if get_dist_info()[1] <= 1:
        print("[npz->npy] point feature_ref at .npy in the manifest (build_manifest --prefer-npy)")


def main():
    ap = argparse.ArgumentParser(description=".npz to .npy + .ts.npy (enables true mmap)")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-dirs", action="store_true")
    ap.add_argument("--delete-src", action="store_true")
    ap.add_argument("--shard", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
