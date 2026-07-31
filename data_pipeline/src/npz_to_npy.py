""".npz → .npy + .ts.npy 存储格式转换（规模化必做，docs/03 §3）。

原因：.npz 是 zip 归档，np.load 的 mmap 对其无效 → 每次读整载解压该视频全部特征。
分离成 {vid}.npy(features) + {vid}.ts.npy(timestamps) 后，dataset 可用 mmap_mode='r'
真惰性按 subsample 索引只读需要的帧，长视频训练 IO/内存友好。

可选两级散列目录（features_npy/{vid[:2]}/{vid}.npy）避免单目录几十万文件。
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def convert_one(npz_path: str, out_dir: str, shard_dirs: bool) -> tuple:
    vid = os.path.splitext(os.path.basename(npz_path))[0]
    sub = os.path.join(out_dir, vid[:2]) if shard_dirs else out_dir
    os.makedirs(sub, exist_ok=True)
    z = np.load(npz_path)
    try:
        feats = np.ascontiguousarray(z["features"])          # [L,P,d]
        ts = np.ascontiguousarray(z["timestamps"]).astype(np.float32)
    finally:
        z.close()
    npy = os.path.join(sub, f"{vid}.npy")
    np.save(npy, feats)                                      # 连续存储 → 真 mmap
    np.save(os.path.join(sub, f"{vid}.ts.npy"), ts)
    return vid, feats.shape, feats.dtype


def run(args):
    files = []
    for root, _, fns in os.walk(args.inp):
        files += [os.path.join(root, f) for f in fns if f.endswith(".npz")]
    files.sort()
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        files = files[i::n]

    total_bytes = 0
    for k, p in enumerate(files):
        vid, shape, dt = convert_one(p, args.out, args.shard_dirs)
        total_bytes += int(np.prod(shape)) * np.dtype(dt).itemsize
        if args.delete_src:
            os.remove(p)
        if (k + 1) % 500 == 0:
            print(f"[npz→npy] {k+1}/{len(files)}  最近 {vid} shape={shape}")
    gb = total_bytes / 1e9
    print(f"[npz→npy] 完成 {len(files)} 个 → {args.out}  （特征体量≈{gb:.1f} GB, {dtype_note()})")
    print(f"[npz→npy] manifest 里 feature_ref 记得指向 .npy（build_manifest --prefer-npy）")


def dtype_note() -> str:
    return "建议 float16 存储、索引后转 float32"


def main():
    ap = argparse.ArgumentParser(description=".npz → .npy+.ts.npy（启用真 mmap）")
    ap.add_argument("--in", dest="inp", required=True, help=".npz 特征目录")
    ap.add_argument("--out", required=True, help="输出 .npy 目录")
    ap.add_argument("--shard-dirs", action="store_true", help="两级散列目录（大规模）")
    ap.add_argument("--delete-src", action="store_true", help="转换后删除源 .npz")
    ap.add_argument("--shard", default=None, help='多机分片 "i/N"')
    run(ap.parse_args())


if __name__ == "__main__":
    main()
