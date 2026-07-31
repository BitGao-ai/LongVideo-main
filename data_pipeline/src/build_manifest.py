"""统一 manifest 构建 + <video> 占位符对齐（命门②，docs/02 §3）。

输入：特征目录（{vid}.npz/.npy）+ QA 标注 jsonl（video_id → question/answer/[gt_start,gt_end]）。
输出：manifests/{split}.jsonl，每行一个 VideoSample（对齐 cst_ssm/data/schema.py）。

占位符对齐两模式：
  single（推荐）：prompt 放 1 个 <video>，运行时按有效帧数展开 → 与 max_frames 解耦。
  expand（严格）：prompt 放 min(L, max_frames) 个 <video> → 与当前 stand-in 逐位对齐。
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np


VIDEO_TOKEN = "<video>"


def _feature_len(path: str) -> int:
    """读特征帧数 L（不整载：.npy 用 mmap 读 shape；.npz 读 header）。"""
    if path.endswith(".npy"):
        return int(np.load(path, mmap_mode="r").shape[0])
    z = np.load(path)                      # .npz：这里只取 shape，随即释放
    try:
        return int(z["features"].shape[0])
    finally:
        z.close()


def build_prompt(question: str, n_frames: int, mode: str, max_frames: int) -> str:
    """构造含占位符的 prompt。question 不应自带 <video>。"""
    q = question.replace(VIDEO_TOKEN, "").strip()
    if mode == "single":
        return f"{VIDEO_TOKEN} {q}"
    k = min(n_frames, max_frames)          # expand：与 subsample 后帧数一致
    return (VIDEO_TOKEN * k) + " " + q


def build_answer(row: dict, task: str) -> str:
    if task == "grounding" and "gt_start" in row and "gt_end" in row:
        return f"from {float(row['gt_start']):.2f}s to {float(row['gt_end']):.2f}s"
    return row.get("answer", "")


def run(args):
    feat_ext = ".npy" if args.prefer_npy else ".npz"
    qa_index: dict = {}
    if args.qa:
        for l in open(args.qa):
            if l.strip():
                r = json.loads(l)
                qa_index.setdefault(r["video_id"], []).append(r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written, n_missing = 0, 0
    # os.walk 兼容扁平目录与两级散列目录（npz_to_npy --shard-dirs）
    feat_files = []
    for root, _, fns in os.walk(args.feature_dir):
        for fn in fns:
            if fn.endswith(feat_ext) and not fn.endswith(".ts.npy"):
                feat_files.append(os.path.join(root, fn))
    feat_files.sort()
    with open(args.out, "w") as fout:
        for fpath in feat_files:
            fn = os.path.basename(fpath)
            vid = fn[: -len(feat_ext)]
            L = _feature_len(fpath)
            ref = os.path.relpath(fpath, args.data_root) if args.data_root else fpath

            qas = qa_index.get(vid)
            if not qas:                                    # 无标注：自监督样本（Stage-1）
                if args.require_qa:
                    n_missing += 1; continue
                qas = [dict(video_id=vid, question="", answer="", task_type=args.task)]

            for r in qas:
                task = r.get("task_type", args.task)
                sample = dict(
                    video_id=vid,
                    feature_ref=ref,
                    prompt=build_prompt(r.get("question", r.get("prompt", "")),
                                        L, args.placeholder_mode, args.max_frames),
                    answer=build_answer(r, task),
                    task_type=task,
                    split=args.split,
                )
                for k in ("gt_start", "gt_end", "duration"):
                    if k in r:
                        sample[k] = round(float(r[k]), 3)
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_written += 1
    print(f"[manifest] 写入 {n_written} 行 → {args.out}（占位符模式={args.placeholder_mode}）")
    if n_missing:
        print(f"[manifest] 跳过 {n_missing} 个无 QA 标注的特征（--require-qa）")
    print(f"[manifest] 下一步：validate_dataset.py --strict 放行前硬校验")


def main():
    ap = argparse.ArgumentParser(description="统一 manifest 构建 + 占位符对齐")
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--qa", default=None, help="QA 标注 jsonl；省略则建自监督 manifest")
    ap.add_argument("--out", default="data/manifests/train.jsonl")
    ap.add_argument("--data-root", default="data", help="feature_ref 相对此根")
    ap.add_argument("--task", default="qa", choices=["qa", "caption", "grounding", "causal"])
    ap.add_argument("--placeholder-mode", default="single", choices=["single", "expand"])
    ap.add_argument("--max-frames", type=int, default=512, help="expand 模式对齐 subsample")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prefer-npy", action="store_true", help="优先读 .npy（否则 .npz）")
    ap.add_argument("--require-qa", action="store_true", help="无 QA 的特征跳过")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
