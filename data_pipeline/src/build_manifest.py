"""Build unified manifest with <video> placeholder alignment."""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np


VIDEO_TOKEN = "<video>"


def _feature_len(path: str) -> int:
    """Return feature frame count without full load."""
    if path.endswith(".npy"):
        return int(np.load(path, mmap_mode="r").shape[0])
    z = np.load(path)
    try:
        return int(z["features"].shape[0])
    finally:
        z.close()


def build_prompt(question: str, n_frames: int, mode: str, max_frames: int) -> str:
    """Build prompt with placeholders. Args: question, n_frames, mode, max_frames."""
    q = question.replace(VIDEO_TOKEN, "").strip()
    if mode == "single":
        return f"{VIDEO_TOKEN} {q}"
    k = min(n_frames, max_frames)
    return (VIDEO_TOKEN * k) + " " + q


def build_answer(row: dict, task: str) -> str:
    """Build answer string for task."""
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
            if not qas:
                if args.require_qa:
                    n_missing += 1; continue
                qas = [dict(video_id=vid, question="", answer="", task_type=args.task)]

            for r in qas:
                task = r.get("task_type", args.task)
                sample = dict(
                    video_id=vid,
                    feature_ref=ref,
                    n_frames=int(L),
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
    print(f"[manifest] wrote {n_written} rows -> {args.out} (mode={args.placeholder_mode})")
    if n_missing:
        print(f"[manifest] skipped {n_missing} features without QA (--require-qa)")
    print(f"[manifest] next: validate_dataset.py --strict")


def main():
    ap = argparse.ArgumentParser(description="Build unified manifest with placeholder alignment")
    ap.add_argument("--feature-dir", required=True)
    ap.add_argument("--qa", default=None, help="QA jsonl; omit for self-supervised manifest")
    ap.add_argument("--out", default="data/manifests/train.jsonl")
    ap.add_argument("--data-root", default="data", help="Root for relative feature_ref")
    ap.add_argument("--task", default="qa", choices=["qa", "caption", "grounding", "causal"])
    ap.add_argument("--placeholder-mode", default="single", choices=["single", "expand"])
    ap.add_argument("--max-frames", type=int, default=8192,
                    help="Used in expand mode; must match DataConfig.max_frames")
    ap.add_argument("--split", default="train")
    ap.add_argument("--prefer-npy", action="store_true", help="Prefer .npy over .npz")
    ap.add_argument("--require-qa", action="store_true", help="Skip features without QA")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
