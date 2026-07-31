"""放行前四项硬校验（docs/03 §4）。--strict 下任一失败即非零退出，拒绝进训练。

① 时间戳严格单调（秒）      —— 连续查询/变步长扫描的前提，违反 → Δt<0 → NaN
② <video> 占位符数对齐      —— single: ==1；expand: ==min(L,max_frames)
③ 特征维度一致             —— 所有 d==feat_dim，P 一致
④ feature_ref 可达可载      —— .npy 有配套 .ts.npy

软告警：Δt 退化为常数（采样器可能没生效）、grounding 标注越界。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def _load_feat_ts(ref: str, root: str):
    path = os.path.join(root, ref)
    if ref.endswith(".npy"):
        feats = np.load(path, mmap_mode="r")
        ts_path = path[:-4] + ".ts.npy"
        if not os.path.exists(ts_path):
            raise FileNotFoundError(f"缺配套时间戳 {ts_path}")
        ts = np.load(ts_path)
        return feats, np.asarray(ts, np.float32)
    z = np.load(path)
    return z["features"], np.asarray(z["timestamps"], np.float32)


def count_video_tokens(prompt: str) -> int:
    return prompt.count("<video>")


def check_row(r: dict, root: str, cfg: dict) -> list:
    """返回该行的错误列表（空=通过）。软告警以 'WARN:' 前缀。"""
    errs = []
    ref = r.get("feature_ref")
    if not ref:
        return ["missing feature_ref"]
    try:
        feats, ts = _load_feat_ts(ref, root)                 # ④ 可达可载
    except Exception as e:
        return [f"unreachable feature_ref: {e}"]

    L = int(feats.shape[0]); d = int(feats.shape[-1])

    # ① 时间戳单调
    if cfg["check_monotonic_ts"]:
        if len(ts) != L:
            errs.append(f"ts len {len(ts)} != frames {L}")
        elif len(ts) >= 2 and not np.all(np.diff(ts) > 0):
            bad = int((np.diff(ts) <= 0).sum())
            errs.append(f"timestamps not strictly increasing ({bad} non-positive Δt)")

    # ② 占位符对齐
    if cfg["check_placeholder_align"]:
        n_tok = count_video_tokens(r.get("prompt", ""))
        if cfg["placeholder_mode"] == "single":
            if n_tok != 1:
                errs.append(f"<video> count {n_tok} != 1 (single mode)")
        else:
            expect = min(L, cfg["max_frames"])
            if n_tok != expect:
                errs.append(f"<video> count {n_tok} != min(L,max_frames)={expect}")

    # ③ 维度一致
    if cfg["check_dim_consistency"] and d != cfg["feat_dim"]:
        errs.append(f"feat_dim {d} != expected {cfg['feat_dim']}")

    # 软告警：Δt 常数
    if cfg["warn_constant_dt"] and len(ts) >= 3:
        dt = np.diff(ts)
        if dt.std() < 1e-4 * (dt.mean() + 1e-9):
            errs.append("WARN: Δt≈常数（采样器可能退化为均匀，EACS 优势无法体现）")

    # 软告警：grounding 越界
    if r.get("task_type") == "grounding" and "gt_end" in r:
        dur = float(r.get("duration", ts[-1] if len(ts) else 0))
        if float(r["gt_end"]) > dur + 1e-3 or float(r.get("gt_start", 0)) < -1e-3:
            errs.append(f"WARN: grounding 标注越界 [{r.get('gt_start')},{r['gt_end']}] vs dur {dur}")
    return errs


def run(args, cfg: dict):
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    n_err = n_warn = 0
    dims = set()
    for i, r in enumerate(rows):
        for e in check_row(r, args.data_root, cfg):
            if e.startswith("WARN:"):
                n_warn += 1
                if args.verbose:
                    print(f"  [warn] 行{i} {r.get('video_id')}: {e}")
            else:
                n_err += 1
                print(f"  [ERR ] 行{i} {r.get('video_id')}: {e}")
        # 收集维度一致性（跨行）
        try:
            f, _ = _load_feat_ts(r["feature_ref"], args.data_root)
            dims.add(int(f.shape[-1]))
        except Exception:
            pass

    print(f"[validate] 共 {len(rows)} 行：错误 {n_err}，告警 {n_warn}")
    if len(dims) > 1:
        print(f"  [ERR ] 跨样本 feat_dim 不一致: {sorted(dims)}"); n_err += 1
    if n_err == 0:
        print("[validate] ✅ 硬校验全通过，可放行训练")
    else:
        print("[validate] ❌ 存在硬错误，禁止放行")
        if cfg["strict"]:
            sys.exit(1)


DEFAULTS = dict(check_monotonic_ts=True, check_placeholder_align=True,
                check_dim_consistency=True, warn_constant_dt=True,
                placeholder_mode="single", max_frames=512, feat_dim=3584, strict=True)


def main():
    ap = argparse.ArgumentParser(description="放行前四项硬校验")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--placeholder-mode", default="single", choices=["single", "expand"])
    ap.add_argument("--max-frames", type=int, default=512)
    ap.add_argument("--feat-dim", type=int, default=3584)
    ap.add_argument("--strict", action="store_true", help="有硬错误即非零退出")
    ap.add_argument("--verbose", action="store_true", help="打印软告警明细")
    a = ap.parse_args()
    cfg = dict(DEFAULTS, placeholder_mode=a.placeholder_mode, max_frames=a.max_frames,
               feat_dim=a.feat_dim, strict=a.strict)
    run(a, cfg)


if __name__ == "__main__":
    main()
