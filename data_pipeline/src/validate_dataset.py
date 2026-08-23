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


def _infer_feat_dim(rows: list, root: str):
    """从首个可读特征推断 d（与训练脚本 infer_feat_dim 同策略）；全部不可达返回 None。
    不同模型视觉塔输出维不同（如 Qwen3-VL-4B=2560、更大变体=3584），写死期望维会误报。"""
    for r in rows:
        ref = r.get("feature_ref")
        if not ref:
            continue
        try:
            feats, _ = _load_feat_ts(ref, root)
            return int(feats.shape[-1])
        except Exception:
            continue
    return None


def check_row(r: dict, root: str, cfg: dict) -> tuple[list, int | None]:
    """返回 (该行的错误列表, 特征维)。空错误列表=通过；软告警以 'WARN:' 前缀。

    维度一并返回，避免外层为了收集 dims 再把 .npz 整载解压一遍
    （.npz 是 zip 归档、mmap 无效，数十万样本上这是实打实的两倍 IO）。
    """
    errs = []
    ref = r.get("feature_ref")
    if not ref:
        return ["missing feature_ref"], None
    try:
        feats, ts = _load_feat_ts(ref, root)                 # ④ 可达可载
    except Exception as e:
        return [f"unreachable feature_ref: {e}"], None

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
    return errs, d


def run(args, cfg: dict):
    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if cfg["check_dim_consistency"] and cfg["feat_dim"] is None:
        d = _infer_feat_dim(rows, args.data_root)
        if d is None:
            print("[validate] 未指定 --feat-dim 且无法从特征推断维度（全部不可达？），跳过维度检查")
            cfg["check_dim_consistency"] = False
        else:
            cfg["feat_dim"] = d
            print(f"[validate] 未指定 --feat-dim：从特征自动推断期望维={d}（不同模型视觉塔输出维不同，"
                  f"如需强约束请显式传 --feat-dim）")
    n_err = n_warn = 0
    dims = set()
    for i, r in enumerate(rows):
        errs, dim = check_row(r, args.data_root, cfg)     # 一次加载，同时拿错误与维度
        for e in errs:
            if e.startswith("WARN:"):
                n_warn += 1
                if args.verbose:
                    print(f"  [warn] 行{i} {r.get('video_id')}: {e}")
            else:
                n_err += 1
                print(f"  [ERR ] 行{i} {r.get('video_id')}: {e}")
        if dim is not None:
            dims.add(dim)

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
                placeholder_mode="single", max_frames=8192, feat_dim=None, strict=True)


def main():
    ap = argparse.ArgumentParser(description="放行前四项硬校验")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--placeholder-mode", default="single", choices=["single", "expand"])
    ap.add_argument("--max-frames", type=int, default=8192,
                    help="expand 模式下校验占位符数 == min(L,max_frames)；须与 DataConfig 一致")
    ap.add_argument("--feat-dim", type=int, default=None,
                    help="期望特征维；缺省自动从特征文件推断（不同模型不同：如 Qwen3-VL-4B=2560）")
    ap.add_argument("--strict", action="store_true", help="有硬错误即非零退出")
    ap.add_argument("--verbose", action="store_true", help="打印软告警明细")
    a = ap.parse_args()
    cfg = dict(DEFAULTS, placeholder_mode=a.placeholder_mode, max_frames=a.max_frames,
               feat_dim=a.feat_dim, strict=a.strict)
    run(a, cfg)


if __name__ == "__main__":
    main()
