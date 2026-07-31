#!/usr/bin/env python3
"""
benchmarks/vfr_eval.py  —  VFR-Stress 评测：性能随'帧率不均匀度(CV)'的退化斜率

核心产出：每个方法一条"定位 MAE vs 不均匀度 CV"曲线，用最小二乘拟合**斜率**。
主张：CST-SSM（用真实 Δt）斜率显著更平缓；Uniform-Δ / Δt-fed Mamba 随 CV 上升明显退化。

预测格式（jsonl）：{"query_id","pred_start","pred_end"}（query_id 含 @lv 后缀，见 vfr_build）
用法：
  python vfr_eval.py --manifest vfr.jsonl --pred cstssm.jsonl --name CST-SSM \
                     --pred uniform.jsonl --name Uniform-Δ --pred dtfed.jsonl --name Δt-fed
  python vfr_eval.py --manifest vfr_demo.jsonl --demo
"""
from __future__ import annotations
import argparse, random
from collections import defaultdict
from common import read_jsonl, segment_from_row, boundary_mae, mean


def lsq_slope(xs, ys):
    """最小二乘斜率 dY/dX。"""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-12 else float("nan")


def eval_method(manifest, preds):
    """按 level 聚合 MAE，返回 (level→(cv,MAE)) 与拟合斜率(MAE~CV)。"""
    by_level_mae = defaultdict(list)
    by_level_cv = defaultdict(list)
    for row in manifest:
        seg = segment_from_row(row)
        if seg.query_id not in preds:
            continue
        p = preds[seg.query_id]
        by_level_mae[row["level"]].append(boundary_mae(p, (seg.gt_start, seg.gt_end)))
        by_level_cv[row["level"]].append(row["cv"])
    levels = sorted(by_level_mae)
    curve = [(lv, mean(by_level_cv[lv]), mean(by_level_mae[lv])) for lv in levels]
    slope = lsq_slope([c for _, c, _ in curve], [m for _, _, m in curve])
    return curve, slope


def demo_methods(manifest, seed=0):
    """
    合成三种方法：base MAE 相近，但对 CV 的敏感度不同。
      CST-SSM   : MAE ≈ 0.3 + 0.15*CV     （几乎不随 CV 涨）
      Δt-fed    : MAE ≈ 0.3 + 0.9 *CV
      Uniform-Δ : MAE ≈ 0.3 + 2.2 *CV     （最敏感）
    """
    rng = random.Random(seed)
    specs = {"CST-SSM": 0.15, "Δt-fed Mamba": 0.9, "Uniform-Δ Mamba": 2.2}
    outs = {}
    for name, k in specs.items():
        preds = {}
        for row in manifest:
            seg = segment_from_row(row)
            err = 0.3 + k * row["cv"]
            ps = seg.gt_start + rng.gauss(0, err)
            pe = seg.gt_end + rng.gauss(0, err)
            preds[seg.query_id] = (ps, pe)
        outs[name] = preds
    return outs


def report(name, curve, slope):
    print(f"\n=== {name} ===  斜率 dMAE/dCV = {slope:.3f}  (越小越鲁棒)")
    print("  level    CV     MAE(s)")
    for lv, cv, m in curve:
        print(f"  {lv:>5}  {cv:5.2f}  {m:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred", action="append", default=[], help="可多次；与 --name 一一对应")
    ap.add_argument("--name", action="append", default=[])
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)

    if args.demo:
        methods = demo_methods(manifest)
        slopes = {}
        for name, preds in methods.items():
            curve, slope = eval_method(manifest, preds)
            report(name, curve, slope); slopes[name] = slope
        print("\n斜率对比（应 CST-SSM 最小）:",
              {k: round(v, 3) for k, v in sorted(slopes.items(), key=lambda x: x[1])})
        return

    assert len(args.pred) == len(args.name), "--pred 与 --name 数量需一致"
    for path, name in zip(args.pred, args.name):
        rows = read_jsonl(path)
        preds = {r["query_id"]: (float(r["pred_start"]), float(r["pred_end"])) for r in rows}
        curve, slope = eval_method(manifest, preds)
        report(name, curve, slope)


if __name__ == "__main__":
    main()
