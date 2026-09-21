#!/usr/bin/env python3
"""VFR-Stress eval: localization MAE slope against frame-rate irregularity (CV).

Usage:
    python vfr_eval.py --manifest vfr.jsonl --pred cstssm.jsonl --name CST-SSM
    python vfr_eval.py --manifest vfr_demo.jsonl --demo
"""
from __future__ import annotations
import argparse, random
from collections import defaultdict
from common import read_jsonl, segment_from_row, boundary_mae, mean


def lsq_slope(xs, ys):
    """Least-squares slope dY/dX."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 1e-12 else float("nan")


def eval_method(manifest, preds):
    """Aggregate MAE per level; returns ((level, cv, MAE) curve, MAE~CV slope)."""
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
    """Synthetic methods with matched base MAE but different CV sensitivity."""
    rng = random.Random(seed)
    specs = {"CST-SSM": 0.15, "dt-fed Mamba": 0.9, "Uniform-d Mamba": 2.2}
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
    print(f"\n=== {name} ===  slope dMAE/dCV = {slope:.3f} (lower is more robust)")
    print("  level    CV     MAE(s)")
    for lv, cv, m in curve:
        print(f"  {lv:>5}  {cv:5.2f}  {m:6.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred", action="append", default=[])
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
        print("\nSlope ranking (CST-SSM should be lowest):",
              {k: round(v, 3) for k, v in sorted(slopes.items(), key=lambda x: x[1])})
        return

    assert len(args.pred) == len(args.name), "--pred and --name counts must match"
    for path, name in zip(args.pred, args.name):
        rows = read_jsonl(path)
        preds = {r["query_id"]: (float(r["pred_start"]), float(r["pred_end"])) for r in rows}
        curve, slope = eval_method(manifest, preds)
        report(name, curve, slope)


if __name__ == "__main__":
    main()
