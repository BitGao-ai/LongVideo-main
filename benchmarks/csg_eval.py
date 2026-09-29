#!/usr/bin/env python3
"""CSG eval: sub-frame MAE, recall at IoU, and floor-break rate.

Usage:
    python csg_eval.py --manifest csg_delta2.jsonl --pred my_preds.jsonl
    python csg_eval.py --manifest csg_demo.jsonl --demo
    # 公平基线对照（设计方案 §6.3）：连续查询 vs 离散+同款插值，配对 t 检验
    python csg_eval.py --manifest m.jsonl --compare cont.jsonl disc_interp.jsonl \
        --names continuous discrete+interp
"""
from __future__ import annotations
import argparse, random
from collections import defaultdict
from common import (read_jsonl, segment_from_row, temporal_iou, boundary_mae,
                    snap_segment, recall_at_iou, mean, paired_t_test)


def evaluate(manifest: list[dict], preds: dict[str, tuple[float, float]],
             snap: bool = False) -> dict:
    ious, maes, breaks, floors = [], [], [], []
    per_delta = defaultdict(lambda: {"mae": [], "break": []})
    n_missing = 0
    for row in manifest:
        seg = segment_from_row(row)
        if seg.query_id not in preds:
            n_missing += 1
            continue
        pred = preds[seg.query_id]
        if snap:
            pred = snap_segment(pred, seg.input_grid)
        gt = (seg.gt_start, seg.gt_end)
        iou = temporal_iou(pred, gt)
        mae = boundary_mae(pred, gt)
        floor = seg.floor()
        ious.append(iou); maes.append(mae); breaks.append(1.0 if mae < floor else 0.0)
        floors.append(floor)
        per_delta[row.get("delta", "NA")]["mae"].append(mae)
        per_delta[row.get("delta", "NA")]["break"].append(1.0 if mae < floor else 0.0)

    mae_eval = mean(maes)
    floor_eval = mean(floors)
    res = {
        "n_eval": len(maes), "n_missing": n_missing,
        "sub_frame_MAE": mae_eval,
        "mean_floor(d/4)": floor_eval,
        "MAE/floor_ratio": (mae_eval / floor_eval) if floor_eval > 0 else float("nan"),
        "R@0.7": recall_at_iou(ious, 0.7),
        "R@0.9": recall_at_iou(ious, 0.9),
        "Floor_Break_Rate": mean(breaks),
        "per_delta": {str(k): {"MAE": round(mean(v["mae"]), 4),
                                "FloorBreak": round(mean(v["break"]), 4)}
                       for k, v in sorted(per_delta.items(), key=lambda x: str(x[0]))},
    }
    return res


def per_sample_mae(manifest: list[dict], preds: dict[str, tuple[float, float]]
                   ) -> tuple[list[str], list[float]]:
    """Per-query boundary MAE for the paired fair-baseline test (§6.3)."""
    qids, maes = [], []
    for row in manifest:
        seg = segment_from_row(row)
        if seg.query_id not in preds:
            continue
        qids.append(seg.query_id)
        maes.append(boundary_mae(preds[seg.query_id], (seg.gt_start, seg.gt_end)))
    return qids, maes


def compare_paired(manifest: list[dict], pred_a: dict, pred_b: dict,
                   name_a: str, name_b: str) -> dict:
    """Paired comparison on common queries; A wins (lower MAE) iff mean diff < 0.

    §6.3 protocol: A = continuous-query predictions, B = discrete model scored
    at frame times with the *same* threshold-crossing interpolation. Only if A
    is significantly better (p < 0.05) may the paper claim C3 (sub-frame
    resolution necessity); otherwise the claim must be downgraded.
    """
    qa, ma = per_sample_mae(manifest, pred_a)
    qb, mb = per_sample_mae(manifest, pred_b)
    sb = set(qb)
    common = [q for q in qa if q in sb]
    ia = {q: i for i, q in enumerate(qa)}
    ib = {q: i for i, q in enumerate(qb)}
    da = [ma[ia[q]] for q in common]
    db = [mb[ib[q]] for q in common]
    t, p = paired_t_test(da, db)
    res = {
        "n_common": len(common),
        f"MAE({name_a})": mean(da), f"MAE({name_b})": mean(db),
        "mean_diff(A-B)": mean([x - y for x, y in zip(da, db)]),
        "paired_t": t, "p_value": p,
        "A_significantly_better(p<0.05)": bool(p < 0.05 and mean(da) < mean(db)),
    }
    return res


def demo_preds(manifest: list[dict], seed: int = 0):
    """Synthetic continuous vs discrete predictions showing the floor effect."""
    rng = random.Random(seed)
    cont, disc = {}, {}
    for row in manifest:
        from common import segment_from_row as _sf
        seg = _sf(row)
        ps = seg.gt_start + rng.gauss(0, 0.15)
        pe = seg.gt_end + rng.gauss(0, 0.15)
        cont[seg.query_id] = (ps, pe)
        disc[seg.query_id] = snap_segment((ps, pe), seg.input_grid)
    return cont, disc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="two pred jsonl files for the paired fair-baseline test")
    ap.add_argument("--names", nargs=2, default=["continuous", "discrete+interp"],
                    metavar=("A", "B"))
    ap.add_argument("--snap-to-grid", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)

    if args.compare:
        def _load(path):
            rows = read_jsonl(path)
            return {r["query_id"]: (float(r["pred_start"]), float(r["pred_end"]))
                    for r in rows}
        res = compare_paired(manifest, _load(args.compare[0]), _load(args.compare[1]),
                             args.names[0], args.names[1])
        print(f"=== Paired fair-baseline comparison (A={args.names[0]}, B={args.names[1]}) ===")
        _print(res)
        if not res["A_significantly_better(p<0.05)"]:
            print("  [note] A is NOT significantly better: per §6.3, downgrade the "
                  "sub-frame-resolution claim (C3) to 'interpolation-free,任意时刻读出'.")
        return

    if args.demo:
        cont, disc = demo_preds(manifest)
        print("=== Continuous (breaks the floor) ===")
        _print(evaluate(manifest, cont, snap=False))
        print("\n=== Discrete (snapped to grid) ===")
        _print(evaluate(manifest, disc, snap=False))
        print("\n=== Paired: continuous vs grid-snapped (demo of --compare) ===")
        _print(compare_paired(manifest, cont, disc, "continuous", "snapped"))
        return

    rows = read_jsonl(args.pred)
    preds = {r["query_id"]: (float(r["pred_start"]), float(r["pred_end"])) for r in rows}
    res = evaluate(manifest, preds, snap=args.snap_to_grid)
    _print(res)


def _print(res: dict):
    for k, v in res.items():
        if k == "per_delta":
            print("  per_delta:")
            for d, m in v.items():
                print(f"    d={d:>6}:  MAE={m['MAE']:.4f}s  FloorBreak={m['FloorBreak']:.3f}")
        else:
            print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")


if __name__ == "__main__":
    main()
