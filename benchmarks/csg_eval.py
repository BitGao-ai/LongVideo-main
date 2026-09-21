#!/usr/bin/env python3
"""CSG eval: sub-frame MAE, recall at IoU, and floor-break rate.

Usage:
    python csg_eval.py --manifest csg_delta2.jsonl --pred my_preds.jsonl
    python csg_eval.py --manifest csg_demo.jsonl --demo
"""
from __future__ import annotations
import argparse, random
from collections import defaultdict
from common import (read_jsonl, segment_from_row, temporal_iou, boundary_mae,
                    snap_segment, recall_at_iou, mean)


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
    ap.add_argument("--snap-to-grid", action="store_true")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)

    if args.demo:
        cont, disc = demo_preds(manifest)
        print("=== Continuous (breaks the floor) ===")
        _print(evaluate(manifest, cont, snap=False))
        print("\n=== Discrete (snapped to grid) ===")
        _print(evaluate(manifest, disc, snap=False))
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
