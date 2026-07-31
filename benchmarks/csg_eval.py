#!/usr/bin/env python3
"""
benchmarks/csg_eval.py  —  CSG 评测：亚帧 MAE / R@IoU / Floor-Break Rate

指标：
  - 亚帧 MAE：边界端点绝对误差均值(秒)，越小越好。
  - R@0.7 / R@0.9：时序 IoU≥阈值的召回。
  - Floor-Break Rate：边界 MAE < δ/4 的样本占比。离散模型≈0；连续模型应显著>0。
  - grid-snapped MAE：把预测吸附到输入网格后的 MAE——即"离散模型能达到的最好水平"，
        用于在同一张图上画出 δ/4 地板，直观展示连续范式穿透地板。

预测格式（jsonl）：{"query_id","pred_start","pred_end"}
用法：
  # 真实：用你的模型对 csg_*.jsonl 出预测后
  python csg_eval.py --manifest csg_delta2.jsonl --pred my_preds.jsonl
  # 只跑通逻辑（合成"连续/离散"两种预测对比）：
  python csg_eval.py --manifest csg_demo.jsonl --demo
  # 把任意预测按离散模型模拟（吸附到网格），看它如何撞地板：
  python csg_eval.py --manifest csg_demo.jsonl --pred my_preds.jsonl --snap-to-grid
"""
from __future__ import annotations
import argparse, random
from collections import defaultdict
from common import (read_jsonl, segment_from_row, temporal_iou, boundary_mae,
                    snap_segment, recall_at_iou, mean)


def evaluate(manifest: list[dict], preds: dict[str, tuple[float, float]],
             snap: bool = False) -> dict:
    ious, maes, breaks = [], [], []
    per_delta = defaultdict(lambda: {"mae": [], "break": []})
    n_missing = 0
    for row in manifest:
        seg = segment_from_row(row)
        if seg.query_id not in preds:
            n_missing += 1
            continue
        pred = preds[seg.query_id]
        if snap:                      # 模拟离散模型：只能输出网格上的时刻
            pred = snap_segment(pred, seg.input_grid)
        gt = (seg.gt_start, seg.gt_end)
        iou = temporal_iou(pred, gt)
        mae = boundary_mae(pred, gt)
        floor = seg.floor()
        ious.append(iou); maes.append(mae); breaks.append(1.0 if mae < floor else 0.0)
        per_delta[row.get("delta", "NA")]["mae"].append(mae)
        per_delta[row.get("delta", "NA")]["break"].append(1.0 if mae < floor else 0.0)

    res = {
        "n_eval": len(maes), "n_missing": n_missing,
        "sub_frame_MAE": mean(maes),
        "mean_floor(δ/4)": mean([segment_from_row(r).floor() for r in manifest]),
        # 头号判据：MAE/floor < 1 ⇒ 穿透离散地板（只有连续查询能做到）
        "MAE/floor_ratio": (mean(maes) / mean([segment_from_row(r).floor() for r in manifest])),
        "R@0.7": recall_at_iou(ious, 0.7),
        "R@0.9": recall_at_iou(ious, 0.9),
        # 次判据：per-sample 低于地板的比例。理论上 离散→0.5(对称于地板)、连续→1.0
        "Floor_Break_Rate": mean(breaks),
        "per_delta": {str(k): {"MAE": round(mean(v["mae"]), 4),
                                "FloorBreak": round(mean(v["break"]), 4)}
                       for k, v in sorted(per_delta.items(), key=lambda x: str(x[0]))},
    }
    return res


# ----------------- demo：合成两种"模型"的预测，验证地板效应 -----------------
def demo_preds(manifest: list[dict], seed: int = 0):
    """
    continuous：在真值附近加小噪声(σ≈0.15s)，可穿透 δ/4。
    discrete  ：先加同样小噪声，再吸附到输入网格 → 被 δ/4 地板卡住。
    """
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
    ap.add_argument("--manifest", required=True, help="csg_build 产出的 jsonl")
    ap.add_argument("--pred", help="预测 jsonl {query_id,pred_start,pred_end}")
    ap.add_argument("--snap-to-grid", action="store_true",
                    help="把预测吸附到输入网格（模拟离散模型）")
    ap.add_argument("--demo", action="store_true",
                    help="合成 continuous vs discrete 两组预测对比")
    args = ap.parse_args()

    manifest = read_jsonl(args.manifest)

    if args.demo:
        cont, disc = demo_preds(manifest)
        print("=== [DEMO] Continuous 模型（可穿透地板）===")
        _print(evaluate(manifest, cont, snap=False))
        print("\n=== [DEMO] Discrete 模型（预测吸附到网格，撞 δ/4 地板）===")
        _print(evaluate(manifest, disc, snap=False))
        print("\n解读（理论预测）：\n"
              "  · 头号判据 MAE/floor_ratio：连续 ≪1（穿透地板）；离散 ≈1（触底，Prop.2）。\n"
              "  · sub_frame_MAE：离散 ≈ δ/4（mean_floor）；连续 ≪ δ/4。\n"
              "  · Floor_Break_Rate：离散 ≈0.5（对称于地板）；连续 →1.0。")
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
                print(f"    δ={d:>6}:  MAE={m['MAE']:.4f}s  FloorBreak={m['FloorBreak']:.3f}")
        else:
            print(f"  {k}: {round(v,4) if isinstance(v,float) else v}")


if __name__ == "__main__":
    main()
