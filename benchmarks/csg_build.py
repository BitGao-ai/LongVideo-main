#!/usr/bin/env python3
"""
benchmarks/csg_build.py  —  构造 CSG（Continuous Sub-frame Grounding，亚帧连续时间定位）评测集

设计目标（设计方案 §5.1）：让"答案所需精度"细于"输入帧距 δ"，从而暴露离散模型的 δ/4 定位下界。
做法：从已有时序定位标注（Charades-STA / ActivityNet-Captions / QVHighlights）出发，
      为每个样本按给定 δ（或非均匀网格）生成"模型可见的输入帧时间戳"，
      同时保留高精度(ms) GT 边界。评测时离散模型只能输出网格上的时刻 → 撞 δ/4 地板。

输入标注格式（jsonl，每行）：
    {"video_id","query_id","query","gt_start","gt_end","duration","fps"}
用法：
    # 真实数据：把 Charades-STA 转成上面格式后
    python csg_build.py --anno charades_test.jsonl --delta 2.0 --out csg_delta2.jsonl
    # 扫多个 δ 画"MAE-δ"曲线：
    python csg_build.py --anno charades_test.jsonl --delta 0.5 1 2 4 --out csg_sweep.jsonl
    # 无数据先跑通逻辑：
    python csg_build.py --demo --out csg_demo.jsonl
"""
from __future__ import annotations
import argparse, random
from common import Segment, write_jsonl, read_jsonl


def uniform_grid(duration: float, delta: float, jitter: float = 0.0,
                 rng: random.Random | None = None) -> list[float]:
    """间隔 δ 的采样网格；jitter>0 时加轻微抖动（更贴近真实抽帧）。"""
    rng = rng or random.Random(0)
    n = max(2, int(duration / delta) + 1)
    grid = []
    for i in range(n):
        t = i * delta
        if jitter:
            t += rng.uniform(-jitter, jitter) * delta
        grid.append(min(max(0.0, t), duration))
    return sorted(set(round(t, 4) for t in grid))


def build(anno_rows: list[dict], deltas: list[float], jitter: float,
          seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for r in anno_rows:
        dur = float(r["duration"])
        gs, ge = float(r["gt_start"]), float(r["gt_end"])
        for d in deltas:
            grid = uniform_grid(dur, d, jitter, rng)
            seg = Segment(
                video_id=r["video_id"],
                query_id=f'{r["query_id"]}@d{d}',
                query=r.get("query", ""),
                gt_start=gs, gt_end=ge, duration=dur,
                input_grid=grid, split=r.get("split", "test"),
            )
            row = {
                **{k: getattr(seg, k) for k in
                   ["video_id", "query_id", "query", "gt_start", "gt_end", "duration", "input_grid", "split"]},
                "delta": d,
                "floor": seg.floor(),          # δ/4，评测时的地板线
            }
            out.append(row)
    return out


def make_demo(n: int = 200, seed: int = 0) -> list[dict]:
    """合成标注：GT 边界为毫秒级随机，duration 30~600s。用于跑通评测逻辑。"""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        dur = rng.uniform(30, 600)
        gs = rng.uniform(0, dur - 5)
        ge = min(dur, gs + rng.uniform(1.0, 15.0))
        rows.append(dict(video_id=f"vid{i:04d}", query_id=f"q{i:04d}",
                         query="a person does something",
                         gt_start=round(gs, 3), gt_end=round(ge, 3),
                         duration=round(dur, 3), fps=30.0, split="test"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", help="输入标注 jsonl")
    ap.add_argument("--demo", action="store_true", help="用合成标注")
    ap.add_argument("--delta", type=float, nargs="+", default=[2.0],
                    help="一个或多个输入帧间隔 δ(秒)")
    ap.add_argument("--jitter", type=float, default=0.0, help="网格抖动比例(0~0.5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = make_demo(seed=args.seed) if args.demo else read_jsonl(args.anno)
    built = build(rows, args.delta, args.jitter, args.seed)
    n = write_jsonl(args.out, built)
    print(f"[csg_build] 样本(标注)={len(rows)}  ×  δ={args.delta} → 输出 {n} 条 → {args.out}")
    print(f"[csg_build] 地板线 δ/4 范围: "
          f"{min(b['floor'] for b in built):.3f} ~ {max(b['floor'] for b in built):.3f} s")


if __name__ == "__main__":
    main()
