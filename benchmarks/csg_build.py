#!/usr/bin/env python3
"""Build the CSG (sub-frame grounding) eval set from temporal annotations.

Usage:
    python csg_build.py --anno charades_test.jsonl --delta 2.0 --out csg_delta2.jsonl
    python csg_build.py --demo --out csg_demo.jsonl
"""
from __future__ import annotations
import argparse, random
from common import Segment, write_jsonl, read_jsonl


def uniform_grid(duration: float, delta: float, jitter: float = 0.0,
                 rng: random.Random | None = None) -> list[float]:
    """Sampling grid with spacing delta and optional jitter."""
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
                "floor": seg.floor(),
            }
            out.append(row)
    return out


def make_demo(n: int = 200, seed: int = 0) -> list[dict]:
    """Synthetic millisecond-precision annotations for plumbing checks."""
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
    ap.add_argument("--anno")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--delta", type=float, nargs="+", default=[2.0])
    ap.add_argument("--jitter", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = make_demo(seed=args.seed) if args.demo else read_jsonl(args.anno)
    built = build(rows, args.delta, args.jitter, args.seed)
    n = write_jsonl(args.out, built)
    print(f"[csg_build] annotations={len(rows)} x delta={args.delta} -> {n} rows -> {args.out}")
    print(f"[csg_build] floor range: "
          f"{min(b['floor'] for b in built):.3f} ~ {max(b['floor'] for b in built):.3f} s")


if __name__ == "__main__":
    main()
