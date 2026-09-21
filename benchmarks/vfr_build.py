#!/usr/bin/env python3
"""Build the VFR-Stress eval set with bursty grids at fixed frame budgets.

Usage:
    python vfr_build.py --anno actnet_test.jsonl --levels 0 0.25 0.5 0.75 1.0 --nframes 128 --out vfr.jsonl
    python vfr_build.py --demo --levels 0 0.5 1.0 --out vfr_demo.jsonl
"""
from __future__ import annotations
import argparse, math, random
from common import write_jsonl, read_jsonl, coefficient_of_variation


def _largest_remainder(weights: list[float], total: int, floor: int = 1) -> list[int]:
    """Apportion total slots by weight with exact sum via largest remainders."""
    n = len(weights)
    if total < n * floor:
        raise ValueError(f"total({total}) must be >= segments({n}) x floor({floor})")
    rest = total - n * floor
    s = sum(weights) or 1.0
    exact = [w / s * rest for w in weights]
    base = [int(math.floor(e)) for e in exact]
    left = rest - sum(base)
    order = sorted(range(n), key=lambda i: exact[i] - base[i], reverse=True)
    for i in order[:left]:
        base[i] += 1
    return [b + floor for b in base]


def bursty_grid(duration: float, n_frames: int, level: float,
                rng: random.Random) -> list[float]:
    """Non-uniform timestamps with exactly n_frames points; level 0 is uniform."""
    n_frames = max(2, int(n_frames))
    if level <= 1e-6:
        return [round(i * duration / (n_frames - 1), 6) for i in range(n_frames)]

    n_seg = max(2, min(n_frames // 8, n_frames))
    bounds = sorted(rng.uniform(0, duration) for _ in range(n_seg - 1))
    edges = [0.0] + bounds + [duration]
    sigma = 2.5 * level
    weights = [math.exp(rng.gauss(0, sigma)) for _ in range(n_seg)]
    alloc = _largest_remainder(weights, n_frames, floor=1)

    grid: list[float] = []
    for (a, b), k in zip(zip(edges[:-1], edges[1:]), alloc):
        span = b - a
        grid += [a + j * span / k for j in range(k)]
    grid = sorted(round(min(max(t, 0.0), duration), 6) for t in grid)
    grid = _dedup_keep_count(grid, n_frames, duration)
    return grid


def _dedup_keep_count(grid: list[float], n: int, duration: float) -> list[float]:
    """Deduplicate then refill from the largest gap to keep exactly n points."""
    uniq = sorted(set(grid))
    while len(uniq) < n:
        gaps = [(uniq[i + 1] - uniq[i], i) for i in range(len(uniq) - 1)]
        if not gaps:
            uniq.append(min(duration, uniq[-1] + 1e-6) if uniq else 0.0)
            uniq = sorted(set(uniq))
            continue
        g, i = max(gaps)
        mid = round(uniq[i] + g / 2, 6)
        if mid in (uniq[i], uniq[i + 1]):
            break
        uniq.insert(i + 1, mid)
    return uniq[:n]


def build(anno_rows, levels, n_frames, seed=0):
    rng = random.Random(seed)
    out = []
    for r in anno_rows:
        dur = float(r["duration"])
        for lv in levels:
            grid = bursty_grid(dur, n_frames, lv, rng)
            cv = coefficient_of_variation(grid)
            out.append(dict(
                video_id=r["video_id"], query_id=f'{r["query_id"]}@lv{lv}',
                base_query_id=r["query_id"], query=r.get("query", ""),
                gt_start=float(r["gt_start"]), gt_end=float(r["gt_end"]),
                duration=dur, input_grid=grid, level=lv, cv=round(cv, 4),
                split=r.get("split", "test"),
            ))
    return out


def make_demo(n=150, seed=0):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        dur = rng.uniform(60, 900)
        gs = rng.uniform(0, dur - 5); ge = min(dur, gs + rng.uniform(2, 20))
        rows.append(dict(video_id=f"vid{i:04d}", query_id=f"q{i:04d}",
                         query="event of interest", gt_start=round(gs, 3),
                         gt_end=round(ge, 3), duration=round(dur, 3), split="test"))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno"); ap.add_argument("--demo", action="store_true")
    ap.add_argument("--levels", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--nframes", type=int, default=128)
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = make_demo(seed=args.seed) if args.demo else read_jsonl(args.anno)
    built = build(rows, args.levels, args.nframes, args.seed)
    write_jsonl(args.out, built)
    from collections import defaultdict
    from common import mean
    by_cv, by_n = defaultdict(list), defaultdict(list)
    for b in built:
        by_cv[b["level"]].append(b["cv"])
        by_n[b["level"]].append(len(b["input_grid"]))
    print(f"[vfr_build] {len(rows)} annotations x levels={args.levels} -> {len(built)} rows -> {args.out}")
    print(f"[vfr_build] target frames={args.nframes}")
    print(f"{'level':>7} {'meanCV':>9} {'frames(min~max)':>16}")
    for lv in sorted(by_cv):
        ns = by_n[lv]
        print(f"{lv:>7} {mean(by_cv[lv]):>9.3f} {f'{min(ns)}~{max(ns)}':>16}")
    bad = [lv for lv, ns in by_n.items() if min(ns) != args.nframes or max(ns) != args.nframes]
    print("[vfr_build] " + ("frame budgets match" if not bad
                            else f"budget mismatch at levels: {bad}"))


if __name__ == "__main__":
    main()
