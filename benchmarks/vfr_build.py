#!/usr/bin/env python3
"""
benchmarks/vfr_build.py  —  构造 VFR-Stress（极端可变帧率）评测集

设计目标（设计方案 §5.2）：同一视频内帧率剧烈波动(0.1~30fps)，考察模型对'非均匀真实时间'的鲁棒性。
关键公平性：同一 (视频, 不均匀度 level) 生成**唯一网格**，所有对照方法看到**完全相同的帧**，
           差异只来自"是否用真实 Δt"。level 越高越不均匀（CV 越大），总帧数≈恒定（等预算）。

输出（jsonl）：在标注基础上加 {input_grid, level, cv}
用法：
    python vfr_build.py --anno actnet_test.jsonl --levels 0 0.25 0.5 0.75 1.0 --nframes 128 --out vfr.jsonl
    python vfr_build.py --demo --levels 0 0.5 1.0 --out vfr_demo.jsonl
"""
from __future__ import annotations
import argparse, math, random
from common import write_jsonl, read_jsonl, coefficient_of_variation


def bursty_grid(duration: float, n_frames: int, level: float,
                rng: random.Random) -> list[float]:
    """
    生成非均匀时间戳。level=0 → 近均匀；level=1 → 强突发。
    机制：把 [0,duration] 切成若干段，每段分配一个'速率权重' w，
          权重的离散度随 level 增大（log 空间展宽），再按权重分配帧数并段内均匀。
    """
    if level <= 1e-6:
        return sorted(round(i * duration / (n_frames - 1), 4) for i in range(n_frames))
    n_seg = max(2, n_frames // 8)
    bounds = sorted(rng.uniform(0, duration) for _ in range(n_seg - 1))
    edges = [0.0] + bounds + [duration]
    # 段速率权重：log-normal，sigma 随 level 增大 → CV 随 level 单调增
    sigma = 2.5 * level
    weights = [math.exp(rng.gauss(0, sigma)) for _ in range(n_seg)]
    wsum = sum(weights)
    # 按权重分配帧数（至少 1 帧/段）
    alloc = [max(1, int(round((n_frames - n_seg) * w / wsum))) + 1 for w in weights]
    grid: list[float] = []
    for (a, b), k in zip(zip(edges[:-1], edges[1:]), alloc):
        if k == 1:
            grid.append(0.5 * (a + b))
        else:
            grid += [a + j * (b - a) / (k - 1) for j in range(k)]
    grid = sorted(set(round(t, 4) for t in grid if 0.0 <= t <= duration))
    return grid


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
    ap.add_argument("--nframes", type=int, default=128, help="每视频总帧数(等预算)")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = make_demo(seed=args.seed) if args.demo else read_jsonl(args.anno)
    built = build(rows, args.levels, args.nframes, args.seed)
    write_jsonl(args.out, built)
    # 报告各 level 的平均 CV，确认单调
    from collections import defaultdict
    from common import mean
    by = defaultdict(list)
    for b in built:
        by[b["level"]].append(b["cv"])
    print(f"[vfr_build] {len(rows)} 标注 × levels={args.levels} → {len(built)} 条 → {args.out}")
    print("[vfr_build] level→平均CV(应随level单调增):",
          {lv: round(mean(v), 3) for lv, v in sorted(by.items())})


if __name__ == "__main__":
    main()
