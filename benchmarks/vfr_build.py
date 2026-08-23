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


def _largest_remainder(weights: list[float], total: int, floor: int = 1) -> list[int]:
    """按权重把 total 个名额分到 len(weights) 段，每段至少 floor 个，**总和精确等于 total**。

    最大余数法：先按比例取整，剩下的名额给小数部分最大的那些段。
    """
    n = len(weights)
    if total < n * floor:
        raise ValueError(f"total({total}) 必须 ≥ 段数({n})×floor({floor})")
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
    """
    生成非均匀时间戳，**总点数恰好 n_frames**。level=0 → 均匀；level=1 → 强突发。

    等预算是这个基准的公平性前提：各 level 看到的信息量必须相同，差异只来自 Δt 是否均匀。
    旧实现 level=0 走精确 n_frames 分支、level>0 走 `max(1,round(...))+1` 再 set() 去重，
    实际帧数 ≈ n_frames + n_seg 且随 level 波动——把"帧数效应"和"非均匀效应"混在了一起，
    恰恰是这个基准要隔离的两个变量。

    机制：把 [0,duration] 切成 n_seg 段，每段一个 log-normal 速率权重（sigma 随 level
    增大 → CV 单调增），用最大余数法分配帧数（总和精确），段内均匀且**左闭右开**——
    因此相邻段边界不会产生重复点，去重也不会掉帧。
    """
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
        # 左闭右开 t_j = a + j·span/k（j=0..k-1）：段末点留给下一段的起点，永不重合
        grid += [a + j * span / k for j in range(k)]
    grid = sorted(round(min(max(t, 0.0), duration), 6) for t in grid)
    # 极端 rng 下可能出现宽度 < 1e-6 的段导致舍入后重合；此时按最大间隙补点，保住等预算
    grid = _dedup_keep_count(grid, n_frames, duration)
    return grid


def _dedup_keep_count(grid: list[float], n: int, duration: float) -> list[float]:
    """去掉重复时间戳后，从当前最大间隙里补点，使总数回到 n（保证严格单调 + 等预算）。"""
    uniq = sorted(set(grid))
    while len(uniq) < n:
        gaps = [(uniq[i + 1] - uniq[i], i) for i in range(len(uniq) - 1)]
        if not gaps:
            uniq.append(min(duration, uniq[-1] + 1e-6) if uniq else 0.0)
            uniq = sorted(set(uniq))
            continue
        g, i = max(gaps)
        mid = round(uniq[i] + g / 2, 6)
        if mid in (uniq[i], uniq[i + 1]):        # 已无可分空间，放弃补点
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
    ap.add_argument("--nframes", type=int, default=128, help="每视频总帧数(等预算)")
    ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = make_demo(seed=args.seed) if args.demo else read_jsonl(args.anno)
    built = build(rows, args.levels, args.nframes, args.seed)
    write_jsonl(args.out, built)
    # 报告各 level 的平均 CV 与**实际帧数**，等预算属性必须可见可核对
    from collections import defaultdict
    from common import mean
    by_cv, by_n = defaultdict(list), defaultdict(list)
    for b in built:
        by_cv[b["level"]].append(b["cv"])
        by_n[b["level"]].append(len(b["input_grid"]))
    print(f"[vfr_build] {len(rows)} 标注 × levels={args.levels} → {len(built)} 条 → {args.out}")
    print(f"[vfr_build] 目标帧数={args.nframes}")
    print(f"{'level':>7} {'平均CV':>9} {'帧数(min~max)':>16}")
    for lv in sorted(by_cv):
        ns = by_n[lv]
        print(f"{lv:>7} {mean(by_cv[lv]):>9.3f} {f'{min(ns)}~{max(ns)}':>16}")
    bad = [lv for lv, ns in by_n.items() if min(ns) != args.nframes or max(ns) != args.nframes]
    print("[vfr_build] " + ("✅ 各 level 帧数一致（等预算成立）" if not bad
                            else f"❌ 帧数不等预算的 level: {bad}"))
    print("[vfr_build] 平均 CV 应随 level 单调增（不均匀度是唯一变量）")


if __name__ == "__main__":
    main()
