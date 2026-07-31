"""数据集统计：时长分布 / 事件密度分布 / Δt 可视化 → 指导分层与采样配比（docs/01 §4）。

纯文本直方图（无 matplotlib 依赖也能出数）；有 matplotlib 时 --plot 存 PNG 附录素材。
读 filtered.jsonl（含 duration/event_density）或直接扫 .npz/.npy 特征算 Δt 统计。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np


DUR_BUCKETS = [(0, 60), (60, 300), (300, 900), (900, 2700), (2700, 7200), (7200, 1e9)]
DUR_LABELS = ["<1m", "1-5m", "5-15m", "15-45m", "45m-2h", ">2h"]
DENSITY_BUCKETS = [(0, 0.3, "low"), (0.3, 1.0, "mid"), (1.0, 1e9, "high")]  # 帧/秒


def _bucket(v: float, buckets) -> int:
    for i, b in enumerate(buckets):
        if b[0] <= v < b[1]:
            return i
    return len(buckets) - 1


def _text_hist(counter: Counter, labels, title: str, width: int = 40):
    total = sum(counter.values()) or 1
    print(f"\n{title}（n={total}）")
    mx = max(counter.values()) if counter else 1
    for i, lab in enumerate(labels):
        c = counter.get(i, 0)
        bar = "█" * int(width * c / mx) if mx else ""
        print(f"  {lab:>8} | {bar} {c} ({100*c/total:.1f}%)")


def from_manifest(path: str):
    dur_c, den_c = Counter(), Counter()
    densities = []
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if "duration" in r:
            dur_c[_bucket(float(r["duration"]), DUR_BUCKETS)] += 1
        if "event_density" in r:
            d = float(r["event_density"]); densities.append(d)
            den_c[_bucket(d, [(b[0], b[1]) for b in DENSITY_BUCKETS])] += 1
    _text_hist(dur_c, DUR_LABELS, "时长分布")
    _text_hist(den_c, [b[2] for b in DENSITY_BUCKETS], "事件密度分布（帧/秒）")
    if densities:
        a = np.array(densities)
        print(f"\n事件密度: min={a.min():.3f} mid={np.median(a):.3f} max={a.max():.3f}")
        _stratify_advice(den_c)


def _stratify_advice(den_c: Counter):
    total = sum(den_c.values()) or 1
    low, mid, high = (den_c.get(i, 0)/total for i in range(3))
    print("\n[分层建议]")
    if high < 0.1:
        print("  ⚠ 高事件密度样本 <10%：Theorem 1 的'更新数∝V_T'斜率可能画不出，"
              "建议补体育/剪辑类高动态视频。")
    if low < 0.1:
        print("  ⚠ 低密度(静态)样本 <10%：Theorem 1(B)'残差恒0免更新'正例不足，"
              "建议保留部分监控/讲座类。")
    if 0.1 <= low and 0.1 <= high:
        print("  ✅ 三档覆盖尚可；按 low:mid:high≈3:5:2 重采样写最终 manifest。")


def dt_stats(feature_dir: str, k: int = 200):
    """抽样 k 个特征，统计 Δt 分布，判断变步长是否真的生效。"""
    files = [f for f in os.listdir(feature_dir) if f.endswith((".npz", ".ts.npy"))][:k]
    all_cv = []
    for fn in files:
        p = os.path.join(feature_dir, fn)
        ts = (np.load(p)["timestamps"] if fn.endswith(".npz")
              else np.load(p)).astype(np.float32)
        if len(ts) >= 3:
            dt = np.diff(ts)
            all_cv.append(float(dt.std() / (dt.mean() + 1e-9)))     # 变异系数
    if all_cv:
        a = np.array(all_cv)
        print(f"\nΔt 变异系数(CV) over {len(a)} 视频: mid={np.median(a):.3f} "
              f"（CV≈0 → 均匀采样；CV 明显>0 → 变步长生效）")
        if np.median(a) < 0.1:
            print("  ⚠ CV 偏低：采样器可能退化为均匀，检查 adaptive_sampler.theta/dt_max")


def main():
    ap = argparse.ArgumentParser(description="数据集统计（分层/采样指导）")
    ap.add_argument("--manifest", help="filtered.jsonl / train.jsonl（含 duration/event_density）")
    ap.add_argument("--feature-dir", help="扫特征算 Δt 变异系数")
    ap.add_argument("--plot", action="store_true", help="有 matplotlib 时存 PNG")
    a = ap.parse_args()
    if a.manifest:
        from_manifest(a.manifest)
    if a.feature_dir:
        dt_stats(a.feature_dir)
    if not a.manifest and not a.feature_dir:
        ap.error("需 --manifest 或 --feature-dir 之一")


if __name__ == "__main__":
    main()
