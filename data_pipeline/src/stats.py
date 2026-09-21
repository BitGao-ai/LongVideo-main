"""Dataset statistics: duration and event-density histograms plus dt analysis."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import numpy as np


DUR_BUCKETS = [(0, 60), (60, 300), (300, 900), (900, 2700), (2700, 7200), (7200, 1e9)]
DUR_LABELS = ["<1m", "1-5m", "5-15m", "15-45m", "45m-2h", ">2h"]
DENSITY_BUCKETS = [(0, 0.3, "low"), (0.3, 1.0, "mid"), (1.0, 1e9, "high")]


def _bucket(v: float, buckets) -> int:
    for i, b in enumerate(buckets):
        if b[0] <= v < b[1]:
            return i
    return len(buckets) - 1


def _text_hist(counter: Counter, labels, title: str, width: int = 40):
    total = sum(counter.values()) or 1
    print(f"\n{title} (n={total})")
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
    _text_hist(dur_c, DUR_LABELS, "Duration distribution")
    _text_hist(den_c, [b[2] for b in DENSITY_BUCKETS], "Event density (frames/s)")
    if densities:
        a = np.array(densities)
        print(f"\nEvent density: min={a.min():.3f} median={np.median(a):.3f} max={a.max():.3f}")
        _stratify_advice(den_c)


def _stratify_advice(den_c: Counter):
    total = sum(den_c.values()) or 1
    low, mid, high = (den_c.get(i, 0)/total for i in range(3))
    print("\n[Stratification advice]")
    if high < 0.1:
        print("  High-density samples <10%: add high-motion videos.")
    if low < 0.1:
        print("  Low-density samples <10%: keep some static videos.")
    if 0.1 <= low and 0.1 <= high:
        print("  Coverage looks fine; resample near low:mid:high = 3:5:2.")


def dt_stats(feature_dir: str, k: int = 200):
    """Sample k features and report dt variation to confirm variable-step sampling."""
    files = [f for f in os.listdir(feature_dir) if f.endswith((".npz", ".ts.npy"))][:k]
    all_cv = []
    for fn in files:
        p = os.path.join(feature_dir, fn)
        ts = (np.load(p)["timestamps"] if fn.endswith(".npz")
              else np.load(p)).astype(np.float32)
        if len(ts) >= 3:
            dt = np.diff(ts)
            all_cv.append(float(dt.std() / (dt.mean() + 1e-9)))
    if all_cv:
        a = np.array(all_cv)
        print(f"\ndt CV over {len(a)} videos: median={np.median(a):.3f} "
              f"(~0 means uniform; >>0 means variable steps work)")
        if np.median(a) < 0.1:
            print("  CV is low: the sampler may have degraded to uniform.")


def main():
    ap = argparse.ArgumentParser(description="Dataset statistics")
    ap.add_argument("--manifest")
    ap.add_argument("--feature-dir")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.manifest:
        from_manifest(a.manifest)
    if a.feature_dir:
        dt_stats(a.feature_dir)
    if not a.manifest and not a.feature_dir:
        ap.error("one of --manifest or --feature-dir is required")


if __name__ == "__main__":
    main()
