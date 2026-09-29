"""Shared benchmark utilities: temporal IoU, boundary errors, JSONL IO."""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence


@dataclass
class Segment:
    """One localization sample; times in seconds."""
    video_id: str
    query_id: str
    query: str
    gt_start: float
    gt_end: float
    duration: float
    input_grid: list[float]
    split: str = "test"

    def local_delta(self, at: float | None = None) -> float:
        """Local sampling interval near the ground-truth span."""
        g = sorted(self.input_grid)
        if len(g) < 2:
            return self.duration
        at = at if at is not None else 0.5 * (self.gt_start + self.gt_end)
        for a, b in zip(g[:-1], g[1:]):
            if a <= at <= b:
                return b - a
        diffs = [b - a for a, b in zip(g[:-1], g[1:])]
        diffs.sort()
        return diffs[len(diffs) // 2]

    def floor(self, at: float | None = None) -> float:
        """Discrete resolution lower bound delta/4."""
        return self.local_delta(at) / 4.0


def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s1, e1 = a; s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 1e-9 else 0.0


def boundary_mae(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    """Mean endpoint absolute error in seconds."""
    return 0.5 * (abs(pred[0] - gt[0]) + abs(pred[1] - gt[1]))


def snap_to_grid(x: float, grid: Sequence[float]) -> float:
    """Snap a predicted time to the nearest input frame timestamp."""
    return min(grid, key=lambda t: abs(t - x))


def snap_segment(pred: tuple[float, float], grid: Sequence[float]) -> tuple[float, float]:
    return (snap_to_grid(pred[0], grid), snap_to_grid(pred[1], grid))


def recall_at_iou(ious: Sequence[float], thr: float) -> float:
    return sum(1 for v in ious if v >= thr) / max(1, len(ious))


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny, eps = 1e-30, 3e-16
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + b * math.log1p(-x) + a * math.log(x)) * _betacf(b, a, 1.0 - x) / b


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Two-sided paired t-test on per-sample scores; returns (t, p).

    Pure-python (no scipy): p via the regularized incomplete beta function.
    Used by the CSG fair-baseline protocol (设计方案 §6.3): a = continuous-query
    errors, b = discrete-model + same-interpolation errors.
    """
    assert len(a) == len(b) and len(a) >= 2, "paired_t_test needs equal-length, n>=2"
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    md = mean(diffs)
    var = sum((d - md) ** 2 for d in diffs) / (n - 1)
    if var < 1e-24:
        if abs(md) < 1e-12:
            return 0.0, 1.0   # no difference at all: t=0, two-sided p=1
        return (float("inf") if md > 0 else float("-inf")), 0.0
    t = md / math.sqrt(var / n)
    p = _betai(0.5 * (n - 1), 0.5, (n - 1) / ((n - 1) + t * t))
    return t, p


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def segment_from_row(r: dict) -> Segment:
    return Segment(
        video_id=r["video_id"], query_id=r["query_id"], query=r.get("query", ""),
        gt_start=float(r["gt_start"]), gt_end=float(r["gt_end"]),
        duration=float(r["duration"]), input_grid=[float(t) for t in r["input_grid"]],
        split=r.get("split", "test"),
    )


def coefficient_of_variation(grid: Sequence[float]) -> float:
    """Coefficient of variation of grid intervals as frame-rate irregularity."""
    g = sorted(grid)
    d = [b - a for a, b in zip(g[:-1], g[1:])]
    if len(d) < 2:
        return 0.0
    m = mean(d)
    var = mean([(x - m) ** 2 for x in d])
    return math.sqrt(var) / m if m > 1e-9 else 0.0
