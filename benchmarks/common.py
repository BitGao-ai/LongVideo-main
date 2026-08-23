"""
benchmarks/common.py
共享工具：时序 IoU、边界 MAE、栅格吸附、离散分辨率下界 δ/4、JSONL 读写。

理论根基（设计方案 命题 2 / 附录 B）：
    任何"只能在采样时刻 {t_k} 输出定位结果"的模型，其期望绝对定位误差 ≥ δ/4，
    其中 δ 为局部采样间隔。连续查询不受此约束。
本模块把该下界做成可计算量（floor / floor-break），供 CSG/VFR 评测复用。
"""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from typing import Iterable, Sequence


# ----------------------------- 数据结构 -----------------------------
@dataclass
class Segment:
    """一个 (视频, 查询) 的定位样本。时间单位统一为秒(float)。"""
    video_id: str
    query_id: str
    query: str
    gt_start: float          # 高精度真值（毫秒级）
    gt_end: float
    duration: float
    input_grid: list[float]  # 模型实际"看到"的输入帧时间戳（可非均匀）
    split: str = "test"

    def local_delta(self, at: float | None = None) -> float:
        """GT 附近的局部采样间隔 δ；at 缺省用区间中点。非均匀网格取最近相邻间隔。"""
        g = sorted(self.input_grid)
        if len(g) < 2:
            return self.duration
        at = at if at is not None else 0.5 * (self.gt_start + self.gt_end)
        # 找到包含 at 的相邻栅格对
        for a, b in zip(g[:-1], g[1:]):
            if a <= at <= b:
                return b - a
        # at 落在网格外：用中位间隔
        diffs = [b - a for a, b in zip(g[:-1], g[1:])]
        diffs.sort()
        return diffs[len(diffs) // 2]

    def floor(self, at: float | None = None) -> float:
        """离散分辨率下界 = δ/4。"""
        return self.local_delta(at) / 4.0


# ----------------------------- 指标 -----------------------------
def temporal_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    s1, e1 = a; s2, e2 = b
    inter = max(0.0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    return inter / union if union > 1e-9 else 0.0


def boundary_mae(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    """两端点绝对误差的平均（秒）。"""
    return 0.5 * (abs(pred[0] - gt[0]) + abs(pred[1] - gt[1]))


def snap_to_grid(x: float, grid: Sequence[float]) -> float:
    """把预测时刻吸附到最近的输入帧时间戳——模拟离散模型的输出约束。"""
    return min(grid, key=lambda t: abs(t - x))


def snap_segment(pred: tuple[float, float], grid: Sequence[float]) -> tuple[float, float]:
    return (snap_to_grid(pred[0], grid), snap_to_grid(pred[1], grid))


# ----------------------------- 汇总 -----------------------------
def recall_at_iou(ious: Sequence[float], thr: float) -> float:
    return sum(1 for v in ious if v >= thr) / max(1, len(ious))


def mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


# ----------------------------- IO -----------------------------
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
    """输入网格 Δt 的变异系数 CV=std/mean，作为'帧率不均匀度'的标量（VFR 用）。"""
    g = sorted(grid)
    d = [b - a for a, b in zip(g[:-1], g[1:])]
    if len(d) < 2:
        return 0.0
    m = mean(d)
    var = mean([(x - m) ** 2 for x in d])
    return math.sqrt(var) / m if m > 1e-9 else 0.0
