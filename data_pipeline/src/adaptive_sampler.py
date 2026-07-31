"""内容自适应变步长采样：把视频采成"动密静疏"的帧序列，产出**真实可变 Δt**。

这是 CST-SSM 的命门①（见 docs/02）。朴素 `ts=arange(T)/fps` 是恒定 Δt，会让 EACS
退化、Theorem 1 无法演示。本采样器用累积显著性触发：帧间变化累积到阈值就采一帧，
静止段自然拉大间隔、动态段变密，并记录每帧在原视频中的**真实秒级时间戳**。

依赖 decord（优先）或 pyav 解码；均缺失时 CLI 用合成信号 dry-run 跑通逻辑。
帧差/光流用 numpy 计算，无重依赖。
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SamplerConfig:
    strategy: str = "adaptive"       # uniform | adaptive | scene
    coarse_fps: float = 4.0          # 粗扫候选帧率
    saliency: str = "absdiff"        # absdiff | flow
    theta: float = 0.12              # 累积显著性触发阈值（越小越密）
    dt_min_s: float = 0.2            # 最小帧间隔（高动态段限帧率）
    dt_max_s: float = 4.0            # 最大帧间隔（静止段兜底）
    max_frames: int = 512            # 采样帧数上限


# ----------------------------- 解码 -----------------------------
def _open_video(path: str):
    """返回 (get_frame_gray(t)->HxW float32 in [0,1], duration_s, fps)。lazy import。"""
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        fps = float(vr.get_avg_fps()) or 25.0
        n = len(vr)
        dur = n / fps

        def get_gray(idx: int) -> np.ndarray:
            frame = vr[idx].asnumpy()                      # HxWx3 uint8
            g = frame.astype(np.float32).mean(-1) / 255.0
            return g
        return ("decord", vr, get_gray, dur, fps, n)
    except Exception:
        pass
    try:
        import av  # type: ignore
        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) or 25.0
        dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
        return ("pyav", container, None, dur, fps, None)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"无法解码 {path}：需要 decord 或 pyav（pip install decord / av）。原始错误: {e}")


# --------------------------- 显著性 ----------------------------
def _absdiff(a: np.ndarray, b: np.ndarray) -> float:
    """归一化帧间平均绝对差（[0,1]）。下采样到 64×64 求速度。"""
    return float(np.abs(a - b).mean())


def _downsample(g: np.ndarray, size: int = 64) -> np.ndarray:
    """最近邻缩放到 size×size（无 scipy 依赖）。"""
    h, w = g.shape
    ys = (np.linspace(0, h - 1, size)).astype(int)
    xs = (np.linspace(0, w - 1, size)).astype(int)
    return g[np.ix_(ys, xs)]


# --------------------------- 采样核心 ---------------------------
def adaptive_indices(gray_seq: np.ndarray, times: np.ndarray, cfg: SamplerConfig):
    """在粗扫灰度帧序列上做累积显著性触发采样。

    参数
      gray_seq : (M, s, s) 粗扫灰度帧（已下采样）
      times    : (M,) 每个粗扫帧的真实秒级时刻
    返回
      sel_idx  : 选中帧在粗扫序列中的索引（含首帧）
      sel_t    : 选中帧的真实时间戳（严格单调）
    """
    M = len(gray_seq)
    if M == 0:
        return np.array([], int), np.array([], np.float32)
    sel = [0]                                   # 首帧必采
    last_t = float(times[0])
    last_frame = gray_seq[0]
    acc = 0.0
    for k in range(1, M):
        acc += _absdiff(gray_seq[k], last_frame)
        gap = float(times[k]) - last_t
        trigger = (acc >= cfg.theta or gap >= cfg.dt_max_s)
        too_soon = gap < cfg.dt_min_s
        if trigger and not too_soon:
            sel.append(k)
            last_t = float(times[k]); last_frame = gray_seq[k]; acc = 0.0
    sel_idx = np.array(sel, int)
    # 超上限：在已选点上均匀二次抽稀（保持首尾与时间戳真实性）
    if len(sel_idx) > cfg.max_frames:
        keep = np.linspace(0, len(sel_idx) - 1, cfg.max_frames).round().astype(int)
        sel_idx = sel_idx[np.unique(keep)]
    return sel_idx, times[sel_idx].astype(np.float32)


def uniform_indices(times: np.ndarray, fps: float, cfg: SamplerConfig):
    """固定 fps 均匀采样（仅基线对照/VFR 均匀臂）。"""
    if len(times) == 0:
        return np.array([], int), np.array([], np.float32)
    step = max(1, int(round((1.0 / max(fps, 1e-6)) / (times[1] - times[0]))) if len(times) > 1 else 1)
    idx = np.arange(0, len(times), step)[: cfg.max_frames]
    return idx, times[idx].astype(np.float32)


def sample_video(path: str, cfg: SamplerConfig):
    """对单视频采样，返回 (frame_indices_in_video, timestamps_s, meta)。

    frame_indices_in_video 供 extract_features 精确取帧再过视觉塔。
    """
    kind, handle, get_gray, dur, fps, n = _open_video(path)
    if kind != "decord":
        raise NotImplementedError("采样示例实现基于 decord 随机访问；pyav 路径请按需补顺序解码。")

    # 粗扫候选帧（按 coarse_fps 在原视频索引上取点）
    stride = max(1, int(round(fps / max(cfg.coarse_fps, 1e-6))))
    cand = np.arange(0, n, stride)
    grays = np.stack([_downsample(get_gray(int(i))) for i in cand]) if len(cand) else np.empty((0, 64, 64))
    times = (cand / fps).astype(np.float32)

    if cfg.strategy == "uniform":
        sel_local, ts = uniform_indices(times, cfg.coarse_fps, cfg)
    else:  # adaptive / scene（scene 可在此基础上叠加场景切换硬点，示例合入 adaptive）
        sel_local, ts = adaptive_indices(grays, times, cfg)

    frame_idx = cand[sel_local]
    event_density = float(len(frame_idx) / dur) if dur > 0 else 0.0
    meta = dict(duration=round(dur, 3), fps=round(fps, 3),
                n_sampled=int(len(frame_idx)), event_density=round(event_density, 4),
                strategy=cfg.strategy)
    return frame_idx.astype(int), ts, meta


# ------------------------------ CLI ------------------------------
def _dry_run(cfg: SamplerConfig):
    """无视频/无解码依赖时，用合成信号验证采样逻辑：前段静止、中段剧变、后段缓变。"""
    rng = np.random.default_rng(0)
    M = 400
    times = np.arange(M) / cfg.coarse_fps
    grays = np.zeros((M, 64, 64), np.float32)
    base = rng.random((64, 64)).astype(np.float32)
    slow = base.copy()
    for k in range(M):
        if k < 120:            # 静止：与 base 完全相同
            grays[k] = base
        elif k < 200:          # 剧变：每帧全新随机
            grays[k] = rng.random((64, 64)).astype(np.float32)
        else:                  # 缓变：每帧相对上一帧仅微小漂移（累积慢）
            slow = slow + 0.002 * rng.standard_normal((64, 64)).astype(np.float32)
            grays[k] = slow
    idx, ts = adaptive_indices(grays, times.astype(np.float32), cfg)
    dt = np.diff(ts)
    print(f"[dry-run] 合成 {M} 粗扫帧 → 采样 {len(idx)} 帧")
    print(f"[dry-run] Δt: min={dt.min():.3f}s max={dt.max():.3f}s mean={dt.mean():.3f}s "
          f"std={dt.std():.3f}s  (std>0 说明变步长生效)")
    # 报告每段"采样密度"= 采样帧数 / 该段粗扫帧数（诚实指标，消除段长差异）
    def density(lo_k, hi_k):
        picked = ((ts >= lo_k / cfg.coarse_fps) & (ts < hi_k / cfg.coarse_fps)).sum()
        return picked / max(hi_k - lo_k, 1)
    print(f"[dry-run] 采样密度(采样/粗扫): 静止={density(0,120):.2f}  "
          f"剧变={density(120,200):.2f}  缓变={density(200,M):.2f}  → 静止应最疏、剧变最密")


def main():
    ap = argparse.ArgumentParser(description="内容自适应变步长采样（产出真实可变 Δt）")
    ap.add_argument("--video", help="单视频路径；省略则跑 dry-run 验证逻辑")
    ap.add_argument("--strategy", default="adaptive", choices=["uniform", "adaptive", "scene"])
    ap.add_argument("--theta", type=float, default=0.12)
    ap.add_argument("--coarse-fps", type=float, default=4.0)
    ap.add_argument("--dt-min", type=float, default=0.2)
    ap.add_argument("--dt-max", type=float, default=4.0)
    ap.add_argument("--max-frames", type=int, default=512)
    a = ap.parse_args()
    cfg = SamplerConfig(strategy=a.strategy, theta=a.theta, coarse_fps=a.coarse_fps,
                        dt_min_s=a.dt_min, dt_max_s=a.dt_max, max_frames=a.max_frames)
    if not a.video:
        _dry_run(cfg); return
    idx, ts, meta = sample_video(a.video, cfg)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"采样帧索引(前10): {idx[:10].tolist()}")
    print(f"时间戳(前10)s   : {np.round(ts[:10], 3).tolist()}")


if __name__ == "__main__":
    main()
