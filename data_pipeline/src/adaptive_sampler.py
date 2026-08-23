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
    # 显著性累积口径：
    #   "frame_diff"（默认，与本文件文档一致）—— 累积**相邻粗扫帧**之间的变化 Σ|g_k − g_{k-1}|，
    #       静止段每帧贡献≈0，acc 不增长，间隔自然拉大。
    #   "anchor_diff"（旧行为，保留以复现历史特征）—— 累积每帧与**上次采样锚帧**的差
    #       Σ|g_k − g_anchor|。恒定小偏移（如镜头缓慢平移后静止）下 acc 随帧数线性增长，
    #       最终必然触发采样，"静止段自然拉大间隔"在这类内容上失效。
    saliency_accum: str = "frame_diff"
    theta: float = 0.12              # 累积显著性触发阈值（越小越密）
    dt_min_s: float = 0.2            # 最小帧间隔（高动态段限帧率）
    # 最大帧间隔（静止段兜底）。这个值直接决定定位任务的精度地板 δ/4：
    # dt_max=4.0 → 地板 1.0s，R@0.9 在 10s 事件上结构性不可达；2.0 → 地板 0.5s。
    dt_max_s: float = 2.0
    # 采样帧数上限。注意这是**上限不是目标**：实际帧数由 theta / dt_max_s 决定，
    # 调大它只是"不再砍"，不会凭空多采帧。
    #
    # 它的定位是**防御异常输入的保险丝，不是日常预算**。原因：dt_max_s 是速率契约
    # （间隔≤2s），max_frames 是绝对数量契约，二者只在 duration ≤ max_frames×dt_max_s
    # 时可同时满足。取 2048 时该临界仅 68 分钟，一部 2 小时视频会被抽稀 2.1× —— Δt 退回
    # 4.5s、定位地板 1.1s，等于把时间分辨率悄悄还给了上限。8192 把临界推到 4.5 小时。
    # 要省资源应显式调大 dt_max_s（有意识地降精度），而不是让 linspace 无声降级。
    # 必须与 DataConfig.max_frames 一致，否则训练侧会再抽稀一次（白花编码与磁盘）。
    max_frames: int = 8192


# ----------------------------- 解码 -----------------------------
def _open_video(path: str):
    """返回 (kind, handle, get_gray, get_gray_batch, duration_s, fps, n_frames)。lazy import。"""
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

        def get_gray_batch(indices, chunk: int = 256):
            """批量取灰度帧并立即下采样 → (len(indices), s, s)。

            逐帧 `vr[i]` 每次都要 seek 到最近关键帧再解码到目标帧；1 小时 25fps 视频、
            coarse_fps=4 就是约 15000 次 seek，是整条抽特征流水线的瓶颈。
            decord 的 get_batch 会对索引做排序/合并解码，快得多。
            分块是为了不让全分辨率帧一次性堆在内存里（下采样后才累积）。
            """
            out = []
            idx = [int(i) for i in indices]
            for s in range(0, len(idx), chunk):
                batch = vr.get_batch(idx[s:s + chunk]).asnumpy()      # (k,H,W,3) uint8
                g = batch.astype(np.float32).mean(-1) / 255.0         # (k,H,W)
                out.extend(_downsample(g[j]) for j in range(g.shape[0]))
            return np.stack(out) if out else np.empty((0, 64, 64), np.float32)

        return ("decord", vr, get_gray, get_gray_batch, dur, fps, n)
    except Exception:
        pass
    try:
        import av  # type: ignore
        container = av.open(path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) or 25.0
        dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
        return ("pyav", container, None, None, dur, fps, None)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"无法解码 {path}：需要 decord 或 pyav（pip install decord / av）。原始错误: {e}")


# --------------------------- 显著性 ----------------------------
def _absdiff(a: np.ndarray, b: np.ndarray) -> float:
    """归一化帧间平均绝对差（[0,1]）。下采样到 64×64 求速度。"""
    return float(np.abs(a - b).mean())


def _downsample(g: np.ndarray, size: int = 64) -> np.ndarray:
    """缩放到 size×size 的**块平均**（区域重采样，无 scipy 依赖）。

    早先是最近邻点采样 g[np.ix_(ys, xs)]：只取 64×64 个孤立像素点，丢掉其余全部信息。
    高频纹理（树叶、人群、字幕滚动）在点采样下会混叠成随机噪声，帧间绝对差 _absdiff
    因此被抬高且不稳定——静止但有纹理的镜头会被误判成高显著性，自适应采样在那里白白
    堆帧。块平均先在每个块内求均值（等价于抗混叠低通再采样），显著性估计噪声大幅下降。

    非整除时用 np.add.reduceat 按不等长块切分，避免 reshape 要求整除。
    """
    h, w = g.shape
    if h == size and w == size:
        return g
    ys = (np.arange(size + 1) * h) // size          # 每个输出行覆盖的输入行区间
    xs = (np.arange(size + 1) * w) // size
    ys[-1], xs[-1] = h, w
    # 退化情形（放大而非缩小）：某些块为空，回退最近邻
    if np.any(np.diff(ys) < 1) or np.any(np.diff(xs) < 1):
        yi = np.linspace(0, h - 1, size).astype(int)
        xi = np.linspace(0, w - 1, size).astype(int)
        return g[np.ix_(yi, xi)]
    rows = np.add.reduceat(g, ys[:-1], axis=0)      # (size, w)
    out = np.add.reduceat(rows, xs[:-1], axis=1)    # (size, size)
    counts = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return (out / counts).astype(g.dtype, copy=False)


# --------------------------- 采样核心 ---------------------------
def adaptive_indices(gray_seq: np.ndarray, times: np.ndarray, cfg: SamplerConfig):
    """在粗扫灰度帧序列上做累积显著性触发采样。

    参数
      gray_seq : (M, s, s) 粗扫灰度帧（已下采样）
      times    : (M,) 每个粗扫帧的真实秒级时刻
    返回
      sel_idx  : 选中帧在粗扫序列中的索引（含首帧）
      sel_t    : 选中帧的真实时间戳（严格单调）

    累积口径由 cfg.saliency_accum 决定，见 SamplerConfig 的说明。
    """
    M = len(gray_seq)
    if M == 0:
        return np.array([], int), np.array([], np.float32)
    by_frame = (cfg.saliency_accum != "anchor_diff")
    sel = [0]                                   # 首帧必采
    last_t = float(times[0])
    anchor = gray_seq[0]                        # 上次采样帧（anchor_diff 口径用）
    prev = gray_seq[0]                          # 上一粗扫帧（frame_diff 口径用）
    acc = 0.0
    for k in range(1, M):
        acc += _absdiff(gray_seq[k], prev if by_frame else anchor)
        prev = gray_seq[k]
        gap = float(times[k]) - last_t
        trigger = (acc >= cfg.theta or gap >= cfg.dt_max_s)
        too_soon = gap < cfg.dt_min_s
        if trigger and not too_soon:
            sel.append(k)
            last_t = float(times[k]); anchor = gray_seq[k]; acc = 0.0
    sel_idx = np.array(sel, int)
    n_raw = len(sel_idx)
    # 超上限：在已选点上均匀二次抽稀（保持首尾与时间戳真实性）
    if n_raw > cfg.max_frames:
        keep = np.linspace(0, n_raw - 1, cfg.max_frames).round().astype(int)
        sel_idx = sel_idx[np.unique(keep)]
    return sel_idx, times[sel_idx].astype(np.float32), n_raw


def uniform_indices(times: np.ndarray, fps: float, cfg: SamplerConfig):
    """固定 fps 均匀采样（仅基线对照/VFR 均匀臂）。

    超过 max_frames 时在**整段上均匀抽稀**，而不是截断到前 max_frames 帧。
    截断意味着 1 小时视频 / coarse_fps=4 / max_frames=512 只覆盖前 128 秒——
    作为"均匀采样"对照臂会系统性低估基线，让自适应采样的优势虚高。
    """
    M = len(times)
    if M == 0:
        return np.array([], int), np.array([], np.float32), 0
    step = 1
    if M > 1:
        dt = float(times[1] - times[0])
        if dt > 0:
            step = max(1, int(round((1.0 / max(fps, 1e-6)) / dt)))
    idx = np.arange(0, M, step)
    n_raw = len(idx)
    if n_raw > cfg.max_frames:                  # 全段均匀抽稀，保住首尾
        keep = np.linspace(0, n_raw - 1, cfg.max_frames).round().astype(int)
        idx = idx[np.unique(keep)]
    return idx, times[idx].astype(np.float32), n_raw


def sample_video(path: str, cfg: SamplerConfig):
    """对单视频采样，返回 (frame_indices_in_video, timestamps_s, meta)。

    frame_indices_in_video 供 extract_features 精确取帧再过视觉塔。
    """
    kind, handle, get_gray, get_gray_batch, dur, fps, n = _open_video(path)
    if kind != "decord":
        raise NotImplementedError("采样示例实现基于 decord 随机访问；pyav 路径请按需补顺序解码。")

    # 粗扫候选帧（按 coarse_fps 在原视频索引上取点）——批量解码，见 get_gray_batch
    stride = max(1, int(round(fps / max(cfg.coarse_fps, 1e-6))))
    cand = np.arange(0, n, stride)
    grays = get_gray_batch(cand) if len(cand) else np.empty((0, 64, 64), np.float32)
    times = (cand / fps).astype(np.float32)

    if cfg.strategy == "uniform":
        sel_local, ts, n_raw = uniform_indices(times, cfg.coarse_fps, cfg)
    else:  # adaptive / scene（scene 可在此基础上叠加场景切换硬点，示例合入 adaptive）
        sel_local, ts, n_raw = adaptive_indices(grays, times, cfg)

    frame_idx = cand[sel_local]
    event_density = float(len(frame_idx) / dur) if dur > 0 else 0.0
    # ---- Δt 诊断：调 theta / dt_max 的唯一依据，也是"上限是否咬人"的证据 ----
    dts = np.diff(ts) if len(ts) > 1 else np.array([0.0], np.float32)
    p50, p90, p99 = (float(x) for x in np.percentile(dts, [50, 90, 99]))
    capped = n_raw > cfg.max_frames
    ratio = (n_raw / len(frame_idx)) if len(frame_idx) else 1.0
    meta = dict(duration=round(dur, 3), fps=round(fps, 3),
                n_sampled=int(len(frame_idx)), event_density=round(event_density, 4),
                strategy=cfg.strategy,
                n_selected_raw=int(n_raw),          # 抽稀前采样器真正选出的帧数
                decimate_ratio=round(ratio, 3),     # >1 即上限咬人
                dt_p50=round(p50, 3), dt_p90=round(p90, 3), dt_p99=round(p99, 3))
    print(f"[sampler] L={len(frame_idx)}（原始选出 {n_raw}）dur={dur:.0f}s "
          f"Δt: 中位={p50:.2f}s p90={p90:.2f}s p99={p99:.2f}s 地板δ/4≈{p50/4:.2f}s", flush=True)
    if capped:
        # 静默降级是最危险的：max_frames 一咬人，dt_max 的"最大间隔"契约当场失效，
        # 而下游任何指标都看不出来。必须显式告警并给出失效后的真实上界。
        print(f"[sampler] ⚠️ max_frames={cfg.max_frames} 已生效：{n_raw} → {len(frame_idx)} "
              f"（抽稀 {ratio:.2f}×）。dt_max_s={cfg.dt_max_s}s 的间隔上限**已失效**，"
              f"实际最大间隔约 {p99:.1f}s。要保住该契约请调大 max_frames 或调大 dt_max_s。",
              flush=True)
    elif p50 > cfg.dt_max_s + 1e-6:
        print(f"[sampler] 提示：Δt 中位 {p50:.2f}s > dt_max_s={cfg.dt_max_s}s，"
              f"多数间隔由静止段兜底触发；若需更密请调低 theta（当前 {cfg.theta}）", flush=True)
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
    idx, ts, _ = adaptive_indices(grays, times.astype(np.float32), cfg)
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
    ap.add_argument("--saliency-accum", default="frame_diff", choices=["frame_diff", "anchor_diff"],
                    help="显著性累积口径：frame_diff=相邻帧差累积（默认，与文档一致）；"
                         "anchor_diff=与上次采样帧的差累积（旧行为，仅用于复现历史特征）")
    ap.add_argument("--theta", type=float, default=0.12)
    ap.add_argument("--coarse-fps", type=float, default=4.0)
    ap.add_argument("--dt-min", type=float, default=0.2)
    ap.add_argument("--dt-max", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=8192)
    a = ap.parse_args()
    cfg = SamplerConfig(strategy=a.strategy, theta=a.theta, coarse_fps=a.coarse_fps,
                        saliency_accum=a.saliency_accum,
                        dt_min_s=a.dt_min, dt_max_s=a.dt_max, max_frames=a.max_frames)
    if not a.video:
        _dry_run(cfg); return
    idx, ts, meta = sample_video(a.video, cfg)
    print(json.dumps(meta, ensure_ascii=False))
    print(f"采样帧索引(前10): {idx[:10].tolist()}")
    print(f"时间戳(前10)s   : {np.round(ts[:10], 3).tolist()}")


if __name__ == "__main__":
    main()
