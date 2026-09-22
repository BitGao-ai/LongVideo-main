"""Content-adaptive variable-step sampler producing real variable timestamps."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np


_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(msg, flush=True)


@dataclass
class SamplerConfig:
    strategy: str = "adaptive"
    coarse_fps: float = 4.0
    saliency: str = "absdiff"
    saliency_accum: str = "frame_diff"
    theta: float = 0.12
    dt_min_s: float = 0.2
    dt_max_s: float = 2.0
    max_frames: int = 8192


def _open_video(path: str):
    """Open video; return (kind, handle, get_gray, get_gray_batch, duration, fps, n_frames)."""
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        fps = float(vr.get_avg_fps()) or 25.0
        n = len(vr)
        dur = n / fps

        def get_gray(idx: int) -> np.ndarray:
            frame = vr[idx].asnumpy()
            g = frame.astype(np.float32).mean(-1) / 255.0
            return g

        def get_gray_batch(indices, chunk: int = 256):
            """Batch-fetch downsampled grayscale frames."""
            out = []
            idx = [int(i) for i in indices]
            for s in range(0, len(idx), chunk):
                batch = vr.get_batch(idx[s:s + chunk]).asnumpy()
                g = batch.astype(np.float32).mean(-1) / 255.0
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
            f"Cannot decode {path}: need decord or pyav. Original error: {e}")


def _absdiff(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute frame difference in [0, 1]."""
    return float(np.abs(a - b).mean())


def _downsample(g: np.ndarray, size: int = 64) -> np.ndarray:
    """Block-average downsample to size x size."""
    h, w = g.shape
    if h == size and w == size:
        return g
    ys = (np.arange(size + 1) * h) // size
    xs = (np.arange(size + 1) * w) // size
    ys[-1], xs[-1] = h, w
    if np.any(np.diff(ys) < 1) or np.any(np.diff(xs) < 1):
        yi = np.linspace(0, h - 1, size).astype(int)
        xi = np.linspace(0, w - 1, size).astype(int)
        return g[np.ix_(yi, xi)]
    rows = np.add.reduceat(g, ys[:-1], axis=0)
    out = np.add.reduceat(rows, xs[:-1], axis=1)
    counts = np.diff(ys)[:, None] * np.diff(xs)[None, :]
    return (out / counts).astype(g.dtype, copy=False)


def adaptive_indices(gray_seq: np.ndarray, times: np.ndarray, cfg: SamplerConfig):
    """Cumulative-saliency sampling over coarse grayscale frames.

    Args:
        gray_seq: (M, s, s) coarse grayscale frames.
        times: (M,) timestamps in seconds.
    Returns:
        (selected indices, selected timestamps, raw count before cap).
    """
    M = len(gray_seq)
    if M == 0:
        return np.array([], int), np.array([], np.float32)
    by_frame = (cfg.saliency_accum != "anchor_diff")
    sel = [0]
    last_t = float(times[0])
    anchor = gray_seq[0]
    prev = gray_seq[0]
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
    if n_raw > cfg.max_frames:
        keep = np.linspace(0, n_raw - 1, cfg.max_frames).round().astype(int)
        sel_idx = sel_idx[np.unique(keep)]
    return sel_idx, times[sel_idx].astype(np.float32), n_raw


def uniform_indices(times: np.ndarray, fps: float, cfg: SamplerConfig):
    """Uniform sampling at fixed fps; resamples across full span if over cap."""
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
    if n_raw > cfg.max_frames:
        keep = np.linspace(0, n_raw - 1, cfg.max_frames).round().astype(int)
        idx = idx[np.unique(keep)]
    return idx, times[idx].astype(np.float32), n_raw


def sample_video(path: str, cfg: SamplerConfig):
    """Sample one video; return (frame indices, timestamps, meta)."""
    kind, handle, get_gray, get_gray_batch, dur, fps, n = _open_video(path)
    if kind != "decord":
        raise NotImplementedError("Sampling requires decord random access.")

    stride = max(1, int(round(fps / max(cfg.coarse_fps, 1e-6))))
    cand = np.arange(0, n, stride)
    grays = get_gray_batch(cand) if len(cand) else np.empty((0, 64, 64), np.float32)
    times = (cand / fps).astype(np.float32)

    if cfg.saliency != "absdiff":
        _warn_once("saliency:" + str(cfg.saliency),
                   f"[sampler] saliency='{cfg.saliency}' is not implemented; using 'absdiff'.")
    if cfg.strategy == "uniform":
        sel_local, ts, n_raw = uniform_indices(times, cfg.coarse_fps, cfg)
    else:
        if cfg.strategy != "adaptive":
            _warn_once("strategy:" + str(cfg.strategy),
                       f"[sampler] strategy '{cfg.strategy}' is not implemented "
                       f"(no scene-cut detection); falling back to 'adaptive'.")
        sel_local, ts, n_raw = adaptive_indices(grays, times, cfg)

    frame_idx = cand[sel_local]
    event_density = float(len(frame_idx) / dur) if dur > 0 else 0.0
    dts = np.diff(ts) if len(ts) > 1 else np.array([0.0], np.float32)
    p50, p90, p99 = (float(x) for x in np.percentile(dts, [50, 90, 99]))
    capped = n_raw > cfg.max_frames
    ratio = (n_raw / len(frame_idx)) if len(frame_idx) else 1.0
    meta = dict(duration=round(dur, 3), fps=round(fps, 3),
                n_sampled=int(len(frame_idx)), event_density=round(event_density, 4),
                strategy=cfg.strategy,
                n_selected_raw=int(n_raw),
                decimate_ratio=round(ratio, 3),
                dt_p50=round(p50, 3), dt_p90=round(p90, 3), dt_p99=round(p99, 3))
    print(f"[sampler] L={len(frame_idx)} (raw {n_raw}) dur={dur:.0f}s "
          f"dt: median={p50:.2f}s p90={p90:.2f}s p99={p99:.2f}s floor~{p50/4:.2f}s", flush=True)
    if capped:
        print(f"[sampler] max_frames={cfg.max_frames} capped: {n_raw} -> {len(frame_idx)} "
              f"({ratio:.2f}x). dt_max_s={cfg.dt_max_s}s no longer holds, "
              f"max interval ~ {p99:.1f}s.", flush=True)
    elif p50 > cfg.dt_max_s + 1e-6:
        print(f"[sampler] median dt {p50:.2f}s > dt_max_s={cfg.dt_max_s}s; "
              f"lower theta (now {cfg.theta}) for denser sampling", flush=True)
    return frame_idx.astype(int), ts, meta


def _dry_run(cfg: SamplerConfig):
    """Validate sampling logic on synthetic static/dynamic/slow-drift signal."""
    rng = np.random.default_rng(0)
    M = 400
    times = np.arange(M) / cfg.coarse_fps
    grays = np.zeros((M, 64, 64), np.float32)
    base = rng.random((64, 64)).astype(np.float32)
    slow = base.copy()
    for k in range(M):
        if k < 120:
            grays[k] = base
        elif k < 200:
            grays[k] = rng.random((64, 64)).astype(np.float32)
        else:
            slow = slow + 0.002 * rng.standard_normal((64, 64)).astype(np.float32)
            grays[k] = slow
    idx, ts, _ = adaptive_indices(grays, times.astype(np.float32), cfg)
    dt = np.diff(ts)
    print(f"[dry-run] synthetic {M} coarse frames -> sampled {len(idx)} frames")
    print(f"[dry-run] dt: min={dt.min():.3f}s max={dt.max():.3f}s mean={dt.mean():.3f}s "
          f"std={dt.std():.3f}s (std>0 means variable steps)")
    def density(lo_k, hi_k):
        picked = ((ts >= lo_k / cfg.coarse_fps) & (ts < hi_k / cfg.coarse_fps)).sum()
        return picked / max(hi_k - lo_k, 1)
    print(f"[dry-run] density (sampled/coarse): static={density(0,120):.2f} "
          f"dynamic={density(120,200):.2f} slow={density(200,M):.2f}")


def main():
    ap = argparse.ArgumentParser(description="Content-adaptive variable-step sampler")
    ap.add_argument("--video", help="Single video path; omit for dry-run")
    ap.add_argument("--strategy", default="adaptive", choices=["uniform", "adaptive", "scene"],
                    help="'scene' is not implemented yet; it warns and falls back to 'adaptive'")
    ap.add_argument("--saliency-accum", default="frame_diff", choices=["frame_diff", "anchor_diff"],
                    help="Saliency accumulation: frame_diff or anchor_diff")
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
    print(f"Sampled indices (first 10): {idx[:10].tolist()}")
    print(f"Timestamps (first 10)s: {np.round(ts[:10], 3).tolist()}")


if __name__ == "__main__":
    main()
