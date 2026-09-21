"""Offline frame-feature extraction with the Qwen3-VL visual tower.

Frames come from the adaptive variable-step sampler (real timestamps), not the
processor's uniform fps. Each .npz embeds a parameter fingerprint so
--skip-existing invalidates stale caches when settings change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Optional

import numpy as np

from .adaptive_sampler import SamplerConfig, sample_video
from .dist_utils import (
    get_dist_info,
    log_prefix,
    map_device_for_rank,
    maybe_set_cuda_device,
    resolve_shard,
    shard_list,
)


_FALLBACK_OUT_HIDDEN = 3584


class VisionTower:
    """Loads the Qwen3-VL visual tower once and encodes indexed frames."""

    def __init__(self, model_name: str, patches: int, chunk_frames: int = 64,
                 dtype: str = "auto", device: str = "auto", max_frame_tokens: int = 256):
        self.model_name = model_name
        self.patches = patches
        self.chunk = chunk_frames
        self.dtype = dtype
        self.device = device
        self.max_frame_tokens = max_frame_tokens
        self._ex = None

    def load(self):
        if self._ex is not None:
            return self._ex
        import torch
        from cst_ssm.integrations.qwen3_vl import Qwen3VLVisionFeatureExtractor

        if self.device != "auto":
            dev = self.device
        elif torch.cuda.is_available():
            dev = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        _dt_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        if self.dtype != "auto":
            tdtype = _dt_map[self.dtype]
        else:
            tdtype = torch.float32 if dev == "cpu" else torch.bfloat16

        print(f"[extract] loading visual tower {self.model_name} (device={dev}, dtype={tdtype})",
              flush=True)
        t0 = time.time()
        self._ex = Qwen3VLVisionFeatureExtractor.from_pretrained(
            self.model_name, torch_dtype=tdtype)
        self._ex.max_frame_tokens = int(self.max_frame_tokens)
        self._ex.visual.to(dev)
        print(f"[extract] visual tower ready in {time.time() - t0:.1f}s", flush=True)
        return self._ex

    def encode(self, video_path: str, frame_idx: np.ndarray, ts: np.ndarray):
        """Encode indexed frames in chunks; returns float16 feats[L,P,d], ts[L]."""
        ex = self.load()
        frame_idx = np.asarray(frame_idx)
        ts = np.asarray(ts, dtype=np.float32)
        L = len(frame_idx)
        c = max(1, int(self.chunk))
        reader = ex.make_reader(video_path)
        if L <= c:
            feats, ts_out = ex.encode_video_at(
                video_path, frame_idx, ts, target_patches=self.patches, reader=reader)
        else:
            parts = []
            n_chunk = (L + c - 1) // c
            t0 = time.time()
            for i, s in enumerate(range(0, L, c), 1):
                fi, tsi = frame_idx[s:s + c], ts[s:s + c]
                f, _ = ex.encode_video_at(
                    video_path, fi, tsi, target_patches=self.patches, reader=reader)
                parts.append(f)
                print(f"[extract] chunk {i}/{n_chunk} done "
                      f"({min(s + c, L)}/{L} frames, {time.time() - t0:.1f}s)", flush=True)
            feats, ts_out = np.concatenate(parts, 0), ts
        return feats.astype(np.float16), ts_out.astype(np.float32)


def _synthetic(frame_idx: np.ndarray, ts: np.ndarray, patches: int, d: int, seed: int):
    rng = np.random.default_rng(seed)
    L = len(frame_idx)
    return rng.standard_normal((L, patches, d)).astype(np.float16), ts.astype(np.float32)


def _out_hidden_from_config(model: str) -> Optional[int]:
    """Read the visual output dim from a local model dir without loading weights."""
    cfg_path = os.path.join(model, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        v = (cfg.get("vision_config") or {}).get("out_hidden_size")
        return int(v) if v else None
    except Exception:
        return None


def _resolve_out_hidden(args) -> int:
    """Dry-run feature dim: explicit flag, then local config.json, then fallback."""
    if args.out_hidden is not None:
        return int(args.out_hidden)
    d = _out_hidden_from_config(args.model)
    if d:
        print(f"[extract] dry-run: using out_hidden_size={d} from {args.model}/config.json")
        return d
    print(f"[extract] dry-run: --model '{args.model}' has no readable config.json; "
          f"falling back to d={_FALLBACK_OUT_HIDDEN}, which may not match the real tower.")
    return _FALLBACK_OUT_HIDDEN


def _sampler_fingerprint(scfg: SamplerConfig, patches: int, model: str, dry_run: bool,
                         max_frame_tokens: int = 0) -> str:
    """Stable hash of every sampling/encoding parameter affecting features."""
    payload = {
        "strategy": scfg.strategy,
        "saliency": scfg.saliency,
        "saliency_accum": scfg.saliency_accum,
        "downsample": "v2-block-mean",
        "theta": scfg.theta,
        "coarse_fps": scfg.coarse_fps,
        "dt_min_s": scfg.dt_min_s,
        "dt_max_s": scfg.dt_max_s,
        "max_frames": scfg.max_frames,
        "patches": patches,
        "pool": "v2-spatial2d",
        "max_frame_tokens": 0 if dry_run else int(max_frame_tokens),
        "model": "dry-run" if dry_run else model,
    }
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _seed_from_id(vid: str) -> int:
    return int(hashlib.sha1(vid.encode()).hexdigest()[:8], 16)


def _cached_fingerprint(npz_path: str) -> Optional[str]:
    try:
        with np.load(npz_path) as z:
            if "sampler_fp" in z.files:
                return str(z["sampler_fp"])
    except Exception:
        pass
    return None


def _save_npz(out_npz: str, feats, ts, fp: str, meta: dict):
    """Save features, timestamps, fingerprint, and provenance metadata."""
    np.savez(out_npz, features=feats, timestamps=ts,
             sampler_fp=np.array(fp),
             meta_model=np.array(str(meta.get("model", ""))),
             meta_out_hidden=np.int64(feats.shape[-1] if getattr(feats, "ndim", 0) >= 1 else 0),
             meta_patches=np.int64(feats.shape[-2] if getattr(feats, "ndim", 0) >= 3 else 0),
             meta_strategy=np.array(str(meta.get("strategy", ""))),
             meta_duration=np.float32(meta.get("duration", 0.0)),
             meta_n_sampled=np.int64(meta.get("n_sampled", len(ts))),
             meta_event_density=np.float32(meta.get("event_density", 0.0)),
             meta_n_selected_raw=np.int64(meta.get("n_selected_raw", len(ts))),
             meta_decimate_ratio=np.float32(meta.get("decimate_ratio", 1.0)),
             meta_dt_p50=np.float32(meta.get("dt_p50", 0.0)),
             meta_dt_p90=np.float32(meta.get("dt_p90", 0.0)),
             meta_dt_p99=np.float32(meta.get("dt_p99", 0.0)))


def run(args):
    scfg = SamplerConfig(strategy=args.sampler, theta=args.theta,
                         coarse_fps=args.coarse_fps, dt_max_s=args.dt_max,
                         saliency_accum=args.saliency_accum,
                         max_frames=args.max_frames)
    os.makedirs(args.out, exist_ok=True)
    rank, world, local_rank = get_dist_info()
    si, sn, shard_desc, shard_src = resolve_shard(args.shard)
    device = map_device_for_rank(args.device, local_rank)
    maybe_set_cuda_device(local_rank)
    out_hidden = _resolve_out_hidden(args) if args.dry_run else 0
    tower = None if args.dry_run else VisionTower(args.model, args.patches, args.chunk_frames,
                                                   dtype=args.dtype, device=device,
                                                   max_frame_tokens=args.max_frame_tokens)

    rows_all = [json.loads(l) for l in open(args.manifest) if l.strip()]
    rows = shard_list(rows_all, si, sn)
    if shard_desc is not None:
        print(f"{log_prefix()}[extract] shard {shard_desc} (source={shard_src}): "
              f"this rank handles {len(rows)}/{len(rows_all)} videos; device={device}", flush=True)

    if tower is not None:
        tower.load()
    if not rows:
        print("[extract] manifest empty (or shard empty); nothing to do")
        return
    print(f"[extract] {len(rows)} videos to process", flush=True)

    fp = _sampler_fingerprint(scfg, args.patches, args.model, args.dry_run,
                              args.max_frame_tokens)
    print(f"[extract] fingerprint sampler_fp={fp}")

    ok, fail, stale = 0, 0, 0
    fail_name = ("extract_failed.jsonl" if world <= 1
                 else f"extract_failed.part-{si:03d}-of-{sn:03d}.jsonl")
    fail_log = open(os.path.join(args.out, fail_name), "a")
    try:
        for idx, r in enumerate(rows, 1):
            vid = r["video_id"]
            out_npz = os.path.join(args.out, f"{vid}.npz")
            if args.skip_existing and os.path.exists(out_npz):
                old_fp = _cached_fingerprint(out_npz)
                if old_fp == fp:
                    continue
                stale += 1
                print(f"[extract] ({idx}/{len(rows)}) {vid}: stale cache "
                      f"{old_fp or 'missing'} != {fp}, re-extracting", flush=True)
            try:
                t0 = time.time()
                if args.dry_run:
                    rng = np.random.default_rng(_seed_from_id(vid))
                    L = int(rng.integers(20, args.max_frames))
                    ts = np.cumsum(rng.uniform(scfg.dt_min_s, scfg.dt_max_s, L)).astype(np.float32)
                    feats, ts = _synthetic(np.arange(L), ts, args.patches, out_hidden, seed=L)
                    meta = dict(duration=float(ts[-1]), n_sampled=L, strategy=args.sampler,
                                event_density=round(L/float(ts[-1]), 4), model="dry-run")
                else:
                    video_path = r.get("video_path") or os.path.join(args.video_root, r.get("rel_path", vid + ".mp4"))
                    print(f"[extract] ({idx}/{len(rows)}) {vid}: sampling -> {video_path}", flush=True)
                    frame_idx, ts, meta = sample_video(video_path, scfg)
                    print(f"[extract] ({idx}/{len(rows)}) {vid}: sampled L={len(frame_idx)} "
                          f"({time.time()-t0:.1f}s), encoding", flush=True)
                    feats, ts = tower.encode(video_path, frame_idx, ts)
                    meta["model"] = args.model
                    print(f"[extract] ({idx}/{len(rows)}) {vid}: done ({time.time()-t0:.1f}s)",
                          flush=True)
                _save_npz(out_npz, feats, ts, fp, meta)
                ok += 1
                print(f"[extract] ({idx}/{len(rows)}) {vid}: saved -> {out_npz} "
                      f"(L={len(ts)}, {time.time()-t0:.1f}s)", flush=True)
                if ok % 50 == 0:
                    print(f"[extract] progress {ok}/{len(rows)} (failed {fail})", flush=True)
            except Exception as e:
                fail += 1
                fail_log.write(json.dumps({"video_id": vid, "error": str(e)}, ensure_ascii=False) + "\n")
                fail_log.flush()
                if ok == 0 and fail <= 3:
                    print(f"[extract] early failures (latest {vid}: {e}); check {fail_name}",
                          flush=True)
    finally:
        fail_log.close()
    print(f"{log_prefix()}[extract] done {ok} .npz -> {args.out}; failed {fail} (see {fail_name})"
          + (f"; {stale} stale caches recomputed" if stale else ""))


def main():
    ap = argparse.ArgumentParser(description="Offline visual feature extraction")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--out-hidden", type=int, default=None)
    ap.add_argument("--patches", type=int, default=9)
    ap.add_argument("--max-frame-tokens", type=int, default=256)
    ap.add_argument("--sampler", default="adaptive", choices=["uniform", "adaptive", "scene"])
    ap.add_argument("--saliency-accum", default="frame_diff", choices=["frame_diff", "anchor_diff"])
    ap.add_argument("--theta", type=float, default=0.12)
    ap.add_argument("--coarse-fps", type=float, default=4.0)
    ap.add_argument("--dt-max", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=8192)
    ap.add_argument("--chunk-frames", type=int, default=64)
    ap.add_argument("--video-root", default="")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--dry-run", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
