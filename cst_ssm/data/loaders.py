"""Unified dataloader factory: synthetic and manifest data share one entry point."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .schema import DataConfig
from .bucketing import LengthGroupedBatchSampler
from .dataset import (VideoTemporalDataset, SyntheticVideoDataset,
                      SyntheticGroundingDataset, ByteTokenizer)
from .collate import collate_fn
from ..utils.distributed import dist_info


@dataclass
class LoaderConfig:
    manifest: str | None = None
    data_root: str = ""
    mode: str = "feature"                 # feature | pixel
    max_frames: int = 8192
    max_text_len: int = 8704
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False
    pad_id: int = 256
    length_bucketing: bool = False
    bucket_multiplier: int = 32
    seed: int = 0
    feat_dim: int = 64
    feat_patches: int = 4
    feat_dtype: str = "auto"              # auto | float16 | bfloat16 | float32
    synth_n: int = 128
    synth_frames: int = 32
    synth_grounding: bool = False


def _npz_member_shape(path: str, key: str = "features") -> tuple | None:
    """Read an array shape from an .npz member header without decompressing data."""
    import zipfile
    try:
        with zipfile.ZipFile(path) as zf:
            name = key + ".npy"
            if name not in zf.namelist():
                return None
            with zf.open(name) as f:
                version = np.lib.format.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
                else:
                    return None
                return tuple(shape)
    except Exception:
        return None


def _feat_shape(path: str, ref: str) -> tuple | None:
    """Shape probe: mmap header for .npy, member header for .npz."""
    if ref.endswith(".npy"):
        return tuple(np.load(path, mmap_mode="r").shape)
    if ref.endswith(".npz"):
        shape = _npz_member_shape(path)
        if shape is not None:
            return shape
        z = np.load(path)
        try:
            return tuple(z["features"].shape)
        finally:
            z.close()
    return None


def _probe_feat_shapes(manifest: str, data_root: str = "", max_probe: int = 8) -> list | None:
    """Collect shapes of the first readable features in a manifest."""
    shapes = []
    try:
        with open(manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    ref = json.loads(line).get("feature_ref")
                except Exception:
                    continue
                if not ref:
                    continue
                try:
                    shape = _feat_shape(os.path.join(data_root, ref), ref)
                except Exception:
                    continue
                if shape:
                    shapes.append(shape)
                    if len(shapes) >= max(1, int(max_probe)):
                        break
    except Exception:
        return None
    return shapes


def _one_value(shapes, axis: int, manifest: str, what: str, hint: str,
               min_ndim: int = 1) -> int | None:
    """Take the unique value on one axis; raise on mixed-source manifests."""
    vals = [int(s[axis]) for s in shapes if len(s) >= min_ndim]
    if not vals:
        return None
    if len(set(vals)) > 1:
        raise ValueError(
            f"manifest {manifest} has inconsistent {what}: {sorted(set(vals))}. {hint}")
    return vals[0]


def infer_feat_dim(manifest: str, data_root: str = "", max_probe: int = 8) -> int | None:
    """Infer feature dim d from readable entries; None when nothing is readable."""
    shapes = _probe_feat_shapes(manifest, data_root, max_probe)
    if not shapes:
        return None
    return _one_value(
        shapes, -1, manifest, "feature dim d",
        "Features were extracted with different backbones; re-extract with one model.")


def infer_feat_patches(manifest: str, data_root: str = "", max_probe: int = 8) -> int | None:
    """Infer per-frame patch count P (shape[-2]); None when unavailable."""
    shapes = _probe_feat_shapes(manifest, data_root, max_probe)
    if not shapes:
        return None
    return _one_value(shapes, -2, manifest, "patches per frame P",
                      "Re-extract features with one --patches setting.", min_ndim=3)


def align_feat_dim(cfg, manifest: str | None, data_root: str = "", tag: str = "cfg",
                   strict: bool = True):
    """Correct cfg.feat_dim to match real features. Returns cfg."""
    from dataclasses import replace
    if not manifest or getattr(cfg, "input_mode", "feature") != "feature":
        return cfg
    d = infer_feat_dim(manifest, data_root)
    if not d:
        msg = (f"[{tag}] cannot infer feature dim from {manifest}: no readable feature. "
               f"Check that feature_ref + --data-root='{data_root}' resolves to files; "
               f"run data_pipeline.src.validate_dataset --strict to locate bad rows.")
        if strict:
            raise SystemExit(msg + f" Aborted instead of training with feat_dim={cfg.feat_dim}.")
        print(msg + f" Continuing with feat_dim={cfg.feat_dim} (strict=False)")
        return cfg
    if d != cfg.feat_dim:
        print(f"[{tag}] feat_dim corrected: {cfg.feat_dim} -> {d} (from {manifest})")
        return replace(cfg, feat_dim=d)
    print(f"[{tag}] feat_dim={d} matches features")
    return cfg


def _peek_n_frames(path: str) -> int | None:
    """Read frame count from headers only; never decompress features."""
    try:
        if path.endswith(".npy"):
            return int(np.load(path, mmap_mode="r").shape[0])
        if path.endswith(".npz"):
            with np.load(path) as z:
                key = "timestamps" if "timestamps" in z.files else None
                return int(z[key].shape[0]) if key else None
    except Exception:
        return None
    return None


def sample_lengths(ds, lc: LoaderConfig) -> list[int] | None:
    """Per-sample frame counts for length bucketing; None falls back to plain sampling."""
    rows = getattr(ds, "rows", None)
    if not rows:
        return None
    cap = int(lc.max_frames)
    out, n_probe = [], 0
    for r in rows:
        n = r.get("n_frames")
        if n is None:
            ref = r.get("feature_ref")
            n = _peek_n_frames(os.path.join(lc.data_root, ref)) if ref else None
            n_probe += 1
        if n is None:
            return None
        out.append(min(int(n), cap))
    if n_probe:
        print(f"[loaders] probed {n_probe} feature headers for lengths; "
              f"rebuild the manifest with n_frames to skip this.")
    return out


def build_dataset(lc: LoaderConfig, tokenizer=None):
    """Build a real manifest dataset or a synthetic fallback."""
    if lc.manifest:
        patches = lc.feat_patches
        inferred_p = infer_feat_patches(lc.manifest, lc.data_root) if lc.mode == "feature" else None
        if inferred_p and inferred_p != patches:
            print(f"[loaders] feat_patches corrected: {patches} -> {inferred_p}")
            patches = inferred_p
        dcfg = DataConfig(manifest=lc.manifest, mode=lc.mode, max_frames=lc.max_frames,
                          max_text_len=lc.max_text_len, feat_dim=lc.feat_dim,
                          feat_patches=patches, feat_dtype=lc.feat_dtype)
        return VideoTemporalDataset(dcfg, tokenizer=tokenizer or ByteTokenizer(),
                                    data_root=lc.data_root)
    if lc.synth_grounding:
        return SyntheticGroundingDataset(n=lc.synth_n, L=lc.synth_frames, P=lc.feat_patches,
                                         d=lc.feat_dim)
    return SyntheticVideoDataset(n=lc.synth_n, L=lc.synth_frames, P=lc.feat_patches,
                                 d=lc.feat_dim)


def build_dataloader(lc: LoaderConfig, tokenizer=None):
    """Build a DataLoader with throughput options and automatic DDP sharding."""
    ds = build_dataset(lc, tokenizer)
    pixel = (lc.mode == "pixel")
    _tok_pad = getattr(tokenizer, "pad_id", None)
    pad_id = _tok_pad if _tok_pad is not None else lc.pad_id

    sampler = None
    shuffle = lc.shuffle
    world, rank = dist_info()
    if world > 1:
        print(f"[loaders] DDP sharding: world_size={world}, rank={rank}")

    batch_sampler = None
    if lc.length_bucketing:
        lengths = sample_lengths(ds, lc)
        if lengths is None:
            print("[loaders] length bucketing requested but lengths unavailable; "
                  "using plain sampling.")
        else:
            batch_sampler = LengthGroupedBatchSampler(
                lengths, batch_size=lc.batch_size, num_replicas=world, rank=rank,
                shuffle=lc.shuffle, drop_last=lc.drop_last, seed=lc.seed,
                bucket_multiplier=lc.bucket_multiplier)
    if batch_sampler is None and world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=lc.shuffle)
        shuffle = False

    if batch_sampler is not None:
        loader = DataLoader(
            ds, batch_sampler=batch_sampler, num_workers=lc.num_workers,
            pin_memory=lc.pin_memory,
            persistent_workers=(lc.persistent_workers and lc.num_workers > 0),
            collate_fn=partial(collate_fn, pad_id=pad_id, pixel=pixel),
        )
        return loader, ds

    loader = DataLoader(
        ds, batch_size=lc.batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=lc.num_workers,
        pin_memory=lc.pin_memory, drop_last=lc.drop_last,
        persistent_workers=(lc.persistent_workers and lc.num_workers > 0),
        collate_fn=partial(collate_fn, pad_id=pad_id, pixel=pixel),
    )
    return loader, ds


def cycle(loader):
    """Yield batches forever, calling set_epoch each epoch for correct shuffling."""
    epoch = 0
    while True:
        for s in (getattr(loader, "batch_sampler", None), getattr(loader, "sampler", None)):
            if hasattr(s, "set_epoch"):
                s.set_epoch(epoch)
                break
        for batch in loader:
            yield batch
        epoch += 1
