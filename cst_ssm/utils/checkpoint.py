"""Sharded safetensors checkpoints with per-file size guarantees."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import torch
from safetensors.torch import save_file, load_file

_DEFAULT_MAX_BYTES = int(3.8 * 1024 ** 3)
_INDEX_NAME = "model.safetensors.index.json"
_HEADER_MARGIN = 0.98


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f}{unit}"
        n /= 1024


def _dtype_bytes(t) -> int:
    return t.numel() * t.element_size() if torch.is_tensor(t) else 0


def _to_cpu(t):
    return t.detach().cpu().contiguous() if torch.is_tensor(t) else t


def _canonicalize(state_dict: dict) -> tuple[dict, dict]:
    """Split tied weights into (canonical tensors, aliases); keeps original refs."""
    seen: dict[tuple, str] = {}
    canonical: dict = {}
    aliases: dict = {}
    for name, t in state_dict.items():
        if not torch.is_tensor(t):
            canonical[name] = t
            continue
        ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else id(t)
        key = (ptr, t.storage_offset(), tuple(t.shape), tuple(t.stride()))
        if key in seen:
            aliases[name] = seen[key]
        else:
            seen[key] = name
            canonical[name] = t
    return canonical, aliases


def _pack_shards(tensors: dict, max_bytes: int) -> list[list[str]]:
    """Greedy size-descending bin packing; oversized tensors get their own shard."""
    items = sorted(tensors.items(), key=lambda kv: _dtype_bytes(kv[1]), reverse=True)
    shards: list[list[str]] = []
    sizes: list[int] = []
    for name, t in items:
        sz = _dtype_bytes(t)
        if sz > max_bytes:
            print(f"[checkpoint] tensor {name} ({sz/1e9:.2f}GB) exceeds shard limit; stored alone.")
            shards.append([name]); sizes.append(sz); continue
        placed = False
        for i in range(len(shards)):
            if sizes[i] + sz <= max_bytes:
                shards[i].append(name); sizes[i] += sz; placed = True; break
        if not placed:
            shards.append([name]); sizes.append(sz)
    return shards


def save_sharded(state_dict: dict, out_dir: str,
                 max_shard_bytes: int = _DEFAULT_MAX_BYTES,
                 extra_metadata: dict | None = None) -> dict:
    """Save a sharded checkpoint to out_dir; returns the index dict."""
    os.makedirs(out_dir, exist_ok=True)
    canonical, aliases = _canonicalize(state_dict)
    effective = max(1, int(max_shard_bytes * _HEADER_MARGIN))
    shards = _pack_shards(canonical, effective)
    n = len(shards)
    weight_map: dict = {}
    total = 0
    written: set = set()
    for i, names in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        part = {name: _to_cpu(canonical[name]) for name in names}
        save_file(part, os.path.join(out_dir, fname))
        for name in names:
            weight_map[name] = fname
            total += _dtype_bytes(part[name])
        del part
        written.add(fname)
    meta = {"total_size": total, "n_shards": n, "max_shard_bytes": max_shard_bytes}
    meta.update(extra_metadata or {})
    index = {
        "metadata": meta,
        "weight_map": weight_map,
        "aliases": aliases,
    }
    with open(os.path.join(out_dir, _INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2)
    stale = [f for f in os.listdir(out_dir)
             if f.startswith("model-") and f.endswith(".safetensors") and f not in written]
    for f in stale:
        os.remove(os.path.join(out_dir, f))
    if stale:
        print(f"[checkpoint] removed {len(stale)} stale shards from a previous save")
    print(f"[checkpoint] saved {len(canonical)} tensors(+{len(aliases)} aliases) to {n} shards, "
          f"total {_human(total)}, max {_human(max_shard_bytes)} -> {out_dir}")
    return index


def frozen_param_names(model) -> set:
    """Names of requires_grad=False parameters (state_dict key space)."""
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def base_weights_recoverable(model) -> bool:
    """Whether frozen weights can be re-fetched externally (HF base wrapper)."""
    llm = getattr(model, "llm", None)
    return llm is not None and hasattr(llm, "hf")


def trainable_state_dict(model) -> dict:
    """state_dict without frozen parameters; keeps all buffers."""
    frozen = frozen_param_names(model)
    return {k: v for k, v in model.state_dict().items() if k not in frozen}


def checkpoint_metadata(path: str) -> dict:
    """Read a sharded checkpoint's index metadata; {} when unavailable."""
    if not os.path.isdir(path):
        return {}
    try:
        with open(os.path.join(path, _INDEX_NAME)) as f:
            return json.load(f).get("metadata", {}) or {}
    except (OSError, ValueError):
        return {}


def load_sharded(out_dir: str, map_location: str = "cpu") -> dict:
    """Restore a full state_dict from out_dir."""
    with open(os.path.join(out_dir, _INDEX_NAME)) as f:
        index = json.load(f)
    state: dict = {}
    cache: dict = {}
    for name, fname in index["weight_map"].items():
        if fname not in cache:
            cache[fname] = load_file(os.path.join(out_dir, fname), device=map_location)
        state[name] = cache[fname][name]
    for alias, canonical in index.get("aliases", {}).items():
        state[alias] = state[canonical]
    return state


def verify_shards(out_dir: str, hard_limit_bytes: int = 4 * 1024 ** 3) -> bool:
    """Verify every shard file stays within the hard size limit."""
    ok = True
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".safetensors"):
            sz = os.path.getsize(os.path.join(out_dir, fn))
            flag = "OK" if sz <= hard_limit_bytes else "OVER LIMIT"
            if sz > hard_limit_bytes:
                ok = False
            print(f"  {fn}: {_human(sz)} {flag}")
    return ok


CRITICAL_SHAPE_PREFIXES = ("vision.",)
"""Parameter prefixes whose shape mismatch must raise instead of being skipped."""


def _vision_mismatch_hint(critical: list, tag: str) -> str:
    """Diagnose vision.proj mismatches by axis: columns mean feat_dim, rows mean d_model."""
    generic = (f"[{tag}] vision.* is the FeatureAdapter (d_in=feat_dim, d_out=d_model): "
               f"shape mismatch means this checkpoint does not match the current config.")
    proj = next((b for b in critical if str(b[0]).endswith("vision.proj.weight")), None)
    if proj is None or len(proj[1]) != 2 or len(proj[2]) != 2:
        return generic
    (ck_d_model, ck_feat), (m_d_model, m_feat) = proj[1], proj[2]
    lines = []
    if ck_feat != m_feat:
        lines.append(
            f"[{tag}] feat_dim differs (checkpoint {ck_feat} vs current {m_feat}): features "
            f"were extracted with a different backbone; re-extract with the same model.")
    if ck_d_model != m_d_model:
        lines.append(
            f"[{tag}] d_model differs (checkpoint {ck_d_model} vs current {m_d_model}): stages "
            f"did not share one --config. Features are reusable; retrain stage-1 with the "
            f"same --config.")
    return "\n".join(lines) if lines else generic


def load_checkpoint(model, path: str, strict: bool = False,
                    max_missing_ratio: float = 0.5, tag: str = "ckpt",
                    critical_prefixes: tuple = CRITICAL_SHAPE_PREFIXES):
    """Load weights into model, reporting missing/unexpected keys and shape skips.

    Shape-mismatched tensors are dropped and counted as missing; critical prefixes
    raise instead. Missing ratios above max_missing_ratio raise.
    """
    import torch as _torch
    if os.path.isdir(path):
        state = load_sharded(path)
    else:
        sd = _torch.load(path, map_location="cpu")
        state = sd.get("model", sd) if isinstance(sd, dict) else sd

    model_sd = model.state_dict()
    shape_bad = [(k, tuple(v.shape), tuple(model_sd[k].shape)) for k, v in state.items()
                 if k in model_sd and hasattr(v, "shape") and v.shape != model_sd[k].shape]
    if shape_bad:
        state = {k: v for k, v in state.items() if k not in {b[0] for b in shape_bad}}

    missing, unexpected = model.load_state_dict(state, strict=strict)
    n_total = len(model_sd)
    ratio = len(missing) / max(n_total, 1)
    print(f"[{tag}] loaded {path}: {n_total - len(missing)}/{n_total} tensors "
          f"(missing {len(missing)}, extra {len(unexpected)}, shape-skipped {len(shape_bad)})")
    if shape_bad:
        _s = ", ".join(f"{k}: {a}->{b}" for k, a, b in shape_bad[:4])
        print(f"[{tag}]   shape-skipped (kept init): {_s}"
              f"{' ...' if len(shape_bad) > 4 else ''}")
    if missing:
        print(f"[{tag}]   missing (kept init): {list(missing)[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[{tag}]   extra (ignored): {list(unexpected)[:6]}{' ...' if len(unexpected) > 6 else ''}")
    if ratio > max_missing_ratio:
        raise RuntimeError(
            f"[{tag}] checkpoint incompatible with model: {ratio:.0%} params missing "
            f"(threshold {max_missing_ratio:.0%}). Check that --config matches training.")
    critical = [b for b in shape_bad if str(b[0]).startswith(tuple(critical_prefixes or ()))]
    if critical:
        _d = "; ".join(f"{k}: checkpoint{a} vs model{b}" for k, a, b in critical[:4])
        raise RuntimeError(
            f"[{tag}] critical shape mismatch, refusing to drop silently: {_d}"
            f"{' ...' if len(critical) > 4 else ''}\n"
            f"{_vision_mismatch_hint(critical, tag)}\n"
            f"[{tag}] Pass critical_prefixes=() to drop these explicitly.")
    return missing, unexpected
