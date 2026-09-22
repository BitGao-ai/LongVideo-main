"""Shared helpers for offline batch scripts (no torch dependency)."""
from __future__ import annotations

import os
import time


_VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".flv", ".ts")


def feature_stem(name: str) -> str:
    """Normalize a video id or relative path into a filesystem-safe feature stem.

    Path separators become '_' so ids stay unique across subdirectories, and a
    trailing video extension is stripped. Shared by quality_filter and
    convert_benchmarks so both name features identically.
    """
    stem = str(name).strip().replace("\\", "_").replace("/", "_")
    low = stem.lower()
    for ext in _VIDEO_EXTS:
        if low.endswith(ext):
            stem = stem[: -len(ext)]
            break
    stem = stem.strip()
    return stem or str(name).strip()


def get_dist_info() -> tuple[int, int, int]:
    """Return (rank, world_size, local_rank); (0, 1, 0) outside torchrun."""
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError:
        rank = 0
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    except ValueError:
        local_rank = rank
    if world < 1:
        world = 1
    rank = max(0, min(rank, world - 1))
    local_rank = max(0, local_rank)
    return rank, world, local_rank


def is_distributed() -> bool:
    _, world, _ = get_dist_info()
    return world > 1


def log_prefix() -> str:
    rank, world, _ = get_dist_info()
    return f"[rank {rank}/{world}] " if world > 1 else ""


def resolve_shard(shard_arg: str | None) -> tuple[int, int, str | None, str]:
    """Resolve --shard plus torchrun env into (rank, world, label, source)."""
    if shard_arg and shard_arg != "auto":
        i, n = map(int, shard_arg.split("/"))
        if not (0 <= i < n):
            raise ValueError(f"invalid --shard: {shard_arg!r}, expected 'i/N' with 0<=i<N")
        return i, n, shard_arg, "manual"
    rank, world, _ = get_dist_info()
    if shard_arg == "auto" or world > 1:
        return rank, world, f"{rank}/{world}", "auto"
    return 0, 1, None, "single"


def shard_list(items: list, rank: int, world: int) -> list:
    """Strided shard: items[rank::world]; disjoint across ranks."""
    if world <= 1:
        return items
    return items[rank::world]


def map_device_for_rank(device: str, local_rank: int) -> str:
    """Map bare 'cuda'/'auto' to this rank's GPU under torchrun."""
    _, world, _ = get_dist_info()
    if world <= 1:
        return device
    if device == "cuda":
        return f"cuda:{local_rank}"
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{local_rank}"
        except Exception:
            pass
    return device


def maybe_set_cuda_device(local_rank: int) -> None:
    """Set the default CUDA device for this rank; no-op without torch/CUDA."""
    _, world, _ = get_dist_info()
    if world <= 1:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
    except Exception:
        pass


def part_path(base: str, rank: int, world: int) -> str:
    """Per-rank output path so parallel ranks never share one file."""
    return f"{base}.part-{rank:03d}-of-{world:03d}"


def mark_done(path: str) -> None:
    try:
        with open(path + ".done", "w") as f:
            f.write("ok")
    except Exception:
        pass


def wait_for_parts(base: str, world: int, timeout_s: float = 1800.0) -> bool:
    """Rank 0 waits for all part files plus .done markers. Returns True when complete."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        ok = True
        for r in range(world):
            p = part_path(base, r, world)
            if not (os.path.exists(p) and os.path.exists(p + ".done")):
                ok = False
                break
        if ok:
            return True
        time.sleep(2.0)
    return False


def merge_jsonl_parts(base: str, world: int) -> int:
    """Concatenate part files in rank order into base. Returns row count."""
    n = 0
    with open(base, "w", encoding="utf-8") as out:
        for r in range(world):
            p = part_path(base, r, world)
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n
