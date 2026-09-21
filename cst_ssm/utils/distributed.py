"""Single source of truth for DDP sharding info and process-group setup."""
from __future__ import annotations

import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist

__all__ = [
    "env_world_size", "env_rank", "env_local_rank", "launched_distributed",
    "process_group_ready", "dist_info", "get_world_size", "get_rank",
    "is_main_process", "resolve_device", "maybe_init_distributed",
    "check_ddp_launch", "check_device_binding", "add_ddp_args", "resolve_ddp",
    "barrier", "seed_everything", "cleanup_distributed",
]


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def env_world_size() -> int:
    return max(1, _env_int("WORLD_SIZE", 1))


def env_rank() -> int:
    return _env_int("RANK", 0)


def env_local_rank() -> int:
    return _env_int("LOCAL_RANK", 0)


def launched_distributed() -> bool:
    """True when launched via torchrun with more than one process."""
    return env_world_size() > 1


def process_group_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_info() -> tuple[int, int]:
    """Return (world_size, rank), valid before the process group is created."""
    if process_group_ready():
        return dist.get_world_size(), dist.get_rank()
    if launched_distributed():
        return env_world_size(), env_rank()
    return 1, 0


def get_world_size() -> int:
    return dist_info()[0]


def get_rank() -> int:
    return dist_info()[1]


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if not process_group_ready():
        return
    dist.barrier()


def seed_everything(seed: int, rank_offset: bool = True) -> None:
    """Seed Python/NumPy/Torch; each rank gets seed + rank by default."""
    _, rank = dist_info()
    s = seed + rank if rank_offset else seed
    random.seed(s)
    np.random.seed(s % (2 ** 32 - 1))
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def cleanup_distributed() -> None:
    if process_group_ready():
        dist.destroy_process_group()


def _explicit_cuda_index(device: str) -> int | None:
    """Extract index from "cuda:N"; None for bare "cuda" or non-CUDA."""
    if not device.startswith("cuda") or ":" not in device:
        return None
    try:
        return int(device.split(":", 1)[1])
    except (TypeError, ValueError):
        return None


def check_device_binding(device: str, script: str = "train script") -> None:
    """Exit when a multi-process run pins a fixed device that is not this rank's."""
    if not launched_distributed():
        return
    idx = _explicit_cuda_index(device)
    if idx is None or idx == env_local_rank():
        return
    sys.exit(
        f"[dist] torchrun detected (WORLD_SIZE={env_world_size()}) but --device "
        f"is fixed to {device!r} while LOCAL_RANK={env_local_rank()}. "
        f"Pass --device cuda and let each rank bind its own GPU.")


def resolve_device(device: str) -> str:
    """Map bare "cuda" to this rank's "cuda:{LOCAL_RANK}" under torchrun."""
    if not device.startswith("cuda") or ":" in device:
        check_device_binding(device)
        return device
    if not launched_distributed():
        return device
    return f"cuda:{env_local_rank()}"


def maybe_init_distributed(ddp: bool, backend: str = "nccl",
                           device: str = "cuda", verbose: bool = True) -> bool:
    """Idempotently create the process group. set_device runs before init."""
    if not ddp:
        return False
    if process_group_ready():
        return True

    local = env_local_rank()
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local)
    elif backend == "nccl":
        backend = "gloo"

    dist.init_process_group(backend=backend)
    if verbose and dist.get_rank() == 0:
        suffix = f", bound to cuda:{local}" if use_cuda else ""
        print(f"[dist] process group ready: backend={backend}, "
              f"world_size={dist.get_world_size()}{suffix}")
    return True


def check_ddp_launch(ddp: bool, script: str = "train script") -> None:
    """Exit when torchrun launched multiple processes but DDP is off."""
    if launched_distributed() and not ddp:
        sys.exit(
            f"[dist] torchrun detected (WORLD_SIZE={env_world_size()}) but DDP is off. "
            f"Pass --ddp to {script} or set train.ddp: true in YAML.")


def add_ddp_args(ap) -> None:
    """Attach shared DDP flags to an argparse parser."""
    ap.add_argument("--ddp", action="store_true", default=None,
                    help="Enable DDP multi-GPU training")
    ap.add_argument("--no-ddp", dest="ddp", action="store_false", help="Force DDP off")
    ap.add_argument("--ddp-backend", default=None, help="nccl (GPU) | gloo (CPU)")


def resolve_ddp(cli_ddp: bool | None, cli_backend: str | None,
                yaml_train: dict | None, script: str = "train script",
                device: str | None = None) -> tuple[bool, str]:
    """Resolve DDP flag (CLI > YAML > off) and validate the launch setup."""
    yt = yaml_train or {}
    ddp = cli_ddp if cli_ddp is not None else bool(yt.get("ddp", False))
    backend = cli_backend or yt.get("ddp_backend") or "nccl"
    check_ddp_launch(ddp, script)
    if device is not None:
        check_device_binding(device, script)
    return ddp, backend
