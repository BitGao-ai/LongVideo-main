"""Training-time memory optimizations: checkpointing, autocast, optimizer, schedules."""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from ..modules.eacs import EACSLayer
from ..modules.event_gate import EventGate


def set_eacs_chunk(model: nn.Module, chunk_size: int) -> int:
    """Set chunked gradient-checkpoint size on all EACS layers (0 disables)."""
    n = 0
    for m in model.modules():
        if isinstance(m, EACSLayer):
            m.chunk_size = chunk_size
            n += 1
    return n


def set_gate_temperature(model: nn.Module, t: float) -> None:
    """Set all event-gate temperatures."""
    for m in model.modules():
        if isinstance(m, EventGate):
            m.set_temperature(t)


def set_eps_anneal(model: nn.Module, v: float) -> None:
    """Set all event-gate epsilon anneal multipliers."""
    for m in model.modules():
        if isinstance(m, EventGate):
            m.set_eps_anneal(v)


def eps_anneal_schedule(step: int, total: int,
                        start: float = 1.0, end: float = 1.0) -> float:
    """Linear anneal from start to end; start == end == 1.0 disables annealing."""
    if total <= 1:
        return end
    frac = min(max(step / total, 0.0), 1.0)
    return float(start + (end - start) * frac)


@contextmanager
def autocast_ctx(enabled: bool = True, dtype: torch.dtype = torch.bfloat16,
                 device_type: str = "cuda"):
    if enabled and (device_type == "cuda" and torch.cuda.is_available() or device_type == "cpu"):
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield


def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05,
                    try_8bit: bool = True):
    """AdamW over trainable params; 8-bit Adam when bitsandbytes is available."""
    params = [p for p in model.parameters() if p.requires_grad]
    if try_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def gate_temperature_schedule(step: int, total: int,
                              t_start: float = 1.0, t_end: float = 0.05) -> float:
    """Exponential temperature anneal from soft gates early to sparse gates late."""
    if total <= 1:
        return t_end
    frac = min(max(step / total, 0.0), 1.0)
    return float(t_start * (t_end / t_start) ** frac)
