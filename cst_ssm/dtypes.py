"""Input dtype alignment so low-precision stored features stay low-precision."""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["autocast_dtype", "autocast_active", "align_to_param", "resolve_feat_dtype"]

_HALF = (torch.float16, torch.bfloat16)


def autocast_dtype(device_type: str) -> torch.dtype | None:
    """Active autocast dtype for the device, or None when autocast is off."""
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
        return None
    except (TypeError, AttributeError, RuntimeError):
        pass
    try:
        if device_type == "cuda":
            if torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
        elif device_type == "cpu":
            if torch.is_autocast_cpu_enabled():
                return torch.get_autocast_cpu_dtype()
    except (TypeError, AttributeError, RuntimeError):
        return None
    return None


def autocast_active(device_type: str) -> bool:
    return autocast_dtype(device_type) is not None


def align_to_param(x: Tensor, param: Tensor) -> Tensor:
    """Align input dtype to a parameter; passes through inside autocast regions."""
    if x.dtype == param.dtype or not x.is_floating_point():
        return x
    if autocast_active(x.device.type):
        return x
    return x.to(param.dtype)


def resolve_feat_dtype(name: str | None):
    """Parse a feat_dtype string to a torch dtype; "auto"/None keeps file dtype."""
    if not name or name == "auto":
        return None
    table = {"float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
             "float32": torch.float32, "fp32": torch.float32}
    if name not in table:
        raise ValueError(
            f"unknown feat_dtype={name!r}; expected auto / float16 / bfloat16 / float32")
    return table[name]
