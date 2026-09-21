"""Variable-step scan CUDA kernel wrapper with PyTorch fallback."""
from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.autograd import Function

_EXT = None
_TRIED = False


def _load():
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(__file__)
        src = os.path.join(here, "csrc", "eacs_scan_cuda.cu")
        _EXT = load(name="eacs_scan_cuda", sources=[src], verbose=False)
    except Exception as e:  # pragma: no cover
        print(f"[scan_cuda] JIT build failed, falling back to PyTorch scan: {e}")
        _EXT = None
    return _EXT


def cuda_available() -> bool:
    return _load() is not None


class _SelectiveScanCUDA(Function):
    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        ext = _load()
        h = ext.scan_fwd(a.contiguous(), b.contiguous())[0]
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h: Tensor):
        a, h = ctx.saved_tensors
        ext = _load()
        grad_a, grad_b = ext.scan_bwd(a.contiguous(), h.contiguous(), grad_h.contiguous())
        return grad_a, grad_b


def selective_scan_cuda(a: Tensor, b: Tensor) -> Tensor:
    """Differentiable CUDA scan; a, b must be CUDA complex64 with shape (B,L,H,N)."""
    return _SelectiveScanCUDA.apply(a, b)
