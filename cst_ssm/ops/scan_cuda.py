"""变步长扫描 CUDA 内核的 Python 封装：JIT 编译加载 + autograd.Function + 自动派发。

- 有 GPU 且能编译 → 用 CUDA 内核（`csrc/eacs_scan_cuda.cu`）。
- 否则 → 回退纯 PyTorch `associative_scan_diag`（数值等价，见 scan.py）。
接口与 `associative_scan_diag(a,b)` 一致：a,b 为 (B,L,H,N) complex64，返回状态序列 h。
"""
from __future__ import annotations

import os

import torch
from torch import Tensor
from torch.autograd import Function

_EXT = None
_TRIED = False


def _load():
    """惰性 JIT 编译并缓存扩展；失败或无 CUDA 返回 None。"""
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
    except Exception as e:  # pragma: no cover - 取决于构建环境
        print(f"[scan_cuda] JIT 编译失败，回退纯 PyTorch 扫描：{e}")
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
    """CUDA 变步长扫描（可微）。要求 a,b 在 cuda 上、complex64、(B,L,H,N)。"""
    return _SelectiveScanCUDA.apply(a, b)
