"""Diagonal linear scan: h_k = a_k * h_{k-1} + b_k."""
from __future__ import annotations

import torch
from torch import Tensor


def _shift_right(x: Tensor, d: int, fill: float) -> Tensor:
    """Shift right along dim=1 by d, filling with fill."""
    if d == 0:
        return x
    pad = torch.full_like(x[:, :d], fill)
    return torch.cat([pad, x[:, :-d]], dim=1)


def associative_scan_diag(a: Tensor, b: Tensor, h_init: Tensor | None = None) -> Tensor:
    """Parallel associative scan returning state sequence h.

    Args a, b: (B, L, H, N) complex.
    """
    A = a
    Bp = b
    L = a.shape[1]
    d = 1
    while d < L:
        A_sh = _shift_right(A, d, 1.0)
        B_sh = _shift_right(Bp, d, 0.0)
        Bp = A * B_sh + Bp
        A = A_sh * A
        d *= 2
    if h_init is not None:
        Bp = Bp + A * h_init.unsqueeze(1)
    return Bp


def sequential_scan_diag(a: Tensor, b: Tensor, h_init: Tensor | None = None) -> Tensor:
    """Sequential scan. For validation use."""
    B, L, H, N = a.shape
    h = torch.zeros(B, H, N, dtype=b.dtype, device=b.device) if h_init is None else h_init
    outs = []
    for k in range(L):
        h = a[:, k] * h + b[:, k]
        outs.append(h)
    return torch.stack(outs, dim=1)


def chunked_scan_diag(a: Tensor, b: Tensor, chunk: int,
                      h_init: Tensor | None = None) -> Tensor:
    """Blocked scan with gradient checkpointing, equivalent to associative_scan_diag."""
    import torch.utils.checkpoint as _ckpt
    L = a.shape[1]
    if chunk <= 0 or L <= chunk or not torch.is_grad_enabled():
        return associative_scan_diag(a, b, h_init)
    outs, h = [], h_init
    for k0 in range(0, L, chunk):
        k1 = min(k0 + chunk, L)
        hk = _ckpt.checkpoint(associative_scan_diag, a[:, k0:k1], b[:, k0:k1], h,
                              use_reentrant=False)
        outs.append(hk)
        h = hk[:, -1]
    return torch.cat(outs, dim=1)


def selective_scan(a: Tensor, b: Tensor, h_init: Tensor | None = None,
                   backend: str = "auto", chunk: int = 0) -> Tensor:
    """Unified scan entry. Uses CUDA kernel on GPU, else PyTorch scan."""
    if backend != "pytorch" and a.is_cuda and h_init is None:
        try:
            from .scan_cuda import cuda_available, selective_scan_cuda
            if cuda_available():
                return selective_scan_cuda(a, b)
        except Exception:
            pass
    if backend == "cuda":
        raise RuntimeError("CUDA variable-step scan unavailable.")
    if chunk and chunk > 0:
        return chunked_scan_diag(a, b, chunk, h_init)
    return associative_scan_diag(a, b, h_init)
