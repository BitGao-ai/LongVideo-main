"""Dense (non-diagonal) matrix exponential: scale-and-square with Taylor fallback."""
from __future__ import annotations

import torch
from torch import Tensor


def matrix_exp(A: Tensor) -> Tensor:
    """exp(A) via torch built-in. A: (..., n, n), real or complex."""
    return torch.matrix_exp(A)


def matrix_exp_scale_square(A: Tensor, taylor_order: int = 8) -> Tensor:
    """Reference scale-and-square Taylor implementation: exp(A) = (exp(A/2^s))^{2^s}."""
    n = A.shape[-1]
    norm = torch.linalg.matrix_norm(A, ord=1).max()
    s = int(max(0, torch.ceil(torch.log2(norm.clamp_min(1e-12))).item())) if norm > 1 else 0
    B = A / (2.0 ** s)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    eye = eye.expand_as(A) if A.dim() > 2 else eye
    term = eye.clone()
    E = eye.clone()
    for k in range(1, taylor_order + 1):
        term = term @ B / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def zoh_discretize_dense(A: Tensor, dt: float, B: Tensor,
                         eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Dense ZOH: Abar = exp(A*dt), Bbar = A^{-1}(Abar - I)B."""
    n = A.shape[-1]
    Abar = matrix_exp(A * dt)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    try:
        Bbar = torch.linalg.solve(A, (Abar - eye)) @ B
    except Exception:
        Bbar = torch.linalg.solve(A + eps * eye, (Abar - eye)) @ B
    return Abar, Bbar
