"""通用（非对角）矩阵指数：Scale-and-Square + 截断 Taylor（设计 §7）。

说明：对角复数 SSM 路径用逐元素复指数（见 ops/discretization.py），**精确且无需本模块**。
本模块用于未来非对角 A（如原始稠密 HiPPO 矩阵）或作参考/数值对拍。
- matrix_exp            : 生产用，torch 内置（本身即 Scale-and-Square + Padé），支持批量/复数。
- matrix_exp_scale_square: 手写 Scale-and-Square + Taylor 参考实现（教学/对拍）。
- zoh_discretize_dense  : 非对角 ZOH 离散化 Ā=exp(AΔ), B̄=A⁻¹(exp(AΔ)−I)B（用 matrix_exp）。
"""
from __future__ import annotations

import torch
from torch import Tensor


def matrix_exp(A: Tensor) -> Tensor:
    """exp(A)（torch 内置 Scale-and-Square + Padé）。A: (..., n, n)，支持复数。"""
    return torch.matrix_exp(A)


def matrix_exp_scale_square(A: Tensor, taylor_order: int = 8) -> Tensor:
    """手写 Scale-and-Square + 截断 Taylor：exp(A)=(exp(A/2^s))^{2^s}。

    ‖A‖ 大时先缩放 A/2^s 使范数 ≤1，Taylor 高精度算 exp(A/2^s)，再平方 s 次。
    """
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
    """非对角 ZOH：Ā=exp(AΔ)，B̄=A⁻¹(exp(AΔ)−I)B（A 奇异时退化为对角逐标量，见 discretization）。"""
    n = A.shape[-1]
    Abar = matrix_exp(A * dt)
    eye = torch.eye(n, dtype=A.dtype, device=A.device)
    try:
        Bbar = torch.linalg.solve(A, (Abar - eye)) @ B
    except Exception:                                # A 近奇异 → 用 (A+εI) 正则
        Bbar = torch.linalg.solve(A + eps * eye, (Abar - eye)) @ B
    return Abar, Bbar
