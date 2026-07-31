"""连续时间 ZOH 离散化算子（可变 Δt，复数对角 SSM）。

对角复数状态矩阵 A=diag(λ)，λ = -exp(a) + i·b（Re(λ)<0 恒稳定）。给定真实物理时间间隔 Δ，
零阶保持（ZOH）离散化（设计方案 §3(P2)、§4.1）：

    Ā(Δ) = exp(λΔ)                      （对角 → 逐元素复指数，精确，无需 scale-and-square）
    B̄(Δ) = (exp(λΔ) - 1)/λ · B           （λ 的实部严格<0，永不除零）

与固定步长 Mamba 的根本区别：Δ 是**真实物理时间**（秒），可任意可变，从而摆脱固定帧率约束。
数值稳定：|λΔ| 很小时 (exp(z)-1)/λ 存在抵消误差，用 Taylor 分支（complex_expm1）规避。
"""
from __future__ import annotations

import torch
from torch import Tensor

_TAYLOR_EPS = 1e-4  # |z| 小于此值时走 Taylor 展开，避免 exp(z)-1 的浮点抵消


def complex_expm1(z: Tensor) -> Tensor:
    """数值稳定的复数 expm1：exp(z)-1。

    torch 未提供复数 expm1；|z| 很小时直接 exp(z)-1 抵消严重，改用 3 阶 Taylor：
        z + z²/2 + z³/6
    """
    small = z.abs() < _TAYLOR_EPS
    taylor = z * (1.0 + z * (0.5 + z * (1.0 / 6.0)))
    exact = torch.exp(z) - 1.0
    return torch.where(small, taylor, exact)


def zoh_discretize(lam: Tensor, dt: Tensor, B: Tensor) -> tuple[Tensor, Tensor]:
    """把连续对角 SSM 按真实 Δt 做 ZOH 离散化。

    形状约定（H=通道数, N=状态维, [*]=前置批/时间维，通常为 (batch, length)）：
        lam : (H, N)        复数，λ = -exp(a)+i·b
        dt  : (*, H) 或 (*, 1)  实数正值，真实物理时间间隔（可含逐通道时标缩放）
        B   : (*, N)        复数或实数，选择性输入矩阵（Mamba 风格，跨通道广播）

    返回：
        dA  : (*, H, N) 复数，离散状态转移 Ā(Δ)=exp(λΔ)
        dB  : (*, H, N) 复数，离散输入矩阵 B̄(Δ)=(exp(λΔ)-1)/λ · B
    """
    # 广播到 (*, H, N)
    lam_e = lam  # (H, N)
    dt_e = dt.unsqueeze(-1)  # (*, H, 1)
    # z = λ·Δ  →  (*, H, N)
    z = lam_e * dt_e  # 广播 (H,N)*(*,H,1) -> (*,H,N)
    dA = torch.exp(z)
    # B̄ = expm1(z)/λ · B ；λ 实部<0 恒非零
    dB_bar = complex_expm1(z) / lam_e  # (*, H, N)
    # B (*, N) -> (*, 1, N) 跨通道广播
    B_e = B.unsqueeze(-2)
    if not torch.is_complex(B_e):
        B_e = B_e.to(dA.dtype)
    dB = dB_bar * B_e
    return dA, dB


def make_lambda(a_log_neg_real: Tensor, a_imag: Tensor) -> Tensor:
    """由参数化分量组装 λ = -exp(a) + i·b，保证 Re(λ)<0（无条件稳定）。

    a_log_neg_real, a_imag : (H, N) 实数
    返回 λ : (H, N) 复数
    """
    real = -torch.exp(a_log_neg_real)
    return torch.complex(real, a_imag)


def effective_dt(dt_phys: Tensor, log_dt_scale: Tensor | None = None,
                 dt_min: float = 1e-3, dt_max: float = 1e3) -> Tensor:
    """把真实物理 Δt（秒）转成逐通道有效 Δt。

    dt_phys      : (*,) 或 (*, 1) 真实时间间隔（秒）——保持"物理时间"语义（线性于真实 Δt）
    log_dt_scale : (H,) 可学习的逐通道时标缩放（log 空间），初始 0（=不缩放）
    返回          : (*, H) 有效 Δt，clamp 到 [dt_min, dt_max] 防数值发散
    """
    if dt_phys.dim() >= 1 and dt_phys.shape[-1] != 1:
        dt_phys = dt_phys.unsqueeze(-1)  # (*, 1)
    if log_dt_scale is None:
        out = dt_phys.expand(*dt_phys.shape[:-1], 1)
        return out.clamp(dt_min, dt_max)
    scale = torch.exp(log_dt_scale)  # (H,)
    out = dt_phys * scale  # (*,1)*(H,) -> (*,H)
    return out.clamp(dt_min, dt_max)
