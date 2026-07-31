"""连续 SSM 层（无事件门控）——CUDA 变步长内核的直接使用点 + 消融基线。

用途：
  1) 设计方案 §6.3 消融组 1/2 的"连续 SSM（不含事件门控）"基线；
  2) 演示 selective_scan（GPU 上走变步长 CUDA 内核，CPU 回退纯 PyTorch）的实际调用。
与 EACS 的区别：这里是**纯线性一阶递推**（h_k=Ā_k h_{k-1}+B̄_k u_k），无门控破坏 associativity，
因此可整段并行扫描；事件门控稀疏更新见 modules/eacs.py。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..ops.discretization import make_lambda, zoh_discretize, effective_dt
from ..ops.spectral_init import BranchSpec, init_branch_lambda
from ..ops.scan import selective_scan


class ContinuousSSMLayer(nn.Module):
    """可变 Δt 的连续对角选择性 SSM（pre-LN 残差块）。"""

    def __init__(self, d_model: int, spec: BranchSpec, add_residual: bool = True):
        super().__init__()
        self.H = d_model
        self.N = spec.n_state
        self.add_residual = add_residual
        a_lnr, a_imag = init_branch_lambda(spec, self.H)
        self.a_log_neg_real = nn.Parameter(a_lnr)
        self.a_imag = nn.Parameter(a_imag)
        self.log_dt_scale = nn.Parameter(torch.zeros(self.H))
        self.norm = nn.LayerNorm(d_model)
        self.proj_u = nn.Linear(d_model, self.H)
        self.proj_B = nn.Linear(d_model, 2 * self.N)
        self.proj_C = nn.Linear(d_model, 2 * self.N)
        self.D = nn.Parameter(torch.zeros(self.H))
        self.proj_out = nn.Linear(self.H, d_model)

    def forward(self, x: Tensor, timestamps: Tensor) -> Tensor:
        x_in = x
        xn = self.norm(x)
        u = self.proj_u(xn).float()                                  # (B,L,H)
        b = self.proj_B(xn); Bc = torch.complex(b[..., :self.N], b[..., self.N:]).to(torch.complex64)
        c = self.proj_C(xn); Cc = torch.complex(c[..., :self.N], c[..., self.N:]).to(torch.complex64)
        lam = make_lambda(self.a_log_neg_real, self.a_imag).to(torch.complex64)

        # 逐步真实 Δt（首帧用次帧间隔兜底）→ 变步长离散化
        t = timestamps.float()
        dt = torch.zeros_like(t)
        dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        if t.shape[1] > 1:
            dt[:, 0] = dt[:, 1]
        dt_eff = effective_dt(dt.unsqueeze(-1), self.log_dt_scale)   # (B,L,H)

        dA, dB = zoh_discretize(lam, dt_eff, Bc)                     # (B,L,H,N)
        dBu = dB * u.unsqueeze(-1)                                    # 输入注入
        h = selective_scan(dA, dBu)                                  # ← 变步长扫描（CUDA/PyTorch）
        y = torch.einsum("bln,blhn->blh", Cc, h).real + self.D * u   # 读出 + 直连
        y = self.proj_out(y.to(x_in.dtype))
        return x_in + y if self.add_residual else y
