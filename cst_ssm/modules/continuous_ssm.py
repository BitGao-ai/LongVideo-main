"""Ungated continuous SSM layer: parallel variable-step scan baseline."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..ops.discretization import make_lambda, zoh_discretize, effective_dt
from ..ops.spectral_init import BranchSpec, init_branch_lambda
from ..ops.scan import selective_scan


class ContinuousSSMLayer(nn.Module):
    """Variable-dt diagonal selective SSM (pre-LN residual block)."""

    def __init__(self, d_model: int, spec: BranchSpec, add_residual: bool = True,
                 chunk_size: int = 0):
        super().__init__()
        self.H = d_model
        self.N = spec.n_state
        self.add_residual = add_residual
        self.chunk_size = chunk_size
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
        u = self.proj_u(xn).float()
        b = self.proj_B(xn).float()
        Bc = torch.complex(b[..., :self.N], b[..., self.N:])
        c = self.proj_C(xn).float()
        Cc = torch.complex(c[..., :self.N], c[..., self.N:])
        lam = make_lambda(self.a_log_neg_real, self.a_imag).to(torch.complex64)

        t = timestamps.float()
        dt = torch.zeros_like(t)
        dt[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        if t.shape[1] > 1:
            dt[:, 0] = dt[:, 1]
        dt_eff = effective_dt(dt.unsqueeze(-1), self.log_dt_scale)

        dA, dB = zoh_discretize(lam, dt_eff, Bc,
                                inv_lam=torch.reciprocal(lam))
        dBu = dB * u.unsqueeze(-1)
        h = selective_scan(dA, dBu, chunk=self.chunk_size)
        y = torch.einsum("bln,blhn->blh", Cc, h).real + self.D * u
        y = self.proj_out(y.to(x_in.dtype))
        return x_in + y if self.add_residual else y
