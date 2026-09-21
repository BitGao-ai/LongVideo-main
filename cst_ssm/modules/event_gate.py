"""Predictive-coding event gate: skip state updates when the residual is small."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RunningStandardizer(nn.Module):
    """Running per-feature mean/var normalization without affine parameters."""

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            flat = x.reshape(-1, x.shape[-1])
            mean = flat.mean(0)
            var = flat.var(0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) / torch.sqrt(var + self.eps)

    def apply_stats(self, x: Tensor) -> Tensor:
        """Normalize with buffered stats without updating them."""
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)


class EventGate(nn.Module):
    """Gate state updates on the normalized prediction residual.

    Args:
        eps_min/eps_max: bounds for the learnable threshold.
        temperature: soft-gate temperature (annealed externally).
        use_ste: straight-through estimator (hard forward, soft backward).
        gate_kind: "event" | "random" | "always".
        cpi_modulation: scales the threshold by a frame-level CPI signal.
    """

    def __init__(self, eps_min: float = 0.01, eps_max: float = 0.9,
                 init_eps: float = 0.1, temperature: float = 0.1,
                 use_ste: bool = True, norm_eta: float = 1e-6,
                 gate_kind: str = "event", random_rate: float = 0.5,
                 cpi_modulation: float = 0.0):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.norm_eta = norm_eta
        self.use_ste = use_ste
        self.gate_kind = gate_kind
        self.random_rate = random_rate
        self.cpi_modulation = cpi_modulation
        frac = (init_eps - eps_min) / max(eps_max - eps_min, 1e-6)
        frac = min(max(frac, 1e-4), 1 - 1e-4)
        raw = torch.logit(torch.tensor(frac))
        self.eps_raw = nn.Parameter(raw)
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("eps_anneal", torch.tensor(1.0))

    @property
    def eps(self) -> Tensor:
        base = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(self.eps_raw)
        return base * self.eps_anneal

    def set_temperature(self, t: float) -> None:
        self.temperature.fill_(float(t))

    def set_eps_anneal(self, v: float) -> None:
        self.eps_anneal.fill_(float(v))

    def residual(self, obs: Tensor, pred: Tensor) -> Tensor:
        """Normalized residual r = ||obs - pred|| / (||obs|| + eta) over the last dim."""
        num = torch.linalg.vector_norm(obs - pred, dim=-1)
        den = torch.linalg.vector_norm(obs, dim=-1) + self.norm_eta
        return num / den

    def forward(self, obs: Tensor, pred: Tensor,
                eps_override: Tensor | None = None,
                cpi: Tensor | None = None,
                temp_override: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return (gate, residual). Training uses soft gates; eval uses hard gates."""
        r = self.residual(obs, pred)
        if self.gate_kind == "always":
            return torch.ones_like(r), r
        if self.gate_kind == "random":
            return (torch.rand_like(r) < self.random_rate).to(r.dtype), r
        eps = self.eps if eps_override is None else eps_override
        if cpi is not None and self.cpi_modulation > 0:
            eps = (eps * (1.0 - self.cpi_modulation * cpi.clamp(0, 1))).clamp_min(1e-3)
        temp = self.temperature if temp_override is None else temp_override
        soft = torch.sigmoid((r - eps) / temp.clamp_min(1e-4))
        if not self.training:
            return (r > eps).to(r.dtype), r
        if self.use_ste:
            hard = (r > eps).to(r.dtype)
            gate = hard + (soft - soft.detach())
            return gate, r
        return soft, r
