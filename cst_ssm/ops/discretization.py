"""Zero-order-hold discretization for diagonal complex SSM with variable dt."""
from __future__ import annotations

import torch
from torch import Tensor

_TAYLOR_EPS = 1e-4


def complex_expm1(z: Tensor, exp_z: Tensor | None = None) -> Tensor:
    """Stable complex expm1: exp(z) - 1."""
    a, b = z.real, z.imag
    em1 = torch.expm1(a)
    sb, cb = torch.sin(b), torch.cos(b)
    re = em1 * cb - 2.0 * torch.sin(0.5 * b) ** 2
    im = (em1 + 1.0) * sb
    return torch.complex(re, im)


def zoh_kernels(lam: Tensor, dt: Tensor,
                inv_lam: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """Discretization kernels from (lambda, dt): returns (dA, dB_bar).

    Args lam: (H, N); dt: (*, H). Returns (*, H, N).
    """
    z = lam * dt.unsqueeze(-1)
    a, b = z.real, z.imag
    em1 = torch.expm1(a)
    ea = em1 + 1.0
    sb, cb = torch.sin(b), torch.cos(b)
    im = ea * sb
    dA = torch.complex(ea * cb, im)
    expm1_z = torch.complex(em1 * cb - 2.0 * torch.sin(0.5 * b) ** 2, im)
    if inv_lam is None:
        inv_lam = torch.reciprocal(lam)
    return dA, expm1_z * inv_lam


def zoh_apply_B(dB_bar: Tensor, B: Tensor) -> Tensor:
    """Apply selective B: dB = dB_bar * B."""
    B_e = B.unsqueeze(-2)
    if not torch.is_complex(B_e):
        B_e = B_e.to(dB_bar.dtype)
    return dB_bar * B_e


def zoh_discretize(lam: Tensor, dt: Tensor, B: Tensor,
                   inv_lam: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """ZOH discretization: returns (dA, dB).

    Args lam: (H, N) complex; dt: (*, H); B: (*, N).
    """
    dA, dB_bar = zoh_kernels(lam, dt, inv_lam)
    return dA, zoh_apply_B(dB_bar, B)


def make_lambda(a_log_neg_real: Tensor, a_imag: Tensor) -> Tensor:
    """Build lambda = -exp(a) + i*b. Returns (H, N) complex."""
    real = -torch.exp(a_log_neg_real.float())
    return torch.complex(real, a_imag.float())


def _clamp_dt(out: Tensor, dt_min: float, dt_max) -> Tensor:
    """Clamp to [dt_min, dt_max]; dt_max may be scalar or tensor."""
    out = out.clamp_min(dt_min)
    return torch.minimum(out, dt_max) if torch.is_tensor(dt_max) else out.clamp_max(dt_max)


def effective_dt(dt_phys: Tensor, log_dt_scale: Tensor | None = None,
                 dt_min: float = 1e-3, dt_max: float | Tensor = 1e3) -> Tensor:
    """Map physical dt (seconds) to per-channel effective dt.

    Args dt_phys: (*,) or (*, 1); log_dt_scale: (H,). Returns (*, H).
    """
    if dt_phys.dim() >= 1 and dt_phys.shape[-1] != 1:
        dt_phys = dt_phys.unsqueeze(-1)
    if log_dt_scale is None:
        out = dt_phys.expand(*dt_phys.shape[:-1], 1)
        return _clamp_dt(out, dt_min, dt_max)
    scale = torch.exp(log_dt_scale)
    out = dt_phys * scale
    return _clamp_dt(out, dt_min, dt_max)
