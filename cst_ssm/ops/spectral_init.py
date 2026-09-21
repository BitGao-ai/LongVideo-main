"""Multiscale spectral init: real parts from target memory horizons, imag from S4D-Lin."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BranchSpec:
    """Spectral config for one temporal branch."""
    name: str
    n_state: int          # State dim N.
    tau_min: float        # Memory horizon lower bound (seconds).
    tau_max: float        # Memory horizon upper bound (seconds).
    imag_scale: float = 1.0


DEFAULT_BRANCHES: tuple[BranchSpec, ...] = (
    BranchSpec("short", n_state=16, tau_min=0.1, tau_max=1.0, imag_scale=1.0),
    BranchSpec("mid", n_state=32, tau_min=10.0, tau_max=60.0, imag_scale=0.3),
    BranchSpec("long", n_state=64, tau_min=600.0, tau_max=3600.0, imag_scale=0.1),
)


def init_branch_lambda(spec: BranchSpec, n_channel: int,
                       generator: torch.Generator | None = None
                       ) -> tuple[Tensor, Tensor]:
    """Init one branch's lambda components; both outputs are (n_channel, n_state).

    Re(lambda) = -1/tau with log-uniform tau; Im(lambda) = pi * n (S4D-Lin).
    """
    N = spec.n_state
    H = n_channel
    log_tau = torch.empty(H, N)
    if generator is not None:
        log_tau.uniform_(math.log(spec.tau_min), math.log(spec.tau_max), generator=generator)
    else:
        log_tau.uniform_(math.log(spec.tau_min), math.log(spec.tau_max))
    a_log_neg_real = -log_tau

    n_idx = torch.arange(N, dtype=torch.float32)
    imag = math.pi * n_idx * spec.imag_scale
    a_imag = imag.unsqueeze(0).expand(H, N).contiguous()
    return a_log_neg_real, a_imag


def build_multiscale_lambdas(branches: tuple[BranchSpec, ...] | list[BranchSpec],
                             n_channel: int,
                             generator: torch.Generator | None = None
                             ) -> list[tuple[Tensor, Tensor]]:
    """Init all branches; returns [(a_log_neg_real, a_imag), ...]."""
    return [init_branch_lambda(spec, n_channel, generator) for spec in branches]


def target_real_band(spec: BranchSpec) -> tuple[float, float]:
    """Target eigenvalue real-part band [-1/tau_min, -1/tau_max] for spectral loss."""
    return (-1.0 / spec.tau_min, -1.0 / spec.tau_max)
