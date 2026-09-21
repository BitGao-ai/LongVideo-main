"""Event-adaptive continuous scan: physical-dt state evolution with sparse updates.

Forward uses segment form: prediction and update both evolve from the last
committed state x_pi over the gap (t_k - t_pi):
    predict (skip): xhat_k = Abar(t_k-t_pi) x_pi + Bbar(t_k-t_pi; B_pi) u_pi
    update:         x_k   = Abar(t_k-t_pi) x_pi + Bbar(t_k-t_pi; B_k) u_k
    gate:           r_k = ||u_k - yhat_k|| / (||u_k|| + eta); skips keep commits.
Single-branch cost is O(L) sequential (gating breaks associativity).
SSM math runs in float32 complex; outputs are cast back to the input dtype.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torch import Tensor

from ..ops.discretization import (make_lambda, zoh_kernels, zoh_apply_B,
                                  effective_dt)
from ..ops.spectral_init import BranchSpec, init_branch_lambda, target_real_band
from .event_gate import EventGate, RunningStandardizer


@dataclass
class EACSOutput:
    y: Tensor              # (B, L, d_model) temporal output
    gates: Tensor          # (B, L) per-step gates (1 = update, 0 = skip)
    update_rate: Tensor    # scalar mean(gate)
    residual: Tensor       # (B, L) normalized prediction residuals


def _readout(h: Tensor, C: Tensor) -> Tensor:
    """Complex state readout y = Re(sum_n C_n * h_n); supports merged branches."""
    eq = "bn,bhn->bh" if h.dim() == 3 else "bsn,bshn->bsh"
    return torch.einsum(eq, C, h).real


@dataclass
class StepParams:
    """All parameters for one segment cell; shared by single and merged paths."""
    lam: Tensor
    inv_lam: Tensor
    log_dt_scale: Tensor
    D: Tensor
    obs_mean: Tensor
    obs_var: Tensor
    obs_eps: float
    gate: EventGate
    disc_mode: str
    dt_init: float
    dt_max: Tensor | float
    proj_dt: nn.Module | None = None
    eps: Tensor | None = None
    temperature: Tensor | None = None


@dataclass
class RobustCfg:
    """Residual-spike fallback hyperparameters."""
    k: int
    spike_ratio: float
    eps_factor: float


def _standardize(x: Tensor, p: StepParams) -> Tensor:
    return (x - p.obs_mean) / torch.sqrt(p.obs_var + p.obs_eps)


def eacs_step(p: StepParams, h_pi, t_pi, u_pi, B_pi, C_pi,
              t_k, u_k, B_k, C_k, eps_override=None, cpi_k=None):
    """One segment cell; returns (h_cur, y_k, gate, r, new commits...)."""
    if p.disc_mode == "learned":
        dt_eff = F.softplus(p.proj_dt(u_k)).clamp(1e-3, 1e3)
    else:
        delta = (t_k - t_pi).clamp_min(0.0)
        if p.disc_mode == "fixed":
            delta = torch.full_like(delta, float(p.dt_init))
        dt_eff = effective_dt(delta.unsqueeze(-1), p.log_dt_scale,
                              dt_max=p.dt_max)

    dA, dB_bar = zoh_kernels(p.lam, dt_eff, p.inv_lam)
    h_free = dA * h_pi

    x_hat = h_free + zoh_apply_B(dB_bar, B_pi) * u_pi.unsqueeze(-1)
    y_hat = _readout(x_hat, C_pi)

    obs = _standardize(u_k, p)
    eps_ov = eps_override if eps_override is not None else p.eps
    g, r = p.gate(obs, _standardize(y_hat, p), eps_override=eps_ov,
                  cpi=cpi_k, temp_override=p.temperature)

    h_upd = h_free + zoh_apply_B(dB_bar, B_k) * u_k.unsqueeze(-1)

    gs = g[..., None, None]
    gv = g[..., None]
    g_upd = gs * h_upd
    one_g = 1 - gs
    h_cur = g_upd + one_g * x_hat
    h_pi_n = g_upd + one_g * h_pi
    t_pi_n = g * t_k + (1 - g) * t_pi
    u_pi_n = gv * u_k + (1 - gv) * u_pi
    B_pi_n = gv * B_k + (1 - gv) * B_pi
    C_pi_n = gv * C_k + (1 - gv) * C_pi

    y_k = _readout(h_cur, C_k) + p.D * u_k
    return h_cur, y_k, g, r, h_pi_n, t_pi_n, u_pi_n, B_pi_n, C_pi_n


def scan_range(p: StepParams, state, u, Bc, Cc, t, k0, k1,
               cpi=None, robust: RobustCfg | None = None):
    """Scan [k0, k1); returns (ys, gs, rs, final state)."""
    h_pi, t_pi, u_pi, B_pi, C_pi = state
    ys, gs, rs = [], [], []
    spike = prev_r = base_eps = None
    if robust is not None:
        spike = torch.zeros_like(t_pi)
        base_eps = p.gate.eps if p.eps is None else p.eps
    for k in range(k0, k1):
        eps_ov = None
        if robust is not None:
            eps_ov = torch.where(spike >= robust.k,
                                 base_eps * robust.eps_factor, base_eps)
        h_cur, y_k, g, r, h_pi, t_pi, u_pi, B_pi, C_pi = eacs_step(
            p, h_pi, t_pi, u_pi, B_pi, C_pi,
            t[:, k], u[:, k], Bc[:, k], Cc[:, k], eps_override=eps_ov,
            cpi_k=(cpi[:, k] if cpi is not None else None))
        ys.append(y_k); gs.append(g); rs.append(r)
        if robust is not None:
            if prev_r is not None:
                is_spike = r > prev_r * robust.spike_ratio
                spike = torch.where(is_spike, spike + 1.0, torch.zeros_like(spike))
            prev_r = r.detach()
    ys = torch.stack(ys, 1); gs = torch.stack(gs, 1); rs = torch.stack(rs, 1)
    return ys, gs, rs, (h_pi, t_pi, u_pi, B_pi, C_pi)



def _write_chunk(dst, part, total: int, k0: int, k1: int):
    """Write one chunk into a preallocated time-axis buffer; avoids peak doubling of cat."""
    if dst is None:
        dst = part.new_empty(part.shape[0], total, *part.shape[2:])
    dst[:, k0:k1] = part
    return dst


class EACSLayer(nn.Module):
    """Single-scale EACS layer (pre-LN residual block)."""

    def __init__(self, d_model: int, spec: BranchSpec,
                 gate: EventGate | None = None,
                 dt_init: float = 1.0, chunk_size: int = 0,
                 add_residual: bool = True, fused_train: bool = False,
                 robust_guard: bool = False, robust_k: int = 3,
                 robust_spike_ratio: float = 2.0, robust_eps_factor: float = 0.3,
                 disc_mode: str = "continuous", use_spectral_init: bool = True):
        super().__init__()
        self.d_model = d_model
        self.spec = spec
        self.H = d_model
        self.N = spec.n_state
        self.dt_init = dt_init
        self.chunk_size = chunk_size
        self.add_residual = add_residual
        self.fused_train = fused_train
        self.robust_guard = robust_guard
        self.robust_k = robust_k
        self.robust_spike_ratio = robust_spike_ratio
        self.robust_eps_factor = robust_eps_factor
        self.disc_mode = disc_mode
        self.use_spectral_init = use_spectral_init
        self.dt_max = max(1e3, 10.0 * float(spec.tau_max))

        if use_spectral_init:
            a_lnr, a_imag = init_branch_lambda(spec, self.H)
        else:
            a_lnr = torch.rand(self.H, self.N) * 4.0 - 2.0
            a_imag = torch.randn(self.H, self.N)
        self.a_log_neg_real = nn.Parameter(a_lnr)
        self.a_imag = nn.Parameter(a_imag)
        self.log_dt_scale = nn.Parameter(torch.zeros(self.H))
        self.proj_dt = nn.Linear(self.H, self.H) if disc_mode == "learned" else None

        self.norm = nn.LayerNorm(d_model)
        self.proj_u = nn.Linear(d_model, self.H)
        self.proj_B = nn.Linear(d_model, 2 * self.N)
        self.proj_C = nn.Linear(d_model, 2 * self.N)
        self.D = nn.Parameter(torch.zeros(self.H))
        self.proj_out = nn.Linear(self.H, d_model)

        self.obs_norm = RunningStandardizer(self.H)
        self.gate = gate if gate is not None else EventGate()

        lo, hi = target_real_band(spec)
        self.register_buffer("band_lo", torch.tensor(lo))
        self.register_buffer("band_hi", torch.tensor(hi))

    def lam(self) -> Tensor:
        return make_lambda(self.a_log_neg_real, self.a_imag)

    def _fast_paths_ok(self) -> bool:
        """Fused kernels require continuous discretization, event gating, no guard or CPI."""
        return (self.disc_mode == "continuous"
                and getattr(self.gate, "gate_kind", "event") == "event"
                and not self.robust_guard
                and getattr(self.gate, "cpi_modulation", 0.0) == 0)

    def _cuda_fused_ok(self, x: Tensor) -> bool:
        if self.training or not x.is_cuda or not self._fast_paths_ok():
            return False
        try:
            from ..ops.eacs_gated_scan import gated_kernel_available
            return bool(gated_kernel_available())
        except Exception:
            return False

    def uses_sequential_scan(self, x: Tensor) -> bool:
        """True when this layer runs a sequential scan (merge candidate for multiscale)."""
        if self._cuda_fused_ok(x):
            return False
        return not (self.training and self.fused_train and self._fast_paths_ok())

    def _project(self, x: Tensor):
        u = self.proj_u(x).float()
        b = self.proj_B(x).float()
        Bc = torch.complex(b[..., :self.N], b[..., self.N:])
        c = self.proj_C(x).float()
        Cc = torch.complex(c[..., :self.N], c[..., self.N:])
        return u, Bc, Cc

    def step_params(self, lam: Tensor) -> StepParams:
        return StepParams(
            lam=lam, inv_lam=torch.reciprocal(lam), log_dt_scale=self.log_dt_scale, D=self.D,
            obs_mean=self.obs_norm.running_mean, obs_var=self.obs_norm.running_var,
            obs_eps=self.obs_norm.eps, gate=self.gate, disc_mode=self.disc_mode,
            dt_init=self.dt_init, dt_max=self.dt_max, proj_dt=self.proj_dt)

    def _robust_cfg(self) -> RobustCfg | None:
        return (RobustCfg(self.robust_k, self.robust_spike_ratio, self.robust_eps_factor)
                if self.robust_guard else None)

    def _step(self, lam, h_pi, t_pi, u_pi, B_pi, C_pi,
              t_k, u_k, B_k, C_k, eps_override=None, cpi_k=None):
        return eacs_step(self.step_params(lam), h_pi, t_pi, u_pi, B_pi, C_pi,
                         t_k, u_k, B_k, C_k, eps_override=eps_override, cpi_k=cpi_k)

    def _scan_range(self, lam, state, u, Bc, Cc, t, k0, k1, cpi=None):
        """Scan [k0, k1); cpi is an optional (B,L) frame-level signal."""
        return scan_range(self.step_params(lam), state, u, Bc, Cc, t, k0, k1,
                          cpi=cpi, robust=self._robust_cfg())

    def _update_obs_stats(self, u: Tensor, frame_mask: Tensor | None) -> None:
        """Update running observation stats from valid frames only."""
        obs = u.reshape(-1, self.H) if frame_mask is None else u[frame_mask.bool()]
        if obs.numel():
            self.obs_norm(obs)

    def forward(self, x: Tensor, timestamps: Tensor, cpi: Tensor | None = None,
                frame_mask: Tensor | None = None) -> EACSOutput:
        """x (B,L,d_model), timestamps (B,L) seconds; optional cpi (B,L) and frame mask."""
        B, L, _ = x.shape
        if self._cuda_fused_ok(x):
            try:
                return self._forward_fused(x, timestamps)
            except Exception:
                pass
        x_in = x
        x = self.norm(x)
        u, Bc, Cc = self._project(x)
        lam = self.lam().to(torch.complex64)
        t = timestamps.float()

        if self.training and self.fused_train and self._fast_paths_ok():
            with torch.no_grad():
                self._update_obs_stats(u, frame_mask)
            from ..ops.eacs_cell_train import fused_gated_scan
            ys, gs, rs = fused_gated_scan(
                u, Bc, Cc, t, lam, self.log_dt_scale, self.D, self.gate.eps,
                self.obs_norm.running_mean, self.obs_norm.running_var,
                float(self.gate.temperature), self.gate.norm_eta,
                self.obs_norm.eps, self.dt_init, gate_mode="ste", use_kernel=True,
                dt_max=self.dt_max)
            delta = self.proj_out(ys.to(x_in.dtype))
            y = x_in + delta if self.add_residual else delta
            return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs)

        if self.training:
            with torch.no_grad():
                self._update_obs_stats(u, frame_mask)
        dt0 = self.dt_init
        h_pi = torch.zeros(B, self.H, self.N, dtype=torch.complex64, device=x.device)
        t_pi = t[:, 0] - dt0
        u_pi = u[:, 0]
        B_pi = Bc[:, 0]
        C_pi = Cc[:, 0]
        state = (h_pi, t_pi, u_pi, B_pi, C_pi)

        cs = self.chunk_size
        if self.training and cs and cs > 0 and L > cs:
            ys = gs = rs = None
            for k0 in range(0, L, cs):
                k1 = min(k0 + cs, L)
                y_c, g_c, r_c, state = ckpt.checkpoint(
                    self._scan_range, lam, state, u, Bc, Cc, t, k0, k1, cpi,
                    use_reentrant=False)
                ys = _write_chunk(ys, y_c, L, k0, k1)
                gs = _write_chunk(gs, g_c, L, k0, k1)
                rs = _write_chunk(rs, r_c, L, k0, k1)
                del y_c, g_c, r_c
        else:
            ys, gs, rs, state = self._scan_range(lam, state, u, Bc, Cc, t, 0, L, cpi=cpi)

        delta = self.proj_out(ys.to(x_in.dtype))
        y = x_in + delta if self.add_residual else delta
        return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs)

    @torch.no_grad()
    def _forward_fused(self, x: Tensor, timestamps: Tensor) -> EACSOutput:
        """Inference path via the fused gated kernel; matches eval sequential forward."""
        from ..ops.eacs_gated_scan import eacs_cell_forward
        x_in = x
        xn = self.norm(x)
        u, Bc, Cc = self._project(xn)
        ys, gates, resid = eacs_cell_forward(
            self.lam(), self.log_dt_scale, u, Bc, Cc, timestamps,
            self.D, self.obs_norm.running_mean, self.obs_norm.running_var,
            eps=float(self.gate.eps.item()), eta=self.gate.norm_eta,
            var_eps=self.obs_norm.eps, dt_init=self.dt_init, dt_max=self.dt_max)
        delta = self.proj_out(ys.to(x_in.dtype))
        y = x_in + delta if self.add_residual else delta
        return EACSOutput(y=y, gates=gates, update_rate=gates.mean(), residual=resid)

    def spectral_reg(self) -> Tensor:
        re = -torch.exp(self.a_log_neg_real)
        below = torch.relu(self.band_lo - re)
        above = torch.relu(re - self.band_hi)
        return (below + above).pow(2).mean()

    def _commit_range(self, lam, state, u, Bc, Cc, t, k0, k1, cpi=None):
        """Scan [k0,k1) recording per-frame commits; returns 8 tensors for checkpointing."""
        h_pi, t_pi, u_pi, B_pi, C_pi = state
        ys, gs, rs = [], [], []
        H, T, U, Bs, Cs = [], [], [], [], []
        p = self.step_params(lam)
        for k in range(k0, k1):
            _, y_k, g, r, h_pi, t_pi, u_pi, B_pi, C_pi = eacs_step(
                p, h_pi, t_pi, u_pi, B_pi, C_pi, t[:, k], u[:, k], Bc[:, k], Cc[:, k],
                cpi_k=(cpi[:, k] if cpi is not None else None))
            ys.append(y_k); gs.append(g); rs.append(r)
            H.append(h_pi); T.append(t_pi); U.append(u_pi); Bs.append(B_pi); Cs.append(C_pi)
        return (torch.stack(ys, 1), torch.stack(gs, 1), torch.stack(rs, 1),
                torch.stack(H, 1), torch.stack(T, 1), torch.stack(U, 1),
                torch.stack(Bs, 1), torch.stack(Cs, 1))

    def run_with_commits(self, x: Tensor, timestamps: Tensor, cpi: Tensor | None = None,
                         use_chunk: bool | None = None):
        """Forward recording per-frame commits for continuous queries.

        Returns (EACSOutput, commits). Gradient flow follows the caller context;
        callers inside an outer checkpoint must pass use_chunk=False.
        """
        B, L, _ = x.shape
        x_in = x
        xn = self.norm(x)
        u, Bc, Cc = self._project(xn)
        lam = self.lam().to(torch.complex64); t = timestamps.float()
        h_pi = torch.zeros(B, self.H, self.N, dtype=torch.complex64, device=x.device)
        state = (h_pi, t[:, 0] - self.dt_init, u[:, 0], Bc[:, 0], Cc[:, 0])

        cs = self.chunk_size
        chunked = (torch.is_grad_enabled() and cs and cs > 0 and L > cs
                   and use_chunk is not False)
        if chunked:
            bufs = [None] * 8
            for k0 in range(0, L, cs):
                k1 = min(k0 + cs, L)
                out = ckpt.checkpoint(self._commit_range, lam, state, u, Bc, Cc, t,
                                      k0, k1, cpi, use_reentrant=False)
                state = tuple(o[:, -1] for o in out[3:])
                bufs = [_write_chunk(bufs[i], out[i], L, k0, k1) for i in range(8)]
                del out
            ys, gs, rs, H, T, U, Bs, Cs = bufs
        else:
            ys, gs, rs, H, T, U, Bs, Cs = self._commit_range(
                lam, state, u, Bc, Cc, t, 0, L, cpi)

        delta = self.proj_out(ys.to(x_in.dtype))
        y = x_in + delta if self.add_residual else delta
        commits = dict(h=H, t=T, u=U, B=Bs, C=Cs, lam=lam,
                       frame_t=t, log_dt_scale=self.log_dt_scale, dt_max=self.dt_max)
        return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs), commits
