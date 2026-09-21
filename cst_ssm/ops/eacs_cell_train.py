"""Differentiable fused training path for the gated EACS cell.

Wraps an L-step gated scan in a single autograd.Function: forward runs grad-free
(forward graph is never built); backward recomputes states and propagates stepwise
VJPs through autograd. Gating uses a straight-through estimator.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .discretization import complex_expm1


def _step_functional(state, inputs, params, T: float, eta: float,
                     var_eps: float, mean: Tensor, var: Tensor,
                     gate_mode: str = "ste", dt_max: float = 1e3,
                     g_hard: Tensor | None = None,
                     inv_lam: Tensor | None = None):
    """Single segment step; state/inputs/params are tensor tuples.

    g_hard optionally pins the hard gate from the forward pass so recomputation
    follows the same trajectory. inv_lam is a precomputed 1/lambda valid only
    for grad-free forward recomputation, never for the VJP loop.
    """
    h_pi, t_pi, u_pi, B_pi, C_pi = state
    u_l, B_l, C_l, t_l = inputs
    lam, log_dt, D, eps = params

    delta = (t_l - t_pi).clamp_min(0.0)
    dt_eff = (delta.unsqueeze(-1) * torch.exp(log_dt)).clamp(1e-3, dt_max)
    z = lam * dt_eff.unsqueeze(-1)
    dA = torch.exp(z)
    if inv_lam is None:
        inv_lam = torch.reciprocal(lam)
    dBbar = complex_expm1(z, exp_z=dA) * inv_lam
    h_free = dA * h_pi
    xhat = h_free + dBbar * B_pi.unsqueeze(1) * u_pi.unsqueeze(-1)
    yhat = (C_pi.unsqueeze(1) * xhat).real.sum(-1)

    s = torch.sqrt(var + var_eps)
    diff = (u_l - yhat) / s
    obs = (u_l - mean) / s
    r = diff.norm(dim=-1) / (obs.norm(dim=-1) + eta)
    soft = torch.sigmoid((r - eps) / max(T, 1e-4))
    if gate_mode == "soft":
        g = soft
    else:
        hard = (r > eps).to(r.dtype) if g_hard is None else g_hard.to(r.dtype)
        g = hard + (soft - soft.detach())

    hupd = h_free + dBbar * B_l.unsqueeze(1) * u_l.unsqueeze(-1)
    gs = g.view(-1, 1, 1); gv = g.view(-1, 1)
    g_upd = gs * hupd
    one_g = 1 - gs
    hcur = g_upd + one_g * xhat
    h_pi_n = g_upd + one_g * h_pi
    t_pi_n = g * t_l + (1 - g) * t_pi
    u_pi_n = gv * u_l + (1 - gv) * u_pi
    B_pi_n = gv * B_l + (1 - gv) * B_pi
    C_pi_n = gv * C_l + (1 - gv) * C_pi
    y = (C_l.unsqueeze(1) * hcur).real.sum(-1) + D * u_l
    return (h_pi_n, t_pi_n, u_pi_n, B_pi_n, C_pi_n), y, g, r


def _init_state(u, Bc, Cc, t, dt_init):
    B, L, H = u.shape
    N = Bc.shape[-1]
    h_pi = torch.zeros(B, H, N, dtype=Bc.dtype, device=u.device)
    return (h_pi, t[:, 0] - dt_init, u[:, 0], Bc[:, 0], Cc[:, 0])


def _or_zeros(x, ref):
    """Zero-filled gradient matching ref when autograd returns None for unused inputs."""
    return torch.zeros_like(ref) if x is None else x


class GatedEACSCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                T, eta, var_eps, dt_init, gate_mode, use_kernel, dt_max):
        B, L, H = u.shape
        if use_kernel and u.is_cuda and gate_mode == "ste":
            try:
                from .eacs_gated_scan import gated_kernel_available, eacs_cell_forward
                if gated_kernel_available():
                    ys, gs, rs = eacs_cell_forward(
                        lam, log_dt, u, Bc, Cc, t, D, mean, var,
                        eps=float(eps), eta=eta, var_eps=var_eps, dt_init=dt_init,
                        dt_max=dt_max)
                    _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                              T, eta, var_eps, dt_init, gate_mode, L, dt_max, gs)
                    return ys, gs, rs
            except Exception:
                pass
        with torch.no_grad():
            state = _init_state(u, Bc, Cc, t, dt_init)
            ys, gs, rs = [], [], []
            inv_lam = torch.reciprocal(lam)
            for l in range(L):
                state, y, g, r = _step_functional(
                    state, (u[:, l], Bc[:, l], Cc[:, l], t[:, l]),
                    (lam, log_dt, D, eps), T, eta, var_eps, mean, var, gate_mode,
                    dt_max=dt_max, inv_lam=inv_lam)
                ys.append(y); gs.append(g); rs.append(r)
            ys = torch.stack(ys, 1); gs = torch.stack(gs, 1); rs = torch.stack(rs, 1)
        _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                  T, eta, var_eps, dt_init, gate_mode, L, dt_max, gs)
        return ys, gs, rs

    @staticmethod
    def backward(ctx, g_ys, g_gs, g_rs):
        u, Bc, Cc, t, lam, log_dt, D, eps, mean, var, gs_fwd = ctx.saved_tensors
        T, eta, var_eps, dt_init, gate_mode, L = ctx.T, ctx.eta, ctx.var_eps, ctx.dt_init, ctx.gate_mode, ctx.L
        dt_max = ctx.dt_max
        B, _, H = u.shape

        def _g(l):
            return None if gate_mode == "soft" else gs_fwd[:, l]

        with torch.no_grad():
            state = _init_state(u, Bc, Cc, t, dt_init)
            prev = []
            inv_lam = torch.reciprocal(lam)
            for l in range(L):
                prev.append(state)
                state, _, _, _ = _step_functional(
                    state, (u[:, l], Bc[:, l], Cc[:, l], t[:, l]),
                    (lam, log_dt, D, eps), T, eta, var_eps, mean, var, gate_mode,
                    dt_max=dt_max, g_hard=_g(l), inv_lam=inv_lam)

        du = torch.zeros_like(u); dBc = torch.zeros_like(Bc); dCc = torch.zeros_like(Cc); dt = torch.zeros_like(t)
        dlam = torch.zeros_like(lam); dlog = torch.zeros_like(log_dt); dD = torch.zeros_like(D); deps = torch.zeros_like(eps)
        ah = torch.zeros_like(prev[0][0]); at = torch.zeros_like(prev[0][1])
        au = torch.zeros_like(prev[0][2]); aB = torch.zeros_like(prev[0][3]); aC = torch.zeros_like(prev[0][4])

        for l in reversed(range(L)):
            ph = prev[l][0].detach().requires_grad_(True)
            pt = prev[l][1].detach().requires_grad_(True)
            pu = prev[l][2].detach().requires_grad_(True)
            pB = prev[l][3].detach().requires_grad_(True)
            pC = prev[l][4].detach().requires_grad_(True)
            ul = u[:, l].detach().requires_grad_(True); Bl = Bc[:, l].detach().requires_grad_(True)
            Cl = Cc[:, l].detach().requires_grad_(True); tl = t[:, l].detach().requires_grad_(True)
            lam_ = lam.detach().requires_grad_(True); log_ = log_dt.detach().requires_grad_(True)
            D_ = D.detach().requires_grad_(True); eps_ = eps.detach().requires_grad_(True)
            with torch.enable_grad():
                (nh, nt, nu, nB, nC), y, g, r = _step_functional(
                    (ph, pt, pu, pB, pC), (ul, Bl, Cl, tl), (lam_, log_, D_, eps_),
                    T, eta, var_eps, mean, var, gate_mode,
                    dt_max=dt_max, g_hard=_g(l))
            outs = [nh, nt, nu, nB, nC, y, g, r]
            gouts = [ah, at, au, aB, aC, g_ys[:, l],
                     g_gs[:, l] if g_gs is not None else torch.zeros_like(g),
                     g_rs[:, l] if g_rs is not None else torch.zeros_like(r)]
            grads = torch.autograd.grad(outs, [ph, pt, pu, pB, pC, ul, Bl, Cl, tl, lam_, log_, D_, eps_],
                                        gouts, allow_unused=True, retain_graph=False)
            ah = _or_zeros(grads[0], ah); at = _or_zeros(grads[1], at)
            au = _or_zeros(grads[2], au); aB = _or_zeros(grads[3], aB); aC = _or_zeros(grads[4], aC)
            du[:, l] += _or_zeros(grads[5], du[:, l]); dBc[:, l] += _or_zeros(grads[6], dBc[:, l])
            dCc[:, l] += _or_zeros(grads[7], dCc[:, l]); dt[:, l] += _or_zeros(grads[8], dt[:, l])
            dlam += _or_zeros(grads[9], dlam); dlog += _or_zeros(grads[10], dlog)
            dD += _or_zeros(grads[11], dD); deps += _or_zeros(grads[12], deps)

        dt[:, 0] += at; du[:, 0] += au; dBc[:, 0] += aB; dCc[:, 0] += aC
        return (du, dBc, dCc, dt, dlam, dlog, dD, deps, None, None,
                None, None, None, None, None, None, None)


def _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
              T, eta, var_eps, dt_init, gate_mode, L, dt_max, gs):
    ctx.save_for_backward(u, Bc, Cc, t, lam, log_dt, D, eps, mean, var, gs)
    ctx.T, ctx.eta, ctx.var_eps, ctx.dt_init, ctx.gate_mode, ctx.L = T, eta, var_eps, dt_init, gate_mode, L
    ctx.dt_max = dt_max


def fused_gated_scan(u: Tensor, Bc: Tensor, Cc: Tensor, t: Tensor,
                     lam: Tensor, log_dt: Tensor, D: Tensor, eps: Tensor,
                     mean: Tensor, var: Tensor, T: float, eta: float,
                     var_eps: float = 1e-5, dt_init: float = 1.0,
                     gate_mode: str = "ste", use_kernel: bool = True,
                     dt_max: float = 1e3):
    """Differentiable fused gated scan; returns (ys[B,L,H], gates[B,L], resid[B,L])."""
    return GatedEACSCellFunction.apply(u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                                       T, eta, var_eps, dt_init, gate_mode, use_kernel,
                                       dt_max)
