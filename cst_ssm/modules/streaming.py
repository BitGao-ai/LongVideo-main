"""Explicit causal EACS state. This path never estimates statistics from future frames."""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor

from .eacs import eacs_step
from .multiscale import MultiScaleOutput


@dataclass
class BranchState:
    h: Tensor
    t: Tensor
    u: Tensor
    B: Tensor
    C: Tensor
    initialized: Tensor

    def detach(self):
        return BranchState(*(x.detach() for x in self.__dict__.values()))


def initial_states(temporal, batch: int, device) -> list[BranchState]:
    states = []
    for br in temporal.branches:
        states.append(BranchState(
            torch.zeros(batch, br.H, br.N, dtype=torch.complex64, device=device),
            torch.zeros(batch, device=device), torch.zeros(batch, br.H, device=device),
            torch.zeros(batch, br.N, dtype=torch.complex64, device=device),
            torch.zeros(batch, br.N, dtype=torch.complex64, device=device),
            torch.zeros(batch, dtype=torch.bool, device=device)))
    return states


def _choose(mask, new, old):
    return torch.where(mask.reshape(mask.shape + (1,) * (new.ndim - mask.ndim)), new, old)


def temporal_step(temporal, x: Tensor, t: Tensor, states: list[BranchState],
                  valid: Tensor, cpi: Tensor | None = None):
    """One physical-time observation; masked rows leave *all* state unchanged.

    Observation normalization uses fixed calibrated buffers, not batch statistics.
    This makes evaluation and training decisions prefix-causal. Calibration, if
    desired, must be performed on training data separately, before this scan.
    """
    ys, gs, rs, ns, new = [], [], [], [], []
    for br, state in zip(temporal.branches, states):
        if br.robust_guard:
            raise ValueError("streaming CPIB does not support robust_guard; disable it explicitly")
        u, B, C = br._project(br.norm(x))
        first = valid & ~state.initialized
        if bool((valid & state.initialized & (t < state.t)).any()):
            raise ValueError("timestamps must not precede the last committed observation")
        t0 = _choose(first, t.float() - br.dt_init, state.t)
        u0, B0, C0 = (_choose(first, z, old) for z, old in
                      ((u, state.u), (B, state.B), (C, state.C)))
        p = br.step_params(br.lam().to(torch.complex64))
        # Buffers are deliberately immutable within a sequence, including replay.
        p = replace(p, obs_mean=p.obs_mean.clone(), obs_var=p.obs_var.clone())
        first_eps = torch.where(first, -torch.ones_like(t.float()), br.gate.eps)
        result = eacs_step(p, state.h, t0, u0, B0, C0,
                           t.float(), u, B, C, cpi_k=cpi, eps_override=first_eps,
                           skip_compute=not br.training)
        hcur, y, g, r, n, hn, tn, un, Bn, Cn = result
        # The first valid observation is a real commit, even for an always-skip gate.
        hn = _choose(first, hcur, hn)
        tn, un, Bn, Cn = (_choose(first, z, old) for z, old in
                          ((t.float(), tn), (u, un), (B, Bn), (C, Cn)))
        g = torch.where(first, torch.ones_like(g), g) * valid
        new.append(BranchState(*(_choose(valid, z, old) for z, old in
                                  zip((hn, tn, un, Bn, Cn),
                                      (state.h, state.t, state.u, state.B, state.C))),
                               state.initialized | valid))
        delta = br.proj_out(y.to(x.dtype))
        ys.append(delta * valid[:, None])
        gs.append(g); rs.append(r * valid)
        ns.append(n * valid * state.initialized)
    weights = temporal.fusion(x).softmax(-1)
    fused = (torch.stack(ys, -1) * weights[:, None, :]).sum(-1)
    y = temporal.out_norm(x + fused) * valid[:, None]
    return y, torch.stack(gs, 1), torch.stack(rs, 1), torch.stack(ns, 1), weights, new


def sequence_output(ys, gs, rs, ns, weights, valid):
    gates = torch.stack(gs, -1)
    denom = valid.sum().clamp_min(1)
    per_branch = gates.sum((0, 2)) / denom
    return MultiScaleOutput(torch.stack(ys, 1), gates, per_branch.mean(), per_branch,
                            torch.stack(weights, 1), torch.stack(rs, -1), torch.stack(ns, -1))
