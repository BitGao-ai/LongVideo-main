"""Continuous query of an explicit adaptive-stream commit index (O(events) storage)."""
from __future__ import annotations

import torch
from ..ops.discretization import zoh_discretize, effective_dt


def query_history(model, history, t_query):
    if model.cfg.eacs_disc_mode != 'continuous':
        raise ValueError('physical-time query requires continuous dynamics; use a discrete offset head otherwise')
    if not torch.isfinite(t_query).all():
        raise ValueError('query times must be finite')
    rows = []
    for b in range(t_query.shape[0]):
        outputs, weights = [], []
        for br, events in zip(model.temporal.branches, history):
            selected = [(s, w, ts) for s, fire, w, ts in events if bool(fire[b])]
            if not selected:
                outputs.append(t_query.new_zeros(t_query.shape[1], model.cfg.d_model))
                weights.append(t_query.new_zeros(t_query.shape[1]))
                continue
            times = torch.stack([ts[b] for _, _, ts in selected])
            index = torch.searchsorted(times.contiguous(), t_query[b].contiguous(), right=True) - 1
            valid = index >= 0
            idx = index.clamp_min(0)
            states = [s for s, _, _ in selected]
            h = torch.stack([s.h[b] for s in states])[idx]
            t = torch.stack([s.t[b] for s in states])[idx]
            u = torch.stack([s.u[b] for s in states])[idx]
            B = torch.stack([s.B[b] for s in states])[idx]
            C = torch.stack([s.C[b] for s in states])[idx]
            delta = (t_query[b] - t).clamp_min(0)
            dt = effective_dt(delta[:, None], br.log_dt_scale, dt_max=br.dt_max)
            dt = torch.where(delta[:, None] == 0, torch.zeros_like(dt), dt)
            lam = br.lam().to(torch.complex64)
            dA, dB = zoh_discretize(lam, dt, B, inv_lam=lam.reciprocal())
            evolved = dA * h + dB * u.unsqueeze(-1)
            y = (C.unsqueeze(1) * evolved).sum(-1).real + br.D * u
            outputs.append(br.proj_out(y.to(br.proj_out.weight.dtype)) * valid[:, None])
            weights.append(torch.stack([w[b] for _, w, _ in selected])[idx] * valid)
        w = torch.stack(weights, -1)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
        rows.append((torch.stack(outputs, -1) * w[:, None, :]).sum(-1))
    return torch.stack(rows)
