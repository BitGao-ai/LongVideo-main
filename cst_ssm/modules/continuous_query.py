"""Continuous-time queries at arbitrary timestamps between frames."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..ops.discretization import zoh_discretize, effective_dt
from .eacs import EACSLayer


def segment_index(frame_t: Tensor, t_query: Tensor) -> Tensor:
    """Segment index of each query time: last k with frame_t[k] <= t*. Returns (B,Q)."""
    L = frame_t.shape[1]
    idx = torch.searchsorted(frame_t.detach().contiguous(),
                             t_query.detach().contiguous(), right=True) - 1
    return idx.clamp(0, L - 1)


def readout_from_commits(layer: EACSLayer, commits: dict, t_query: Tensor,
                         idx: Tensor | None = None) -> Tensor:
    """Evolve per-frame commits to query times; returns (B,Q,d).

    Args:
        idx: optional precomputed segment indices shared across branches.
    """
    lam = commits["lam"]
    frame_t = commits["frame_t"].contiguous()
    B, L = frame_t.shape
    Q = t_query.shape[1]

    if t_query is frame_t or t_query is commits["frame_t"]:
        h_pi, t_pi = commits["h"], commits["t"]
        u_pi, B_pi, C_pi = commits["u"], commits["B"], commits["C"]
    else:
        if idx is None:
            idx = segment_index(frame_t, t_query)

        def gather(seq: Tensor) -> Tensor:
            extra = seq.shape[2:]
            index = idx.view(B, Q, *([1] * len(extra))).expand(B, Q, *extra)
            return torch.gather(seq, 1, index)

        h_pi = gather(commits["h"])
        t_pi = gather(commits["t"])
        u_pi = gather(commits["u"])
        B_pi = gather(commits["B"])
        C_pi = gather(commits["C"])

    delta = (t_query - t_pi).clamp_min(0.0)
    dt_eff = effective_dt(delta.unsqueeze(-1), commits["log_dt_scale"],
                          dt_max=commits.get("dt_max", 1e3))
    dA, dB = zoh_discretize(lam, dt_eff, B_pi, inv_lam=torch.reciprocal(lam))
    x_star = dA * h_pi + dB * u_pi.unsqueeze(-1)
    y = torch.einsum("bqn,bqhn->bqh", C_pi, x_star).real
    y = y + layer.D * u_pi
    return layer.proj_out(y)


def continuous_query(layer: EACSLayer, x: Tensor, timestamps: Tensor,
                     t_query: Tensor, cpi: Tensor | None = None,
                     use_chunk: bool | None = None) -> Tensor:
    """Query readouts at t_query: x (B,L,d), timestamps (B,L), t_query (B,Q) -> (B,Q,d)."""
    _, commits = layer.run_with_commits(x, timestamps, cpi=cpi, use_chunk=use_chunk)
    return readout_from_commits(layer, commits, t_query)


class ContinuousQuery(nn.Module):
    """Thin wrapper binding one EACSLayer to the module-level query functions."""

    def __init__(self, layer: EACSLayer):
        super().__init__()
        self.layer = layer

    def query(self, x: Tensor, timestamps: Tensor, t_query: Tensor,
              cpi: Tensor | None = None) -> Tensor:
        """x (B,L,d), timestamps (B,L), t_query (B,Q) -> readouts (B,Q,d)."""
        return continuous_query(self.layer, x, timestamps, t_query, cpi=cpi)

    def readout_at(self, commits: dict, t_query: Tensor) -> Tensor:
        return readout_from_commits(self.layer, commits, t_query)
