"""Empirical sparse/dense trajectory consistency, not a certified error bound."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.func import functional_call

from .eacs import EACSLayer
from .multiscale import MultiScaleEACS


@dataclass
class DenseConsistencyOutput:
    loss: Tensor
    mse: Tensor
    per_sample_mse: Tensor
    sparse_y: Tensor
    dense_y: Tensor


def _replica(temporal: nn.Module, dense: bool) -> nn.Module:
    # Copy module settings/buffers, not parameter storage. Functional calls below
    # supply either live student parameters or detached teacher parameters.
    model = copy.deepcopy(temporal, {id(p): p for p in temporal.parameters()})
    branches = model.branches if isinstance(model, MultiScaleEACS) else [model]
    for branch in branches:
        branch.obs_norm.eval()
        if dense:
            branch.gate.gate_kind = "always"
            branch.robust_guard = False
            branch.fused_train = False
    return model


def dense_trajectory_consistency(
    temporal: EACSLayer | MultiScaleEACS,
    features: Tensor,
    timestamps: Tensor,
    frame_mask: Tensor | None = None,
    cpi: Tensor | None = None,
    *,
    sparse_output=None,
) -> DenseConsistencyOutput:
    """Compare actual sparse outputs to a detached always-update teacher.

    Both trajectories use the supplied temporal weights and a snapshot of its
    running statistics. With ``sparse_output`` (Tensor or output object with y),
    reuse the actual student's forward graph; otherwise recompute the student's
    configured forward with frozen observation statistics. Neither execution
    changes the original module's mode, gate settings, parameters or buffers.

    This diagnostic supports one EACSLayer or MultiScaleEACS. Legacy EACS only
    masks accounting, not recurrence: masks must therefore be right-padded valid
    prefixes (empty samples are allowed). Internal holes/left padding are rejected
    rather than falsely claiming that masked frames cannot affect the trajectory.

    ``loss`` is differentiable through the sparse side only. All valid output
    coordinates contribute to its MSE; empty batches return differentiable zero.
    This is an optional empirical regularizer, not a hard skip-error guarantee.
    Two extra scans and lightweight module replicas can be expensive; pass the
    existing sparse output to avoid the student recomputation.
    """
    if not isinstance(temporal, (EACSLayer, MultiScaleEACS)):
        raise TypeError("temporal must be EACSLayer or MultiScaleEACS")
    if features.ndim != 3 or timestamps.shape != features.shape[:2]:
        raise ValueError("expected features (B,L,d) and timestamps (B,L)")
    B, L, d = features.shape
    if L == 0 or d == 0:
        raise ValueError("sequence length and feature width must be positive")
    valid = (torch.ones(B, L, dtype=torch.bool, device=features.device)
             if frame_mask is None else frame_mask.bool())
    if valid.shape != (B, L):
        raise ValueError("frame_mask must match (B,L)")
    if cpi is not None and cpi.shape != (B, L):
        raise ValueError("cpi must match (B,L)")
    if bool((valid[:, 1:] & ~valid[:, :-1]).any()):
        raise ValueError("legacy trajectory diagnostic requires right-padded valid prefixes")
    if not bool(torch.isfinite(timestamps[valid]).all()):
        raise ValueError("valid timestamps must be finite")
    pairs = valid[:, 1:] & valid[:, :-1]
    if bool(((timestamps[:, 1:] < timestamps[:, :-1]) & pairs).any()):
        raise ValueError("valid timestamps must be nondecreasing")

    def run(dense):
        model = _replica(temporal, dense)
        params = {k: (v.detach() if dense else v) for k, v in temporal.named_parameters()}
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        rows = []
        for b in range(B):
            n = int(valid[b].sum())
            if not n:
                rows.append(features.new_zeros(1, L, d))
                continue
            x = features[b:b + 1, :n]
            t = timestamps[b:b + 1, :n]
            prior = None if cpi is None else cpi[b:b + 1, :n]
            if dense:
                x, t = x.detach(), t.detach()
                prior = None if prior is None else prior.detach()
            out = functional_call(model, (params, buffers), (x, t), {"cpi": prior})
            rows.append(torch.cat([out.y, out.y.new_zeros(1, L - n, d)], dim=1))
        return torch.cat(rows, dim=0) if rows else features.new_zeros(B, L, d)

    if sparse_output is None:
        sparse_y = run(False)
    else:
        sparse_y = sparse_output if isinstance(sparse_output, Tensor) else sparse_output.y
        if sparse_y.shape != features.shape:
            raise ValueError("sparse output must match features (B,L,d)")
    with torch.no_grad():
        dense_y = run(True)
    # Use where rather than multiplication: NaNs in padding are irrelevant.
    error = torch.where(valid[..., None], sparse_y - dense_y, 0.0).float().square()
    per_sample = error.sum((1, 2)) / (valid.sum(1) * d).clamp_min(1)
    loss = error.sum() / (valid.sum() * d).clamp_min(1)
    if not bool(valid.any()):
        # Preserve a safe zero graph even when the caller's input needs no grad.
        loss = loss + next(temporal.parameters()).sum() * 0.0
    return DenseConsistencyOutput(loss, loss.detach(), per_sample.detach(), sparse_y, dense_y)
