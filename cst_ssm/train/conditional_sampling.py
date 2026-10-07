"""InfoNCE with caller-supplied, per-context candidate sets.

This module validates tensor contracts, not sampling distributions. A conditional
MI interpretation additionally requires a positive from p(Y|Z,C) and negatives
independently sampled from p(Y|C), independent of Z given C. Merely matching a
context embedding or retrieving nearby histories does NOT establish this.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def conditional_candidate_infonce(
    z: Tensor,
    candidates: Tensor,
    context: Tensor,
    critic,
    temperature: float = 0.1,
    positive_index: Tensor | None = None,
    valid: Tensor | None = None,
) -> Tensor:
    """Return contrastive loss; do not certify a conditional MI bound.

    z: (N,d), candidates: (N,K,d_y), context: (N,d_c). Each row has its
    OWN candidates; row zero's candidates are never used as row one's negatives.
    The positive is at index zero unless positive_index (N,) is supplied.
    critic(z_row, candidate_rows, context_row) must return (1,K), matching the
    existing ConditionalCritic interface. Candidate targets are detached.

    Under the stated distributional assumptions (which the caller must justify),
    log(K) - E[loss] is a conditional InfoNCE lower bound. A finite sampled loss
    need not give a pointwise valid bound, and this function returns only loss.
    """
    if z.ndim != 2 or context.ndim != 2 or candidates.ndim != 3:
        raise ValueError("expected z/context matrices and candidates (N,K,d_y)")
    n = z.shape[0]
    if candidates.shape[0] != n or context.shape[0] != n:
        raise ValueError("candidate and context batch sizes must match z")
    k = candidates.shape[1]
    if k < 2:
        raise ValueError("each context requires at least one positive and one negative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    labels = (torch.zeros(n, dtype=torch.long, device=z.device)
              if positive_index is None else positive_index)
    if labels.shape != (n,) or labels.dtype != torch.long:
        raise ValueError("positive_index must be a long tensor of shape (N,)")
    if bool(((labels < 0) | (labels >= k)).any()):
        raise ValueError("positive_index is outside the candidate set")
    if valid is not None:
        if valid.shape != (n,) or valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor of shape (N,)")
        z, candidates, context, labels = z[valid], candidates[valid], context[valid], labels[valid]
    if z.shape[0] == 0:
        return z.sum() * 0.0
    logits = []
    for i in range(z.shape[0]):
        row = critic(z[i:i + 1], candidates[i].detach(), context[i:i + 1])
        if row.shape != (1, k):
            raise ValueError("critic must return (1,K) for each per-context candidate set")
        logits.append(row.squeeze(0))
    return F.cross_entropy(torch.stack(logits) / temperature, labels)
