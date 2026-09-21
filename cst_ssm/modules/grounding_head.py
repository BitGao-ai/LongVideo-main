"""Query-conditioned temporal scoring head for sub-frame localization."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    """Mean of (B,T,d) over valid positions; plain mean when mask is None."""
    if mask is None:
        return x.mean(1)
    m = mask.to(x.dtype).unsqueeze(-1)
    denom = m.sum(1).clamp_min(1.0)
    return (x * m).sum(1) / denom


class GroundingHead(nn.Module):
    """Scores query relevance of visual readouts.

    encode_query: text embeddings (B,T,d_txt) -> query vector (B,d_model).
    forward: query (B,d_model) + readouts (B,Q,d_model) -> logits (B,Q).
    """

    def __init__(self, d_model: int, d_txt: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model
        self.q_proj = nn.Sequential(
            nn.Linear(d_txt, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.score = nn.Sequential(
            nn.Linear(3 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def encode_query(self, text_emb: Tensor, text_mask: Tensor | None = None) -> Tensor:
        return self.q_proj(masked_mean(text_emb, text_mask))

    def forward(self, query_vec: Tensor, readouts: Tensor) -> Tensor:
        Q = readouts.shape[1]
        q = query_vec.unsqueeze(1).expand(-1, Q, -1)
        feat = torch.cat([readouts, q, readouts * q], dim=-1)
        return self.score(feat).squeeze(-1)


def span_targets(t_query: Tensor, gt_start: Tensor, gt_end: Tensor) -> Tensor:
    """Binary (B,Q) targets: 1 where the query time falls inside [gt_start, gt_end]."""
    return ((t_query >= gt_start.unsqueeze(1)) & (t_query <= gt_end.unsqueeze(1))).float()


def grounding_loss(logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """BCE-with-logits over query times, optionally restricted to valid queries."""
    if mask is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    m = mask.to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1.0)
