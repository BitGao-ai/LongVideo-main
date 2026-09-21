"""CPIB-Distill losses: conditional InfoNCE, bottleneck KL, counterfactual consistency."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BilinearCritic(nn.Module):
    """Bilinear scorer f(z, z') = z^T W z'."""

    def __init__(self, dim: int, symmetric: bool = False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * (dim ** -0.5))
        self.symmetric = symmetric

    @property
    def effective_W(self) -> Tensor:
        return self.W + self.W.T if self.symmetric else self.W

    def forward(self, z: Tensor, zp: Tensor) -> Tensor:
        """z (..., d), zp (..., d) -> scores (...)."""
        return torch.einsum("...d,de,...e->...", z, self.effective_W, zp)


def conditional_infonce_loss(z: Tensor, z_future: Tensor, critic: BilinearCritic,
                             temperature: float = 0.1,
                             frame_mask: Tensor | None = None) -> Tensor:
    """Conditional InfoNCE: z_t predicts the pre-shifted future frame z_future[t].

    Args:
        z: (B, L, d) current frame features.
        z_future: (B, L, d) aligned future features (caller pre-shifts).
        frame_mask: (B, L) validity mask.
    """
    B, L, d = z.shape
    if L < 2:
        return z.sum() * 0.0

    z_t = z[:, :-1].reshape(-1, d)
    z_pos = z_future[:, :-1].reshape(-1, d).detach()

    if frame_mask is not None:
        valid = (frame_mask[:, :-1] & frame_mask[:, 1:]).reshape(-1)
        if not valid.any():
            return z.sum() * 0.0
        z_t = z_t[valid]
        z_pos = z_pos[valid]

    N = z_t.shape[0]
    if N < 2:
        return z.sum() * 0.0

    logits = (z_t @ critic.effective_W) @ z_pos.t() / temperature
    labels = torch.arange(N, device=z.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def information_bottleneck_kl(scores: Tensor, rho: float = 0.3,
                              frame_mask: Tensor | None = None) -> Tensor:
    """Rate-budget regularizer: mean KL(Bern(s) || Bern(rho)) over tokens.

    Args:
        scores: (B, L, P) CGU keep scores.
        rho: target keep rate.
    """
    eps = 1e-7
    s = scores.clamp(eps, 1 - eps)
    kl = s * torch.log(s / rho) + (1 - s) * torch.log((1 - s) / (1 - rho))

    if frame_mask is not None:
        mask = frame_mask.unsqueeze(-1).expand_as(kl)
        kl = kl * mask
        n = mask.sum().clamp_min(1)
    else:
        n = kl.numel()
    return kl.sum() / n


def counterfactual_consistency_loss(
    scores: Tensor,
    frame_feat: Tensor,
    tokens: Tensor,
    distiller_forward,
    K: int = 2,
    frame_mask: Tensor | None = None,
    ablate_frac: float = 0.1,
) -> Tensor:
    """Counterfactual consistency: ablated-token scores track per-frame marginal effects.

    Args:
        scores: (B, L, P) CGU scores.
        frame_feat: (B, L, d) normal distillation output.
        tokens: (B, L, P, d) source tokens.
        distiller_forward: callable(tokens, scores) -> (frame_feat, cpi).
        K: Monte-Carlo samples.
    """
    B, L, P, d = tokens.shape
    if P < 2:
        return scores.sum() * 0.0
    n_ab = max(1, min(P - 1, int(round(ablate_frac * P))))

    total_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    for _ in range(K):
        ablate_idx = torch.rand(B, L, P, device=scores.device).argsort(dim=-1)[..., :n_ab]
        scores_cf = scores.clone()
        scores_cf.scatter_(2, ablate_idx, 0.0)

        with torch.no_grad():
            feat_cf, _ = distiller_forward(tokens, scores_cf)

        delta = torch.linalg.vector_norm(
            frame_feat.detach() - feat_cf, dim=-1)

        s_ablated = scores.gather(2, ablate_idx).mean(dim=-1)

        denom = torch.linalg.vector_norm(frame_feat.detach(), dim=-1)
        delta_norm = (delta / denom.clamp_min(1e-8)).clamp(0.0, 1.0)

        diff = (s_ablated - delta_norm.detach()) ** 2
        if frame_mask is not None:
            diff = diff * frame_mask
            n = frame_mask.sum().clamp_min(1)
        else:
            n = diff.numel()
        total_loss = total_loss + diff.sum() / n

    return total_loss / K


@dataclass
class CPIBWeights:
    """CPIB loss term weights."""
    nce: float = 0.1
    kl: float = 0.01
    cf: float = 0.05
    cf_ablate_frac: float = 0.1
    rho: float = 0.3
    cf_anneal_steps: int = 2000


def cpib_loss(
    scores: Tensor,
    frame_feat: Tensor,
    tokens: Tensor,
    z_future: Tensor,
    distiller_forward,
    w: CPIBWeights,
    step: int = 0,
    frame_mask: Tensor | None = None,
    critic: BilinearCritic | None = None,
) -> tuple[Tensor, dict]:
    """Full CPIB auxiliary loss; returns (total, components)."""
    comp = {}
    total = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    if critic is not None and w.nce > 0:
        l_nce = conditional_infonce_loss(frame_feat, z_future, critic,
                                         frame_mask=frame_mask)
        total = total + w.nce * l_nce
        comp["nce"] = l_nce.detach()

    if w.kl > 0:
        r_kl = information_bottleneck_kl(scores, rho=w.rho, frame_mask=frame_mask)
        total = total + w.kl * r_kl
        comp["ib_kl"] = r_kl.detach()

    cf_weight = w.cf * max(0.0, 1.0 - step / max(w.cf_anneal_steps, 1))
    if cf_weight > 1e-8:
        l_cf = counterfactual_consistency_loss(
            scores, frame_feat, tokens, distiller_forward,
            K=2, frame_mask=frame_mask, ablate_frac=w.cf_ablate_frac)
        total = total + cf_weight * l_cf
        comp["cf"] = l_cf.detach()

    comp["cpib_total"] = total.detach()
    return total, comp
