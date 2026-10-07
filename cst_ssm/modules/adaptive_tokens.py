"""Packed, history-conditioned spatial pruning with a stochastic payload bottleneck.

The reference implementation groups retained coordinates before attention. It is
intentionally a correctness path, not a fused ragged-attention performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cgu import CGU
from ..dtypes import align_to_param


@dataclass
class AdaptiveTokenOutput:
    tokens: Tensor
    mask: Tensor
    scores: Tensor
    frame: Tensor
    cpi: Tensor
    positions: Tensor
    rate_kl: Tensor
    bottleneck_kl: Tensor
    selection_bits: Tensor
    layer_counts: list[Tensor]
    raw_target: Tensor
    attention_sizes: list[list[int]]


class PackedWindowBlock(nn.Module):
    """Attention over actual retained tokens, grouped by shifted grid windows."""

    def __init__(self, d: int, heads: int, window: int, shift: int):
        super().__init__()
        self.window, self.shift = window, shift
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: Tensor, positions: Tensor, grid_width: int):
        if x.shape[0] == 0:
            return x, []
        row, col = positions // grid_width, positions % grid_width
        # Non-cyclic shifted windows avoid falsely connecting opposite borders.
        groups = torch.stack(((row + self.shift) // self.window,
                              (col + self.shift) // self.window), dim=-1)
        _, membership = torch.unique(groups, dim=0, return_inverse=True)
        out = torch.zeros_like(x)
        sizes = []
        for group in membership.unique():
            idx = (membership == group).nonzero(as_tuple=True)[0]
            local = self.norm1(x[idx]).unsqueeze(0)
            att, _ = self.attn(local, local, local, need_weights=False)
            out = out.index_copy(0, idx, att.squeeze(0))
            sizes.append(idx.numel())
        x = x + out
        return x + self.mlp(self.norm2(x)), sizes


class AdaptiveTokenEncoder(nn.Module):
    """One frame in, variable key tokens plus at most one summary out.

    ``ablate_mask`` is True for original tokens to remove *before* any attention,
    CGU, or summary. ``deterministic`` uses posterior means (useful for paired
    interventions); the information-rate bound applies to the stochastic training
    payload, not to deterministic inference nor to external score side channels.
    ``selection_bits`` is a conservative metadata code length in **nats** (legacy
    field name): P log(2) per selection layer, plus the output length alphabet.
    """

    def __init__(self, d: int, hidden: int = 256, min_tokens: int = 1,
                 max_tokens: int = 16, threshold: float = 0.5,
                 vision_depth: int = 4, heads: int = 8, window: int = 7,
                 patch: int = 16, input_mode: str = "feature", feat_dim: int = 768):
        super().__init__()
        if not 1 <= min_tokens <= max_tokens:
            raise ValueError("require 1 <= min_tokens <= max_tokens")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be in (0, 1)")
        if input_mode not in {"pixel", "feature"}:
            raise ValueError("input_mode must be pixel or feature")
        if vision_depth < 1 or window < 1 or patch < 1 or (input_mode == "pixel" and d % heads):
            raise ValueError("invalid depth/window/patch or non-divisible attention heads")
        self.d, self.input_mode = d, input_mode
        self.min_tokens, self.max_tokens, self.threshold = min_tokens, max_tokens, threshold
        self.patch = patch
        self.proj = (nn.Conv2d(3, d, patch, stride=patch) if input_mode == "pixel"
                     else nn.Linear(feat_dim, d))
        self.norm = nn.LayerNorm(d)
        depth = vision_depth if input_mode == "pixel" else 1
        self.blocks = nn.ModuleList([
            PackedWindowBlock(d, heads, window, 0 if i % 2 == 0 else window // 2)
            for i in range(depth)
        ]) if input_mode == "pixel" else nn.ModuleList()
        self.cgus = nn.ModuleList([CGU(d, d, hidden) for _ in range(depth)])
        for cgu in self.cgus:
            # Legacy CGU starts at constant 0.5; nonzero weights break budget ties.
            nn.init.normal_(cgu.mlp[-1].weight, std=0.02)
        self.mu = nn.Linear(d, d)
        self.logvar = nn.Linear(d, d)
        nn.init.eye_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -2.0)

    def _input_tokens(self, x: Tensor):
        x = align_to_param(x, self.proj.weight)
        if self.input_mode == "pixel":
            if x.ndim != 4 or x.shape[1] != 3:
                raise ValueError("pixel frames must have shape (B,3,H,W)")
            if x.shape[-2] < self.patch or x.shape[-1] < self.patch:
                raise ValueError("frame is smaller than one patch")
            # Include border pixels instead of silently dropping partial patches.
            ph, pw = (-x.shape[-2]) % self.patch, (-x.shape[-1]) % self.patch
            z = self.proj(F.pad(x, (0, pw, 0, ph)))
            width = z.shape[-1]
            z = z.flatten(2).transpose(1, 2)
        else:
            if x.ndim != 3:
                raise ValueError("feature frames must have shape (B,P,feat_dim)")
            z = self.proj(x)
            width = z.shape[1]
        return self.norm(z), width

    def forward_frame(self, x: Tensor, context: Tensor, valid: Tensor,
                      ablate_mask: Tensor | None = None, deterministic: bool = False,
                      noise: Tensor | None = None, rho: float = 0.3,
                      collect_loss: bool = True) -> AdaptiveTokenOutput:
        if not 0 < rho < 1:
            raise ValueError("rho must be strictly between zero and one")
        row_mask = valid.bool().reshape(valid.shape[0], *([1] * (x.ndim - 1)))
        x = torch.where(row_mask, x, 0)
        if self.input_mode == "feature" and ablate_mask is not None:
            x = torch.where(ablate_mask.bool().unsqueeze(-1), 0, x)
        z, width = self._input_tokens(x)
        batch, original_p, d = z.shape
        if original_p == 0:
            raise ValueError("at least one original spatial token is required")
        if context.shape != (batch, d) or valid.shape != (batch,):
            raise ValueError("context or frame validity shape mismatch")
        context = context.to(z.dtype)
        available = valid.bool().unsqueeze(1).expand(batch, original_p)
        if ablate_mask is not None:
            if ablate_mask.shape != (batch, original_p):
                raise ValueError("ablate_mask must index original spatial tokens")
            available = available & ~ablate_mask.bool()
        target = (torch.where(available.unsqueeze(-1), z, 0).sum(1)
                  / available.sum(1).clamp_min(1).unsqueeze(-1)).detach()
        packed, positions, score_rows, rates, counts_rows, att_rows = [], [], [], [], [], []
        for b in range(batch):
            pos = available[b].nonzero(as_tuple=True)[0]
            current = z[b, pos]
            original_scores = z.new_zeros(original_p)
            summary_sum = z.new_zeros(d)
            summary_mass = 0
            rate_terms, layer_counts, attention_sizes = [], [], []
            for layer, cgu in enumerate(self.cgus):
                if self.blocks:
                    current, sizes = self.blocks[layer](current, pos, width)
                else:
                    sizes = []
                attention_sizes.append(sizes)
                if current.shape[0] == 0:
                    layer_counts.append(0)
                    continue
                score = cgu(current[None, None], context[b:b + 1, None])[0, 0]
                original_scores = original_scores.scatter(0, pos, score)
                if collect_loss:
                    s = score.float().clamp(1e-6, 1 - 1e-6)
                    rate_terms.append((s * (s / rho).log()
                                       + (1 - s) * ((1 - s) / (1 - rho)).log()).mean())
                count = int((score.detach() >= self.threshold).sum())
                count = min(current.shape[0], self.max_tokens, max(self.min_tokens, count))
                chosen = score.topk(count).indices.sort().values
                keep = torch.zeros_like(score, dtype=torch.bool).scatter(0, chosen, True)
                if (~keep).any():
                    # Removed details cannot return as keys but remain in a single
                    # accumulated summary. Interventions were removed upstream.
                    low_score = score[~keep]
                    st_low = 1 + (low_score - low_score.detach())
                    summary_sum = summary_sum + (current[~keep] * st_low[:, None]).sum(0)
                    summary_mass += int((~keep).sum())
                st = 1 + (score[chosen] - score[chosen].detach())
                current = current[chosen] * st[:, None]
                pos = pos[chosen]
                layer_counts.append(count)
            if summary_mass:
                current = torch.cat((current, (summary_sum / summary_mass)[None]), dim=0)
                pos = torch.cat((pos, pos.new_full((1,), -1)))
            packed.append(current)
            positions.append(pos)
            score_rows.append(original_scores)
            rates.append(torch.stack(rate_terms).mean() if rate_terms else z[b].sum() * 0)
            counts_rows.append(layer_counts)
            att_rows.append(attention_sizes)

        kmax = max(1, max(t.shape[0] for t in packed))
        lengths = torch.tensor([t.shape[0] for t in packed], device=z.device)
        mask = torch.arange(kmax, device=z.device)[None] < lengths[:, None]
        payload = torch.stack([F.pad(t, (0, 0, 0, kmax - t.shape[0])) for t in packed])
        pos_out = torch.stack([F.pad(p, (0, kmax - p.shape[0]), value=-2) for p in positions])
        mu = self.mu(payload)
        logvar = self.logvar(payload).clamp(-10.0, 4.0)
        if collect_loss:
            kl = 0.5 * (mu.float().square() + logvar.float().exp() - 1 - logvar.float())
            kl = (kl * mask.unsqueeze(-1)).sum((1, 2))
        else:
            kl = mu.new_zeros(batch, dtype=torch.float32)
        if self.training and not deterministic:
            if noise is None:
                noise = torch.randn_like(mu)
            if noise.shape != mu.shape:
                raise ValueError("noise must match packed payload shape")
            payload = mu + (0.5 * logvar).exp() * noise.to(mu)
        else:
            payload = mu
        payload = payload * mask.unsqueeze(-1)
        frame = payload.sum(1) / lengths.clamp_min(1).unsqueeze(-1)
        scores = torch.stack(score_rows)
        score_mean = (scores * available).sum(1) / available.sum(1).clamp_min(1)
        key_counts = torch.tensor([int((p >= 0).sum()) for p in positions], device=z.device)
        hard_cpi = key_counts.to(z.dtype) / original_p
        # Only a finite-valued count signal escapes the stochastic payload.
        # Its alphabet is covered by selection_bits; scores supply STE gradients.
        cpi = hard_cpi + (score_mean - score_mean.detach())
        depth = len(self.cgus)
        metadata_nats = (depth * original_p + math.ceil(math.log2(original_p + 2))) * math.log(2)
        metadata = torch.full((batch,), metadata_nats, dtype=torch.float32,
                              device=z.device) * valid
        layer_counts = [torch.tensor([r[i] for r in counts_rows], device=z.device)
                        for i in range(depth)]
        attention_sizes = [[n for row in att_rows for n in row[i]] for i in range(depth)]
        return AdaptiveTokenOutput(payload, mask, scores, frame, cpi, pos_out,
                                   torch.stack(rates), kl, metadata, layer_counts,
                                   target, attention_sizes)
