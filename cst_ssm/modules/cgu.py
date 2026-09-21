"""Causal gating unit: per-token keep scores with causal context and soft distillation."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pooling import single_query_pool


class CGU(nn.Module):
    """Per-token keep scores s in (0,1).

    Args:
        tokens: (B, L, P, d) spatial tokens.
        context: optional (B, L, d_ctx) causal history summary.
    Returns:
        scores: (B, L, P).
    """

    def __init__(self, d_in: int, d_ctx: int = 0, hidden: int = 256):
        super().__init__()
        self.d_in = d_in
        self.d_ctx = d_ctx
        total_in = d_in + d_ctx
        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, tokens: Tensor, context: Tensor | None = None) -> Tensor:
        B, L, P, d = tokens.shape
        if context is not None:
            W, bias = self.mlp[0].weight, self.mlp[0].bias
            h = F.linear(tokens, W[:, :d], bias) + F.linear(context, W[:, d:]).unsqueeze(2)
        else:
            h = self.mlp[0](tokens)
        h = self.mlp[1](h)
        return torch.sigmoid(self.mlp[2](h).squeeze(-1))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CausalEMA(nn.Module):
    """Causal exponential moving average context: context_t uses frames before t."""

    def __init__(self, dim: int, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.dim = dim

    @staticmethod
    def _exclusive_scan(g: Tensor, b: Tensor) -> Tensor:
        """Exclusive prefix scan over the monoid (g, b); inputs (n, ...)."""
        n = g.shape[0]
        npow = 1 << (n - 1).bit_length()
        if npow > n:
            pad_g = (npow - n, *g.shape[1:])
            pad_b = (npow - n, *b.shape[1:])
            g = torch.cat([g, g.new_ones(pad_g)], dim=0)
            b = torch.cat([b, b.new_zeros(pad_b)], dim=0)
        tmp_g, tmp_b = g.clone(), b.clone()
        step = 1
        while step < npow:
            lo = torch.arange(step - 1, npow - step, step * 2, device=g.device)
            hi = lo + step
            g_lo, b_lo = tmp_g[lo], tmp_b[lo]
            g_hi, b_hi = tmp_g[hi], tmp_b[hi]
            tmp_g[hi] = g_lo * g_hi
            tmp_b[hi] = b_hi + g_hi * b_lo
            step *= 2
        tmp_g[-1] = torch.ones_like(tmp_g[-1])
        tmp_b[-1] = torch.zeros_like(tmp_b[-1])
        step = npow // 2
        while step >= 1:
            lo = torch.arange(step - 1, npow - step, step * 2, device=g.device)
            hi = lo + step
            p_g, p_b = tmp_g[hi], tmp_b[hi]
            t_g, t_b = tmp_g[lo], tmp_b[lo]
            tmp_g[lo], tmp_b[lo] = p_g, p_b
            tmp_g[hi] = p_g * t_g
            tmp_b[hi] = t_g * p_b + t_b
            step //= 2
        return tmp_b[:n]

    def forward(self, feat: Tensor) -> Tensor:
        """feat (B, L, d) -> context (B, L, d) with context[:, 0] = 0."""
        B, L, d = feat.shape
        if L <= 1:
            return torch.zeros(B, L, d, device=feat.device, dtype=feat.dtype)
        alpha = float(self.alpha)
        scan_dtype = (torch.float32 if feat.dtype in (torch.bfloat16, torch.float16)
                      else feat.dtype)
        x = feat.to(scan_dtype)
        g = x.new_full((L,), alpha)
        b = (1.0 - alpha) * x
        s = self._exclusive_scan(
            g.unsqueeze(1).unsqueeze(2), b.permute(1, 0, 2))
        ctx = s.permute(1, 0, 2).to(feat.dtype)
        return ctx


class SSMStateContext(nn.Module):
    """Projects EACS committed states to a CGU causal context."""

    def __init__(self, d_state_total: int, d_ctx: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_state_total, d_ctx),
            nn.GELU(),
            nn.Linear(d_ctx, d_ctx),
        )
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h_pi: Tensor) -> Tensor:
        """h_pi (B, L, H, N) complex -> context (B, L, d_ctx) real."""
        B, L, H, N = h_pi.shape
        h_real = torch.cat([h_pi.real, h_pi.imag], dim=-1)
        h_flat = h_real.reshape(B, L, -1)
        return self.proj(h_flat)


class TokenDistiller(nn.Module):
    """Soft information-bottleneck compression of tokens to frame features.

    High-score tokens are kept via weighted mean; low-score content is summarized
    with attention pooling biased by log(1 - s). Returns frame features and CPI.
    """

    def __init__(self, d: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
        self.attn_pool = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

    def forward(self, tokens: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
        B, L, P, d = tokens.shape
        cpi_frame = scores.mean(dim=-1)

        w = scores.unsqueeze(-1)
        retained = (tokens * w).sum(dim=2) / (w.sum(dim=2) + 1e-6)

        compressed = single_query_pool(
            self.attn_pool, self.query.reshape(d), tokens,
            bias=torch.log1p(-scores).clamp_min(-14.0))

        g = self.gate(torch.cat([retained, compressed], dim=-1))
        frame_feat = g * retained + (1 - g) * compressed

        return frame_feat, cpi_frame


class CPIBDistill(nn.Module):
    """Full CPIB-Distill block: CGU + causal context + token distillation.

    Inserted between the spatial encoder and the temporal model.
    tokens (B,L,P,d) -> (frame_feat (B,L,d), cpi_frame (B,L), cpi_tokens (B,L,P)).
    context_mode "ema" uses a causal EMA of frame means; "ssm" uses projected
    EACS committed states passed via ssm_context.
    """

    def __init__(self, d: int, hidden: int = 256, ema_alpha: float = 0.9, n_heads: int = 8,
                 context_mode: str = "ema", ssm_state_dim: int = 0):
        super().__init__()
        self.context_mode = context_mode
        self.cgu = CGU(d_in=d, d_ctx=d, hidden=hidden)
        self.ema = CausalEMA(d, alpha=ema_alpha)
        self.distiller = TokenDistiller(d, n_heads=n_heads)
        self.ssm_ctx: SSMStateContext | None = None
        if context_mode == "ssm" and ssm_state_dim > 0:
            self.ssm_ctx = SSMStateContext(ssm_state_dim, d)

    def forward(self, tokens: Tensor, ssm_context: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """tokens (B,L,P,d), optional ssm_context (B,L,d) -> (frame_feat, cpi_frame, scores)."""
        if self.context_mode == "ssm" and ssm_context is not None:
            context = ssm_context
        else:
            frame_mean = tokens.mean(dim=2)
            context = self.ema(frame_mean)

        scores = self.cgu(tokens, context)

        frame_feat, cpi_frame = self.distiller(tokens, scores)

        return frame_feat, cpi_frame, scores

    def project_ssm_states(self, h_pi: Tensor) -> Tensor:
        """Project EACS committed states (B,L,H,N) to a CGU context (B,L,d)."""
        if self.ssm_ctx is not None:
            return self.ssm_ctx(h_pi)
        return h_pi.real.mean(dim=-1)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
