"""Error-bounded differential KV cache and causal CPI sparse attention (inference)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DiffKVConfig:
    rank: int = 8
    base_interval: int = 8
    error_threshold: float = 0.1
    cpi_sparse: bool = True
    cpi_keep_ratio: float = 0.3


class LowRankResidualHead(nn.Module):
    """Shared low-rank basis plus per-frame coefficients for K/V residuals."""

    def __init__(self, d: int, rank: int = 8):
        super().__init__()
        if rank >= d:
            raise ValueError(f"rank({rank}) must be < feature dim({d})")
        self.d = d
        self.rank = rank
        q, _ = torch.linalg.qr(torch.randn(d, rank))
        self.basis = nn.Parameter(q.contiguous())
        self.encoder = nn.Linear(d, rank)
        nn.init.zeros_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

    def encode(self, diff: Tensor) -> Tensor:
        """Residual vector (..., d) -> stored coefficients (..., r)."""
        return self.encoder(diff)

    def decode(self, coeffs: Tensor) -> Tensor:
        """Coefficients (..., r) -> reconstructed residual (..., d)."""
        return F.linear(coeffs, self.basis)

    @property
    def per_frame_storage(self) -> int:
        return self.rank


class DifferentialKVCache(nn.Module):
    """Streaming KV cache: full base frames plus low-rank residual coefficients.

    Training uses the stateless sequence_reconstruction_loss; streaming state
    below is inference-only.
    """

    def __init__(self, cfg: DiffKVConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d = d_model
        self.head_K = LowRankResidualHead(d_model, cfg.rank)
        self.head_V = LowRankResidualHead(d_model, cfg.rank)
        self.reset()

    def reset(self):
        self._base_K: Tensor | None = None
        self._base_V: Tensor | None = None
        self._base_feat: Tensor | None = None
        self._base_idx: int = 0
        self._residual_K: list[Tensor] = []
        self._residual_V: list[Tensor] = []
        self._seg_of_frame: list[int] = []
        self._bases: list[tuple[Tensor, Tensor]] = []
        self._n_frames: int = 0
        self._n_base_frames: int = 0
        self._n_res_frames: int = 0

    def should_insert_base(self, feat_t: Tensor) -> bool:
        """Insert a base frame on high relative residual energy or interval expiry."""
        if self._base_feat is None:
            return True
        diff = torch.linalg.vector_norm(feat_t - self._base_feat, dim=-1)
        base = torch.linalg.vector_norm(self._base_feat, dim=-1)
        rel = (diff / (base + 1e-6)).mean()
        if bool(rel.item() > self.cfg.error_threshold):
            return True
        G = int(self.cfg.base_interval)
        return G > 0 and (self._n_frames - self._base_idx) >= G

    def update(self, K_t: Tensor, V_t: Tensor, feat_t: Tensor,
               t: int) -> tuple[Tensor, Tensor]:
        """Streaming update; returns reconstructed full (K, V) for frame t.

        K_t, V_t: (B, d); feat_t: (B, d). Residual frames store only r coefficients.
        """
        self._n_frames += 1
        if self.should_insert_base(feat_t):
            self._base_K = K_t.detach().clone()
            self._base_V = V_t.detach().clone()
            self._base_feat = feat_t.detach().clone()
            self._base_idx = self._n_frames
            self._bases.append((self._base_K, self._base_V))
            self._n_base_frames += 1
            return K_t, V_t

        coeff_K = self.head_K.encode(K_t - self._base_K)
        coeff_V = self.head_V.encode(V_t - self._base_V)
        self._residual_K.append(coeff_K.detach())
        self._residual_V.append(coeff_V.detach())
        self._seg_of_frame.append(len(self._bases) - 1)
        self._n_res_frames += 1
        return (self._base_K + self.head_K.decode(coeff_K),
                self._base_V + self.head_V.decode(coeff_V))

    @torch.no_grad()
    def reconstruct_residual_frames(self) -> tuple[Tensor | None, Tensor | None]:
        """Rebuild all cached residual frames; (None, None) when empty."""
        if not self._residual_K:
            return None, None
        K = torch.stack([self._bases[s][0] + self.head_K.decode(c)
                         for s, c in zip(self._seg_of_frame, self._residual_K)])
        V = torch.stack([self._bases[s][1] + self.head_V.decode(c)
                         for s, c in zip(self._seg_of_frame, self._residual_V)])
        return K, V

    @property
    def compression_ratio(self) -> float:
        """Stored floats over full-storage floats (K+V channels)."""
        if self._n_frames == 0:
            return 1.0
        full = self._n_frames * 2 * self.d
        actual = (self._n_base_frames * 2 * self.d
                  + self._n_res_frames * 2 * self.cfg.rank)
        return actual / full

    @property
    def n_base_frames(self) -> int:
        return self._n_base_frames

    @property
    def n_residual_frames(self) -> int:
        return self._n_res_frames

    @property
    def stats(self) -> dict:
        return {"n_frames": self._n_frames, "n_base": self._n_base_frames,
                "n_residual": self._n_res_frames, "rank": self.cfg.rank,
                "d": self.d, "compression_ratio": self.compression_ratio}

    def sequence_reconstruction_loss(self, K_seq: Tensor, V_seq: Tensor) -> Tensor:
        """Stateless training loss: frame 0 is the base; reconstruct later residuals.

        K_seq, V_seq: (B, L, d). Base slices are detached; per-batch independent.
        """
        if K_seq.shape[1] < 2:
            return K_seq.sum() * 0.0
        diff_K = K_seq[:, 1:] - K_seq[:, :1].detach()
        diff_V = V_seq[:, 1:] - V_seq[:, :1].detach()
        recon_K = self.head_K.decode(self.head_K.encode(diff_K))
        recon_V = self.head_V.decode(self.head_V.encode(diff_V))
        return F.mse_loss(recon_K, diff_K) + F.mse_loss(recon_V, diff_V)


class CPISparseAttention(nn.Module):
    """Causal CPI sparse attention: top-k queries attend causally, rest use summaries.

    Low-CPI positions output the causal CPI-weighted summary of prior low-CPI
    values; high-CPI (top-k) positions attend over high-CPI keys plus their own
    summary. x (B,N,d), optional cpi (B,N) -> (B,N,d).
    """

    def __init__(self, d: int, n_heads: int = 8, keep_ratio: float = 0.3,
                 causal: bool = True):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d({d}) must be divisible by n_heads({n_heads})")
        self.keep_ratio = keep_ratio
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: Tensor, cpi: Tensor | None = None) -> Tensor:
        """x (B,N,d), optional cpi (B,N); no CPI falls back to full attention."""
        B, N, d = x.shape
        h, hd = self.n_heads, self.head_dim
        Q = self.q_proj(x).view(B, N, h, hd)
        K = self.k_proj(x).view(B, N, h, hd)
        V = self.v_proj(x).view(B, N, h, hd)

        if cpi is None:
            Qt, Kt, Vt = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
            attn = (Qt @ Kt.transpose(-2, -1)) * self.scale
            if self.causal:
                mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool),
                                  diagonal=1)
                attn = attn.masked_fill(mask, float("-inf"))
            out = (attn.softmax(-1) @ Vt).transpose(1, 2).reshape(B, N, d)
            return self.out_proj(out)

        k = max(1, int(N * self.keep_ratio))
        _, topk_idx = cpi.topk(k, dim=-1)

        is_low = torch.ones(B, N, dtype=torch.bool, device=x.device)
        is_low.scatter_(1, topk_idx, False)
        w = cpi.clamp_min(0.0) * is_low
        cum_wV = (V * w[:, :, None, None]).cumsum(dim=1)
        cum_w = w.cumsum(dim=1)
        has_hist = cum_w > 1e-8
        summary = cum_wV / cum_w[:, :, None, None].clamp_min(1e-8)

        out_low = torch.where(has_hist[:, :, None, None], summary, V)

        idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, k, h, hd)
        Q_top = Q.gather(1, idx).transpose(1, 2)
        K_top = K.gather(1, idx).transpose(1, 2)
        V_top = V.gather(1, idx).transpose(1, 2)
        S_top = summary.gather(1, idx).transpose(1, 2)

        attn_kk = (Q_top @ K_top.transpose(-2, -1)) * self.scale
        attn_s = (Q_top * S_top).sum(-1, keepdim=True) * self.scale
        if self.causal:
            pos_q = topk_idx.unsqueeze(2)
            pos_k = topk_idx.unsqueeze(1)
            block = (pos_k > pos_q).view(B, 1, k, k)
            attn_kk = attn_kk.masked_fill(block, float("-inf"))
        w = torch.softmax(torch.cat([attn_kk, attn_s], dim=-1), dim=-1)
        w_kk, w_s = w[..., :k], w[..., k:]
        out_top = w_kk @ V_top + w_s * S_top
        out_top = out_top.transpose(1, 2).reshape(B, k, d)

        out = out_low.reshape(B, N, d)
        out = out.scatter(1, topk_idx.unsqueeze(-1).expand(B, k, d), out_top)
        return self.out_proj(out)


def diffkv_reconstruction_loss(model: nn.Module, K_seq: Tensor, V_seq: Tensor,
                               feat_seq: Tensor | None = None) -> Tensor:
    """Training helper: low-rank K/V residual reconstruction loss (stateless)."""
    cache = getattr(model, "diff_kv", None)
    if cache is None:
        return K_seq.sum() * 0.0
    return cache.sequence_reconstruction_loss(K_seq, V_seq)
