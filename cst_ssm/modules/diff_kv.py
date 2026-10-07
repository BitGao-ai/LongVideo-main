"""Differential K/V codec prototype and bounded-memory CPI attention.

This module is not a Hugging Face past_key_values implementation. Its error
contract concerns reconstructed K/V tensors, not downstream attention outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

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
    error_norm_floor: float = 1e-6

    def __post_init__(self):
        if self.rank < 1:
            raise ValueError("rank must be positive")
        if self.base_interval < 0:
            raise ValueError("base_interval must be nonnegative (0 disables expiry)")
        if not math.isfinite(self.error_threshold) or self.error_threshold < 0:
            raise ValueError("error_threshold must be finite and nonnegative")
        if not math.isfinite(self.error_norm_floor) or self.error_norm_floor <= 0:
            raise ValueError("error_norm_floor must be finite and positive")
        if not 0 <= self.cpi_keep_ratio <= 1:
            raise ValueError("cpi_keep_ratio must be in [0, 1]")


class LowRankResidualHead(nn.Module):
    """Shared low-rank basis plus per-frame coefficients for K/V residuals."""

    def __init__(self, d: int, rank: int = 8):
        super().__init__()
        if not 0 < rank < d:
            raise ValueError(f"rank({rank}) must be positive and < feature dim({d})")
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
        self._codec_versions: tuple | None = None

    def _check_codec(self):
        versions = tuple((id(p), p._version) for p in self.parameters())
        if self._codec_versions is not None and versions != self._codec_versions:
            raise RuntimeError("codec parameters changed while cached; call reset()")
        return versions

    def _validate_kv(self, K: Tensor, V: Tensor):
        if K.ndim != 2 or K.shape != V.shape or K.shape[-1] != self.d:
            raise ValueError(f"K and V must have identical (B, {self.d}) shapes")
        if K.device != V.device or K.dtype != V.dtype or not K.is_floating_point():
            raise ValueError("K and V must share floating dtype and device")
        if not bool(torch.isfinite(K).all() & torch.isfinite(V).all()):
            raise ValueError("K and V must be finite")
        if self._base_K is not None and (
                K.shape != self._base_K.shape or K.device != self._base_K.device
                or K.dtype != self._base_K.dtype):
            raise ValueError("cached batch shape, dtype or device changed; call reset()")

    def should_insert_base(self, K_t: Tensor, V_t: Tensor,
                           K_hat: Tensor | None = None,
                           V_hat: Tensor | None = None) -> bool:
        """Check actual per-sample reconstruction, never a feature-distance proxy.

        For both K and V, ||truth - reconstruction||_2 / max(||truth||_2, floor)
        must be <= error_threshold. One violating sample refreshes the entire
        batched base. This conservative policy does not dilute outliers by a mean.
        Called before incrementing the frame counter.
        """
        self._validate_kv(K_t, V_t)
        self._check_codec()
        if self._base_K is None:
            return True
        G = int(self.cfg.base_interval)
        if G > 0 and (self._n_frames + 1 - self._base_idx) >= G:
            return True
        if K_hat is None:
            K_hat = self._base_K + self.head_K.decode(self.head_K.encode(K_t - self._base_K))
        if V_hat is None:
            V_hat = self._base_V + self.head_V.decode(self.head_V.encode(V_t - self._base_V))
        for truth, recon in ((K_t, K_hat), (V_t, V_hat)):
            if recon.shape != truth.shape or not bool(torch.isfinite(recon).all()):
                raise ValueError("codec reconstruction must have the input shape and be finite")
            # Float64 avoids overflow of norms for large finite float32 inputs.
            rel = torch.linalg.vector_norm(truth.double() - recon.double(), dim=-1)
            rel = rel / torch.linalg.vector_norm(truth.double(), dim=-1).clamp_min(
                self.cfg.error_norm_floor)
            if bool((rel > self.cfg.error_threshold).any()):
                return True
        return False

    @torch.no_grad()
    def update(self, K_t: Tensor, V_t: Tensor, feat_t: Tensor | None = None,
               t: int | None = None) -> tuple[Tensor, Tensor]:
        """Inference-only insertion; returns reconstructed K/V for this frame.

        feat_t and t are accepted for source compatibility but do not determine
        the error contract. Interval expiry counts update calls, not timestamps.
        Codec weights must remain fixed until reset().
        """
        self._validate_kv(K_t, V_t)
        versions = self._check_codec()
        coeff_K = coeff_V = K_hat = V_hat = None
        if self._base_K is not None:
            coeff_K = self.head_K.encode(K_t - self._base_K)
            coeff_V = self.head_V.encode(V_t - self._base_V)
            K_hat = self._base_K + self.head_K.decode(coeff_K)
            V_hat = self._base_V + self.head_V.decode(coeff_V)
            if not all(bool(torch.isfinite(x).all()) for x in (coeff_K, coeff_V, K_hat, V_hat)):
                raise ValueError("codec produced non-finite coefficients or reconstruction")
        insert = self.should_insert_base(K_t, V_t, K_hat, V_hat)
        self._n_frames += 1
        self._codec_versions = versions
        if insert:
            self._base_K = K_t.detach().clone()
            self._base_V = V_t.detach().clone()
            self._base_idx = self._n_frames
            self._bases.append((self._base_K, self._base_V))
            self._n_base_frames += 1
            return K_t, V_t

        self._residual_K.append(coeff_K.detach().clone())
        self._residual_V.append(coeff_V.detach().clone())
        self._seg_of_frame.append(len(self._bases) - 1)
        self._n_res_frames += 1
        return K_hat, V_hat

    @torch.no_grad()
    def reconstruct_residual_frames(self) -> tuple[Tensor | None, Tensor | None]:
        """Rebuild all cached residual frames; (None, None) when empty."""
        self._check_codec()
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

    def sequence_reconstruction_loss(self, K_seq: Tensor, V_seq: Tensor,
                                     mask: Tensor | None = None) -> Tensor:
        """Reconstruct valid residuals relative to each row's first valid base."""
        if K_seq.shape != V_seq.shape or K_seq.ndim != 3:
            raise ValueError("K_seq and V_seq must have matching (B,L,d) shapes")
        valid = (torch.ones(K_seq.shape[:2], dtype=torch.bool, device=K_seq.device)
                 if mask is None else mask.bool())
        if valid.shape != K_seq.shape[:2]:
            raise ValueError("mask must have shape (B,L)")
        diffs_k, diffs_v = [], []
        for b in range(K_seq.shape[0]):
            k, v = K_seq[b, valid[b]], V_seq[b, valid[b]]
            if k.shape[0] > 1:
                diffs_k.append(k[1:] - k[:1].detach())
                diffs_v.append(v[1:] - v[:1].detach())
        if not diffs_k:
            return (K_seq[valid].sum() + V_seq[valid].sum()) * 0.0
        diff_K, diff_V = torch.cat(diffs_k), torch.cat(diffs_v)
        recon_K = self.head_K.decode(self.head_K.encode(diff_K))
        recon_V = self.head_V.decode(self.head_V.encode(diff_V))
        dtype = torch.float64 if diff_K.dtype == torch.float64 else torch.float32
        return (F.mse_loss(recon_K.to(dtype), diff_K.to(dtype))
                + F.mse_loss(recon_V.to(dtype), diff_V.to(dtype)))


class CPISparseAttention(nn.Module):
    """Bounded CPI attention with prefix-causal selection and separate K/V summaries.

    Causal selection uses a fixed score threshold, not full-sequence top-k:
    future scores cannot change a past output. At most max_key_tokens recent
    high-score keys are kept; evictions join the summary. keep_ratio remains
    a compatibility parameter: its default threshold is 1 - keep_ratio, NOT
    a guarantee on the fraction selected. Cost is O(N * max_key_tokens), with
    O(max_key_tokens) online working state (training activations still grow).

    Noncausal mode uses capped top-k. Uncapped fixed-ratio top-k would have
    quadratic cost; this implementation never claims otherwise. Without CPI,
    the explicit fallback is ordinary dense attention, also quadratic.
    """

    def __init__(self, d: int, n_heads: int = 8, keep_ratio: float = 0.3,
                 causal: bool = True, max_key_tokens: int = 64,
                 score_threshold: float | None = None):
        super().__init__()
        if n_heads < 1 or d % n_heads != 0:
            raise ValueError(f"d({d}) must be divisible by positive n_heads({n_heads})")
        if not 0 <= keep_ratio <= 1 or max_key_tokens < 1:
            raise ValueError("keep_ratio must be in [0,1] and max_key_tokens positive")
        self.keep_ratio = keep_ratio
        self.score_threshold = 1 - keep_ratio if score_threshold is None else score_threshold
        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be in [0,1]")
        self.max_key_tokens = max_key_tokens
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def _attend(self, q: Tensor, keys: list[Tensor], values: list[Tensor]) -> Tensor:
        k, v = torch.stack(keys), torch.stack(values)
        scores = torch.einsum("hd,khd->hk", q, k) * self.scale
        return torch.einsum("hk,khd->hd", scores.softmax(-1), v)

    def forward(self, x: Tensor, cpi: Tensor | None = None,
                mask: Tensor | None = None) -> Tensor:
        """x (B,N,d); cpi and validity mask (B,N); invalid outputs are exactly zero."""
        B, N, d = x.shape
        valid = torch.ones(B, N, dtype=torch.bool, device=x.device) if mask is None else mask.bool()
        if valid.shape != (B, N):
            raise ValueError("mask must have shape (B,N)")
        if cpi is not None:
            if cpi.shape != (B, N) or not bool(torch.isfinite(cpi[valid]).all()):
                raise ValueError("valid CPI scores must be finite with shape (B,N)")
            if bool(((cpi[valid] < 0) | (cpi[valid] > 1)).any()):
                raise ValueError("CPI scores must lie in [0,1]")
        if not bool(torch.isfinite(x[valid]).all()):
            raise ValueError("valid input tokens must be finite")
        if N == 0:
            return x
        # Padding can contain arbitrary values, including NaN; never project it.
        x = torch.where(valid.unsqueeze(-1), x, torch.zeros_like(x))
        h, hd = self.n_heads, self.head_dim
        Q = self.q_proj(x).view(B, N, h, hd)
        K = self.k_proj(x).view(B, N, h, hd)
        V = self.v_proj(x).view(B, N, h, hd)

        if cpi is None:
            allowed = valid[:, None, None, :].expand(B, h, N, N)
            if self.causal:
                allowed = allowed & torch.ones(N, N, device=x.device, dtype=torch.bool).tril()
            scores = (Q.transpose(1, 2) @ K.transpose(1, 2).transpose(-2, -1)) * self.scale
            dead = ~allowed.any(-1, keepdim=True)
            scores = scores.masked_fill(~allowed, float("-inf")).masked_fill(dead, 0)
            weights = scores.softmax(-1).masked_fill(dead, 0)
            out = (weights @ V.transpose(1, 2)).transpose(1, 2).reshape(B, N, d)
        else:
            rows = []
            for b in range(B):
                if not self.causal:
                    idx = valid[b].nonzero(as_tuple=True)[0]
                    n_top = min(self.max_key_tokens, max(1, int(idx.numel() * self.keep_ratio)))
                    top = idx[cpi[b, idx].topk(min(n_top, idx.numel())).indices]
                    is_top = torch.zeros(N, dtype=torch.bool, device=x.device)
                    is_top[top] = True
                    low = valid[b] & ~is_top
                    keys, vals = list(K[b, top].unbind()), list(V[b, top].unbind())
                    summary_v = V[b, low].mean(0) if bool(low.any()) else None
                    if summary_v is not None:
                        keys.append(K[b, low].mean(0)); vals.append(summary_v)
                    outputs = []
                    for t in range(N):
                        if not bool(valid[b, t]):
                            outputs.append(V[b, t] * 0)
                        elif bool(is_top[t]):
                            outputs.append(self._attend(Q[b, t], keys, vals))
                        else:
                            outputs.append(summary_v)
                else:
                    keys, vals = [], []
                    acc_dtype = torch.float64 if K.dtype == torch.float64 else torch.float32
                    sum_k = torch.zeros(h, hd, device=K.device, dtype=acc_dtype)
                    sum_v = torch.zeros(h, hd, device=V.device, dtype=acc_dtype)
                    count = 0
                    outputs = []
                    for t in range(N):
                        if not bool(valid[b, t]):
                            outputs.append(V[b, t] * 0)
                            continue
                        high = bool(cpi[b, t] >= self.score_threshold)
                        if high:
                            keys.append(K[b, t]); vals.append(V[b, t])
                            if len(keys) > self.max_key_tokens:
                                sum_k = sum_k + keys.pop(0)
                                sum_v = sum_v + vals.pop(0)
                                count += 1
                            ak, av = keys, vals
                            if count:
                                ak = keys + [(sum_k / count).to(K.dtype)]
                                av = vals + [(sum_v / count).to(V.dtype)]
                            outputs.append(self._attend(Q[b, t], ak, av))
                        else:
                            sum_k = sum_k + K[b, t]
                            sum_v = sum_v + V[b, t]
                            count += 1
                            outputs.append((sum_v / count).to(V.dtype))
                rows.append(torch.stack(outputs))
            out = torch.stack(rows).reshape(B, N, d)
        return self.out_proj(out).masked_fill(~valid.unsqueeze(-1), 0)


def diffkv_reconstruction_loss(model: nn.Module, K_seq: Tensor, V_seq: Tensor,
                               feat_seq: Tensor | None = None) -> Tensor:
    """Training helper: low-rank K/V residual reconstruction loss (stateless)."""
    cache = getattr(model, "diff_kv", None)
    if cache is None:
        return K_seq.sum() * 0.0
    return cache.sequence_reconstruction_loss(K_seq, V_seq)
