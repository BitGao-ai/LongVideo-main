"""LLM layer: decoder-only LM with gated cross-attention to visual states."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


@runtime_checkable
class VisionBackbone(Protocol):
    def forward(self, frames: Tensor) -> Tensor:
        ...


@runtime_checkable
class LLMBackbone(Protocol):
    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None, labels: Tensor | None = None,
                need_logits: bool = True) -> dict:
        ...


def _safe_softmax(attn: Tensor, dim: int = -1) -> Tensor:
    """Softmax that outputs zeros (not NaN) for fully masked rows."""
    visible = torch.isfinite(attn).any(dim=dim, keepdim=True)
    w = attn.masked_fill(~visible, 0.0).softmax(dim)
    return w * visible


def _sdpa(q: Tensor, k: Tensor, v: Tensor, scale: float,
          allow: Tensor | None = None, is_causal: bool = False) -> Tensor:
    """Memory-efficient attention via SDPA with zero output for fully masked rows.

    Args:
        allow: optional bool mask broadcastable to (B,heads,Lq,Lk); True = visible.
    """
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow,
                                         is_causal=is_causal, scale=scale)
    if allow is not None:
        dead = ~allow.any(dim=-1, keepdim=True)
        out = out.masked_fill(dead, 0.0)
    return out


class GatedCrossAttention(nn.Module):
    """Text-to-visual gated cross-attention; zero-init gate preserves base LM behavior."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, text: Tensor, visual: Tensor,
                visual_mask: Tensor | None = None) -> Tensor:
        B, Lt, d = text.shape
        Lv = visual.shape[1]
        q = self.q(self.norm_q(text)).view(B, Lt, self.n_heads, -1).transpose(1, 2)
        kv = self.kv(self.norm_kv(visual)).view(B, Lv, 2, self.n_heads, -1)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)
        allow = None if visual_mask is None else visual_mask[:, None, None, :]
        out = _sdpa(q, k, v, self.scale, allow=allow)
        out = out.transpose(1, 2).reshape(B, Lt, d)
        return text + torch.tanh(self.gate) * self.proj(out)


class _CausalSelfAttn(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if key_padding_mask is None:
            out = _sdpa(q, k, v, self.scale, is_causal=True)
        else:
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril_()
            allow = causal & key_padding_mask[:, None, None, :]
            out = _sdpa(q, k, v, self.scale, allow=allow)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class _DecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttn(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class LLMConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layer: int = 6
    n_head: int = 8
    cross_every: int = 2
    max_len: int = 2048
    grad_checkpoint: bool = False  # Per-layer checkpointing; compute for memory.
    loss_chunk: int = 1024         # Chunked LM loss over (B,T,V); 0 disables.


def _ce_chunk_sum(hidden: Tensor, labels: Tensor, lm_head: nn.Module,
                  ignore_index: int) -> Tensor:
    """Cross-entropy sum over one token chunk (checkpointable unit)."""
    return F.cross_entropy(lm_head(hidden), labels,
                           ignore_index=ignore_index, reduction="sum")


def chunked_lm_loss(hidden: Tensor, labels: Tensor, lm_head: nn.Module,
                    chunk_size: int, ignore_index: int = -100) -> Tensor:
    """Chunked next-token cross-entropy without materializing (B,T,V) logits."""
    B, T, d = hidden.shape
    if T < 2:
        return hidden.sum() * 0.0
    flat_h = hidden[:, :-1].reshape(-1, d)
    flat_y = labels[:, 1:].reshape(-1)
    n_valid = (flat_y != ignore_index).sum()
    use_ckpt = torch.is_grad_enabled() and flat_h.requires_grad
    total = flat_h.new_zeros((), dtype=torch.float32)
    for s in range(0, flat_h.shape[0], chunk_size):
        hs, ys = flat_h[s:s + chunk_size], flat_y[s:s + chunk_size]
        if use_ckpt:
            part = checkpoint(_ce_chunk_sum, hs, ys, lm_head, ignore_index,
                              use_reentrant=False)
        else:
            part = _ce_chunk_sum(hs, ys, lm_head, ignore_index)
        total = total + part
    return total / n_valid.clamp_min(1).to(total.dtype)


def use_chunked_loss(loss_chunk: int, labels: Tensor | None,
                     need_logits: bool = True) -> bool:
    """Whether to use chunked CE: enabled with labels and no logits consumer."""
    if loss_chunk <= 0 or labels is None:
        return False
    return torch.is_grad_enabled() or not need_logits


class VisualConditionedLM(nn.Module):
    """Decoder-only LM with gated cross-attention to visual states."""

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.dim))
        self.layers = nn.ModuleList([_DecoderLayer(cfg.dim, cfg.n_head) for _ in range(cfg.n_layer)])
        self.cross = nn.ModuleDict({
            str(i): GatedCrossAttention(cfg.dim, cfg.n_head)
            for i in range(cfg.n_layer) if i % cfg.cross_every == 0
        })
        self.norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

    def _maybe_ckpt(self, module, *args):
        """Apply gradient checkpointing to one layer when enabled for training."""
        if not (self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return module(*args)
        return checkpoint(module, *args, use_reentrant=False)

    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None,
                visual_mask: Tensor | None = None,
                labels: Tensor | None = None,
                need_logits: bool = True) -> dict:
        """need_logits False skips (B,T,V) materialization even without grad."""
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(
                f"Text length T={T} exceeds LLMConfig.max_len={self.cfg.max_len}.")
        h = self.embed(input_ids) + self.pos[:, :T]
        for i, layer in enumerate(self.layers):
            h = self._maybe_ckpt(layer, h, attention_mask)
            if str(i) in self.cross:
                h = self._maybe_ckpt(self.cross[str(i)], h, visual_states, visual_mask)
        h = self.norm(h)
        if use_chunked_loss(self.cfg.loss_chunk, labels, need_logits):
            return {"loss": chunked_lm_loss(h, labels, self.lm_head, self.cfg.loss_chunk)}
        logits = self.lm_head(h)
        out = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            out["loss"] = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return out
