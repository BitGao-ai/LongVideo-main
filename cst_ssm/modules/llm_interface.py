"""LLM 推理层（设计方案 §3.4）：文本 Token 全自注意力 + 视觉状态交叉注意力（无视觉历史 KV 缓存）。

- GatedCrossAttention : Flamingo 风格 tanh 门控交叉注意力，初始门=0（初始等价原 LLM，训练稳定）。
- VisualConditionedLM : 自包含 decoder-only LLM，层间插入门控交叉注意力到视觉状态。默认小规模，
                        既是可训练的具体实现，也是无需下载 7B 权重即可 smoke test 的 stand-in。
- 低耦合：VisionBackbone / LLMBackbone 为结构化协议，可换任意骨干（含 HF 因果 LM，见 README 集成说明）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------- 低耦合协议 ---------------------------
@runtime_checkable
class VisionBackbone(Protocol):
    def forward(self, frames: Tensor) -> Tensor:  # (B,L,3,H,W) 或 (B,L,P,d) → (B,L,d)
        ...


@runtime_checkable
class LLMBackbone(Protocol):
    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None, labels: Tensor | None = None) -> dict:
        ...


# --------------------------- 模块 ---------------------------
class GatedCrossAttention(nn.Module):
    """文本查询 → 视觉状态键/值的门控交叉注意力。tanh 门初始 0，保证初始不改变 LLM 行为。"""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))   # tanh 门，初始 0

    def forward(self, text: Tensor, visual: Tensor,
                visual_mask: Tensor | None = None) -> Tensor:
        B, Lt, d = text.shape
        Lv = visual.shape[1]
        q = self.q(self.norm_q(text)).view(B, Lt, self.n_heads, -1).transpose(1, 2)
        kv = self.kv(self.norm_kv(visual)).view(B, Lv, 2, self.n_heads, -1)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)     # (B,heads,Lv,hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,heads,Lt,Lv)
        if visual_mask is not None:
            attn = attn.masked_fill(~visual_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Lt, d)
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
        attn = (q @ k.transpose(-2, -1)) * self.scale
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), 1)
        attn = attn.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:                # (B,T) True=有效
            attn = attn.masked_fill(~key_padding_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, d)
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
    cross_every: int = 2       # 每隔多少层插入一次门控交叉注意力
    max_len: int = 2048


class VisualConditionedLM(nn.Module):
    """decoder-only LLM，层间插入门控交叉注意力到视觉状态。"""

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
        self.lm_head.weight = self.embed.weight            # 权重绑定省显存

    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None,
                visual_mask: Tensor | None = None,
                labels: Tensor | None = None) -> dict:
        B, T = input_ids.shape
        h = self.embed(input_ids) + self.pos[:, :T]
        for i, layer in enumerate(self.layers):
            h = layer(h, attention_mask)
            if str(i) in self.cross:
                h = self.cross[str(i)](h, visual_states, visual_mask)
        h = self.norm(h)
        logits = self.lm_head(h)
        out = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            out["loss"] = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return out
