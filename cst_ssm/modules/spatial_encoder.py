"""空间编码层：窗口化局部 Transformer + 跨窗口连接（设计方案 §3.1）。

把单帧特征图划分为不重叠窗口、窗口内自注意力（复杂度 O(P·w²) 而非 O(P²)），
每隔一层做一次循环移位实现跨窗口信息交互（Swin 风格），保证全局感受野。
输出带时间戳的帧级空间特征序列 (B,L,d)，供 EACS 消费。

两种入口：
  WindowedSpatialEncoder : 从像素帧 (B,L,3,H,W) 编码（自带 patch embed）
  FeatureAdapter         : 从预抽取特征 (B,L,P,d_in) 适配（内存友好，配合特征缓存数据流）
低耦合：也可换任意实现 VisionBackbone 协议的骨干（见 llm_interface.VisionBackbone）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def window_partition(x: Tensor, ws: int) -> Tensor:
    """(B, gh, gw, d) → (B*nwin, ws*ws, d)。要求 gh,gw 可被 ws 整除。"""
    B, gh, gw, d = x.shape
    x = x.view(B, gh // ws, ws, gw // ws, ws, d)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, d)
    return x


def window_reverse(win: Tensor, ws: int, gh: int, gw: int) -> Tensor:
    """(B*nwin, ws*ws, d) → (B, gh, gw, d)。"""
    d = win.shape[-1]
    B = win.shape[0] // ((gh // ws) * (gw // ws))
    x = win.view(B, gh // ws, gw // ws, ws, ws, d)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, gh, gw, d)
    return x


class WindowAttention(nn.Module):
    """窗口内多头自注意力。"""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:  # x: (nwin, T, d)
        Bn, T, d = x.shape
        qkv = self.qkv(x).reshape(Bn, T, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (nwin, heads, T, hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(Bn, T, d)
        return self.proj(out)


class SwinBlock(nn.Module):
    """一个窗口注意力块（pre-LN + 窗口注意力 + MLP），可选循环移位。"""

    def __init__(self, dim: int, n_heads: int, window: int, shift: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: Tensor) -> Tensor:  # x: (B, gh, gw, d)
        B, gh, gw, d = x.shape
        ws = self.window
        shortcut = x
        x = self.norm1(x)
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        win = window_partition(x, ws)
        win = self.attn(win)
        x = window_reverse(win, ws, gh, gw)
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class WindowedSpatialEncoder(nn.Module):
    """从像素帧编码为帧级特征序列。"""

    def __init__(self, dim: int = 384, patch: int = 16, window: int = 7,
                 depth: int = 4, n_heads: int = 6, in_ch: int = 3):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.window = window
        self.blocks = nn.ModuleList([
            SwinBlock(dim, n_heads, window, shift=(0 if i % 2 == 0 else window // 2))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def _pad_to_window(self, x: Tensor) -> Tensor:
        # x: (B, gh, gw, d) → pad gh,gw 到 window 的整数倍
        B, gh, gw, d = x.shape
        ws = self.window
        ph = (ws - gh % ws) % ws
        pw = (ws - gw % ws) % ws
        if ph or pw:
            x = F.pad(x.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)
        return x

    def forward(self, frames: Tensor) -> Tensor:
        """frames: (B, L, 3, H, W) → 帧级特征 (B, L, dim)。"""
        B, L, C, H, W = frames.shape
        x = frames.reshape(B * L, C, H, W)
        x = self.patch_embed(x)                       # (B*L, dim, gh, gw)
        x = x.permute(0, 2, 3, 1)                      # (B*L, gh, gw, dim)
        x = self._pad_to_window(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feat = x.mean(dim=(1, 2))                      # 空间池化 → (B*L, dim)
        return feat.view(B, L, -1)


class FeatureAdapter(nn.Module):
    """从预抽取的 patch 特征 (B,L,P,d_in) 适配为帧级特征 (B,L,d_out)。

    配合"特征缓存"数据流（见 data/）：视觉骨干离线跑好，训练只读特征，省显存与算力。
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.attn_pool = nn.MultiheadAttention(d_out, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_out) * 0.02)

    def forward(self, feats: Tensor) -> Tensor:
        B, L, P, d_in = feats.shape
        x = self.norm(self.proj(feats)).reshape(B * L, P, -1)
        q = self.query.expand(B * L, 1, -1)
        pooled, _ = self.attn_pool(q, x, x)           # 注意力池化 → (B*L,1,d_out)
        return pooled.view(B, L, -1)
