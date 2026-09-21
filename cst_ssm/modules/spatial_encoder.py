"""Spatial encoding: windowed local Transformer and pre-extracted feature adapter."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .pooling import single_query_pool
from ..dtypes import align_to_param


def window_partition(x: Tensor, ws: int) -> Tensor:
    """(B, gh, gw, d) -> (B*nwin, ws*ws, d); gh and gw must be divisible by ws."""
    B, gh, gw, d = x.shape
    x = x.view(B, gh // ws, ws, gw // ws, ws, d)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, d)
    return x


def window_reverse(win: Tensor, ws: int, gh: int, gw: int) -> Tensor:
    """(B*nwin, ws*ws, d) -> (B, gh, gw, d)."""
    d = win.shape[-1]
    B = win.shape[0] // ((gh // ws) * (gw // ws))
    x = win.view(B, gh // ws, gw // ws, ws, ws, d)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, gh, gw, d)
    return x


class WindowAttention(nn.Module):
    """Multi-head self-attention inside one window."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x: (nwin_total, T, d); mask: optional (nwin, T, T) with True = blocked."""
        Bn, T, d = x.shape
        qkv = self.qkv(x).reshape(Bn, T, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(Bn // nw, nw, self.n_heads, T, T)
            m = mask[None, :, None]
            dead = m.all(-1, keepdim=True)
            attn = attn.masked_fill(m & ~dead, float("-inf"))
            attn = attn.softmax(-1)
            attn = attn.masked_fill(dead, 0.0)
            attn = attn.view(Bn, self.n_heads, T, T)
        else:
            attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(Bn, T, d)
        return self.proj(out)


class SwinBlock(nn.Module):
    """Window attention block (pre-LN + attention + MLP) with optional cyclic shift."""

    def __init__(self, dim: int, n_heads: int, window: int, shift: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def _shift_mask(self, gh: int, gw: int, device) -> Tensor:
        """Shifted-window mask; returns (nwin, ws*ws, ws*ws) with True = blocked."""
        ws, sh = self.window, self.shift
        img_mask = torch.zeros(1, gh, gw, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        w_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        cnt = 0
        for hs in h_slices:
            for wsl in w_slices:
                img_mask[:, hs, wsl, :] = cnt
                cnt += 1
        mask_w = window_partition(img_mask, ws).squeeze(-1)
        return mask_w.unsqueeze(1) != mask_w.unsqueeze(2)

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        """x: (B, gh, gw, d); valid: optional (1, gh, gw) marking real (non-padded) patches."""
        B, gh, gw, d = x.shape
        ws = self.window
        shortcut = x
        x = self.norm1(x)
        attn_mask = None
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            if valid is not None:
                valid = torch.roll(valid, shifts=(-self.shift, -self.shift), dims=(1, 2))
            attn_mask = self._shift_mask(gh, gw, x.device)
        if valid is not None:
            vw = window_partition(valid.unsqueeze(-1).to(x.dtype), ws).squeeze(-1) > 0.5
            key_mask = ~vw.unsqueeze(1).expand(-1, ws * ws, -1)
            attn_mask = key_mask if attn_mask is None else (attn_mask | key_mask)
        win = window_partition(x, ws)
        win = self.attn(win, attn_mask)
        x = window_reverse(win, ws, gh, gw)
        if self.shift:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class WindowedSpatialEncoder(nn.Module):
    """Encodes pixel frames to a per-frame feature sequence."""

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

    def _pad_to_window(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """Pad (B, gh, gw, d) to window multiples; also return the validity mask."""
        B, gh, gw, d = x.shape
        ws = self.window
        ph = (ws - gh % ws) % ws
        pw = (ws - gw % ws) % ws
        if not (ph or pw):
            return x, None
        x = F.pad(x.permute(0, 3, 1, 2), (0, pw, 0, ph)).permute(0, 2, 3, 1)
        valid = torch.zeros(1, gh + ph, gw + pw, dtype=torch.bool, device=x.device)
        valid[:, :gh, :gw] = True
        return x, valid

    def forward(self, frames: Tensor) -> Tensor:
        """frames (B, L, 3, H, W) -> frame features (B, L, dim)."""
        B, L, C, H, W = frames.shape
        x = frames.reshape(B * L, C, H, W)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        gh0, gw0 = x.shape[1], x.shape[2]
        x, valid = self._pad_to_window(x)
        for blk in self.blocks:
            x = blk(x, valid)
        x = self.norm(x)
        x = x[:, :gh0, :gw0]
        feat = x.mean(dim=(1, 2))
        return feat.view(B, L, -1)


class FeatureAdapter(nn.Module):
    """Adapts pre-extracted patch features (B,L,P,d_in) to frame features (B,L,d_out)."""

    def __init__(self, d_in: int, d_out: int, n_heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.attn_pool = nn.MultiheadAttention(d_out, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_out) * 0.02)

    def get_tokens(self, feats: Tensor) -> Tensor:
        """Projected per-token features (B,L,P,d_out)."""
        if feats.shape[-1] != self.proj.in_features:
            raise ValueError(
                f"FeatureAdapter input dim {feats.shape[-1]} != configured feat_dim "
                f"{self.proj.in_features}. feat_dim is fixed by the extraction backbone; "
                f"check model.feat_dim against the manifest features.")
        feats = align_to_param(feats, self.proj.weight)
        return self.norm(self.proj(feats))

    def forward(self, feats: Tensor) -> Tensor:
        return single_query_pool(self.attn_pool, self.query.reshape(-1),
                                 self.get_tokens(feats))
