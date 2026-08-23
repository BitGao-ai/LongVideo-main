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

from .pooling import single_query_pool


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

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """x: (nwin_total, T, d)；mask: 可选 (nwin, T, T) bool，True=禁止该注意力对。

        mask 的第 0 维是**单张图的窗口数** nwin，而 x 的第 0 维是 batch·nwin。
        必须先把 attn 拆成 (batch, nwin, heads, T, T) 再按 nwin 广播——
        直接 `attn.masked_fill(mask.unsqueeze(1))` 只在 nwin==1 或 batch==1 时侥幸成立，
        否则 dim0 (batch·nwin vs nwin) 无法广播直接报错（224×224/window=7 即 nwin=4）。

        整行被屏蔽时走安全 softmax（与 diff_kv._safe_softmax 同思路）：补齐位 mask 接入后，
        完全落在补齐区的窗口整行皆为 -inf，直接 softmax 会得到 0/0 = NaN 并污染整帧。
        """
        Bn, T, d = x.shape
        qkv = self.qkv(x).reshape(Bn, T, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (nwin_total, heads, T, hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(Bn // nw, nw, self.n_heads, T, T)
            m = mask[None, :, None]                       # (1,nwin,1,T,T) 广播
            dead = m.all(-1, keepdim=True)                # 整行被屏蔽 → softmax 会 NaN
            # 先把死行的 mask 抹掉，避开 0/0；softmax 后再把该行整体置零，
            # 语义上等于"该 query 没有任何可见 key，输出不携带信息"。
            attn = attn.masked_fill(m & ~dead, float("-inf"))
            attn = attn.softmax(-1)
            attn = attn.masked_fill(dead, 0.0)
            attn = attn.view(Bn, self.n_heads, T, T)
        else:
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

    def _shift_mask(self, gh: int, gw: int, device) -> Tensor:
        """标准 Swin 移位窗口 attention mask：roll 后处于不同区域的 token 禁止互相注意。

        无此 mask 时移位窗口会把空间上不相邻的区域卷进同一窗口，造成边界信息泄漏。
        返回 (nwin, ws*ws, ws*ws) bool，True=屏蔽。
        """
        ws, sh = self.window, self.shift
        img_mask = torch.zeros(1, gh, gw, 1, device=device)
        h_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        w_slices = (slice(0, -ws), slice(-ws, -sh), slice(-sh, None))
        cnt = 0
        for hs in h_slices:
            for wsl in w_slices:
                img_mask[:, hs, wsl, :] = cnt
                cnt += 1
        mask_w = window_partition(img_mask, ws).squeeze(-1)      # (nwin, ws*ws)
        return mask_w.unsqueeze(1) != mask_w.unsqueeze(2)        # (nwin, T, T)

    def forward(self, x: Tensor, valid: Tensor | None = None) -> Tensor:
        """x: (B, gh, gw, d)；valid: 可选 (1, gh, gw) bool，False=为对齐窗口补出的零 patch。

        valid 存在时，补齐位不得作为 key/value 参与注意力：它们是零向量，混进 softmax
        分母等价于一个与分辨率强相关的软温度缩放（240×240/ws=7 时补齐位占 49%），
        分辨率一变行为就跳变。仍允许补齐位作为 query（其输出后续会被裁掉）。
        """
        B, gh, gw, d = x.shape
        ws = self.window
        shortcut = x
        x = self.norm1(x)
        attn_mask = None
        if self.shift:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
            if valid is not None:      # mask 必须跟着 roll，否则与 token 错位
                valid = torch.roll(valid, shifts=(-self.shift, -self.shift), dims=(1, 2))
            attn_mask = self._shift_mask(gh, gw, x.device)
        if valid is not None:
            # (1,gh,gw) → (nwin, T)：窗口内每个 key 是否有效
            vw = window_partition(valid.unsqueeze(-1).to(x.dtype), ws).squeeze(-1) > 0.5
            key_mask = ~vw.unsqueeze(1).expand(-1, ws * ws, -1)      # (nwin,T,T) True=屏蔽
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

    def _pad_to_window(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        """x: (B, gh, gw, d) → (补齐后的 x, valid)。

        valid: (1, gh_pad, gw_pad) bool，False=补出来的零 patch；无需补齐时返回 None。
        整批共用一张（补齐量只由分辨率决定，与样本无关），因此第 0 维取 1 靠广播即可，
        也正好对上 SwinBlock 里 (nwin, T, T) 这一档 mask 的形状约定。
        """
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
        """frames: (B, L, 3, H, W) → 帧级特征 (B, L, dim)。"""
        B, L, C, H, W = frames.shape
        x = frames.reshape(B * L, C, H, W)
        x = self.patch_embed(x)                       # (B*L, dim, gh, gw)
        x = x.permute(0, 2, 3, 1)                      # (B*L, gh, gw, dim)
        gh0, gw0 = x.shape[1], x.shape[2]              # 补齐前的真实网格
        x, valid = self._pad_to_window(x)
        for blk in self.blocks:
            x = blk(x, valid)
        x = self.norm(x)
        # 裁掉为对齐窗口补的零 patch 再池化：否则补齐位会稀释帧级特征，
        # 且稀释比例随分辨率与 window 的整除关系跳变（gh=15,ws=7 时补 6 行，占 29%）。
        x = x[:, :gh0, :gw0]
        feat = x.mean(dim=(1, 2))                      # 空间池化 → (B*L, dim)
        return feat.view(B, L, -1)


class FeatureAdapter(nn.Module):
    """从预抽取的 patch 特征 (B,L,P,d_in) 适配为帧级特征 (B,L,d_out)。

    配合“特征缓存”数据流（见 data/）：视觉骨干离线跑好，训练只读特征，省显存与算力。
    提供 get_tokens() 返回投影后的逐 token 特征 (B,L,P,d_out)，供 CPIBDistill/CGU 消费。
    """

    def __init__(self, d_in: int, d_out: int, n_heads: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.attn_pool = nn.MultiheadAttention(d_out, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_out) * 0.02)

    def get_tokens(self, feats: Tensor) -> Tensor:
        """返回投影+归一化后的逐 token 特征 (B,L,P,d_out)，供 CGU 蒸馏。"""
        # 维度错配在这里就说清楚：否则 F.linear 只会抛一句 "mat1 and mat2 shapes cannot be
        # multiplied (9216x2560 and 768x384)"，摊平后的 B*L*P 与 in/out 混在一起，
        # 看不出到底是配置的 feat_dim 错了还是数据错了。
        if feats.shape[-1] != self.proj.in_features:
            raise ValueError(
                f"FeatureAdapter 输入特征维 {feats.shape[-1]} ≠ 建模用的 feat_dim "
                f"{self.proj.in_features}。feat_dim 由抽特征的视觉塔唯一决定"
                f"（如 Qwen3-VL-4B=2560），请检查 --config 里的 model.feat_dim 是否与 "
                f"--manifest 指向的特征同源；训练脚本用 cst_ssm.data.align_feat_dim 自动对齐。")
        return self.norm(self.proj(feats))              # (B, L, P, d_out)

    def forward(self, feats: Tensor) -> Tensor:
        # 单 query 注意力池化，走 pooling.single_query_pool：与 nn.MultiheadAttention
        # 数学等价，但不物化 (B,L,P,d) 的 K/V，也不把它们留给反传。参数与 state_dict 不变。
        return single_query_pool(self.attn_pool, self.query.reshape(-1),
                                 self.get_tokens(feats))
