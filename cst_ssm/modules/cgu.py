"""因果门控单元（Causal Gating Unit, CGU）——创新点1 CPIB-Distill 的核心组件。

设计文档 §1（CHST_创新点细化.pdf）：
  - 对每个时空 token x_{t,i}，结合 SSM 历史状态 c_{1:t-1}，输出保留权重 s_{t,i}=σ(MLP(x⊕c))
  - 参数 < 0.3M/层；2层 MLP（隐藏维 256）
  - s_{t,i} 同时驱动：①Token 蒸馏（信息瓶颈）②帧级 CPI 信号 s̄_t（贯通 EventGate/稀疏注意力）

本实现：
  - CGU 模块：输入 (B,L,P,d) token 特征 + 因果上下文 → 逐 token 保留分数 (B,L,P)
  - TokenDistiller：按分数做软蒸馏（高分保留 + 低分加权池化为摘要），输出帧级特征 (B,L,d)
  - 因果上下文：支持两种模式：
    (a) EMA 模式（默认）：用帧级特征的因果 EMA 近似 c_{1:t-1}
    (b) SSM 模式：用 EACS 提交状态 h_pi 的投影作为真实因果上下文（设计文档原始要求）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CGU(nn.Module):
    """因果门控单元：逐 token 输出保留权重 s_{t,i} ∈ (0,1)。

    输入：
        tokens  : (B, L, P, d)  每帧 P 个空间 token
        context : (B, L, d_ctx) 因果上下文（历史摘要），可选
    输出：
        scores  : (B, L, P)  保留分数
    """

    def __init__(self, d_in: int, d_ctx: int = 0, hidden: int = 256):
        super().__init__()
        total_in = d_in + d_ctx
        self.mlp = nn.Sequential(
            nn.Linear(total_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # 初始化偏置使初始 s ≈ 0.5（不激进蒸馏）
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, tokens: Tensor, context: Tensor | None = None) -> Tensor:
        B, L, P, d = tokens.shape
        if context is not None:
            # (B,L,d_ctx) → (B,L,P,d_ctx)
            ctx = context.unsqueeze(2).expand(B, L, P, -1)
            inp = torch.cat([tokens, ctx], dim=-1)
        else:
            inp = tokens
        return torch.sigmoid(self.mlp(inp).squeeze(-1))   # (B, L, P)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CausalEMA(nn.Module):
    """因果指数移动平均：为 CGU 提供 c_{1:t-1} 上下文。

    context_t = α·context_{t-1} + (1-α)·feat_{t-1}（因果：只用 t 之前的帧）。
    第 0 帧上下文为零向量。
    """

    def __init__(self, dim: int, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.dim = dim

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (B, L, d) → context: (B, L, d)，context[:,0]=0。"""
        B, L, d = feat.shape
        ctx = torch.zeros(B, L, d, device=feat.device, dtype=feat.dtype)
        if L <= 1:
            return ctx
        # 递推：ctx_t = α·ctx_{t-1} + (1-α)·feat_{t-1}
        prev = feat[:, :-1]                               # (B, L-1, d)
        # 用 cumsum 技巧并行化（几何衰减加权）
        # 简化实现：对长序列也够用（L ≤ 512）
        alpha = self.alpha
        for t in range(1, L):
            ctx[:, t] = alpha * ctx[:, t - 1] + (1 - alpha) * prev[:, t - 1]
        return ctx


class SSMStateContext(nn.Module):
    """用 EACS 提交状态 h_pi 的投影作为 CGU 的真实因果上下文（设计文档原始要求）。

    EACS 的提交状态 h_pi[k] 代表帧 k 之前最近一次更新后的状态，即 c_{1:t-1}。
    本模块将复数状态 h_pi (B,L,H,N) 投影为实数上下文 (B,L,d_ctx)。
    因果性保证：h_pi[k] 仅包含帧 k 之前的历史信息。
    """

    def __init__(self, d_state_total: int, d_ctx: int):
        """
        d_state_total: EACS 状态展平维度（H*N 的实部+虚部 = 2*H*N，或取实部 H*N）
        d_ctx: 输出上下文维度（= CGU 的 d_ctx）
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_state_total, d_ctx),
            nn.GELU(),
            nn.Linear(d_ctx, d_ctx),
        )
        # 初始化为零，使得初始时 SSM 上下文不影响 CGU（渐进引入）
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, h_pi: Tensor) -> Tensor:
        """h_pi: (B, L, H, N) 复数提交状态 → context: (B, L, d_ctx) 实数。

        取实部展平后投影；第 0 帧 h_pi=0 → context=0（因果）。
        """
        B, L, H, N = h_pi.shape
        # 取实部 + 虚部拼接（保留完整信息）
        h_real = torch.cat([h_pi.real, h_pi.imag], dim=-1)  # (B, L, H, 2N)
        h_flat = h_real.reshape(B, L, -1)                   # (B, L, H*2N)
        return self.proj(h_flat)                             # (B, L, d_ctx)


class TokenDistiller(nn.Module):
    """Token 蒸馏：按 CGU 分数做软信息瓶颈压缩。

    策略（可微、无硬截断）：
      - 高分 token 直接保留（加权残差）
      - 低分 token 信息压缩为一个摘要向量（加权池化）
      - 最终帧级特征 = 加权保留 + 门控摘要

    输入：tokens (B,L,P,d), scores (B,L,P)
    输出：frame_feat (B,L,d), cpi_frame (B,L) 帧级 CPI = mean(scores)
    """

    def __init__(self, d: int, n_heads: int = 8):
        super().__init__()
        self.attn_pool = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.gate = nn.Sequential(nn.Linear(d * 2, d), nn.Sigmoid())

    def forward(self, tokens: Tensor, scores: Tensor) -> tuple[Tensor, Tensor]:
        B, L, P, d = tokens.shape
        # 帧级 CPI（供 EventGate / 稀疏注意力复用）
        cpi_frame = scores.mean(dim=-1)                   # (B, L)

        # 软加权保留：score 高的 token 权重大
        w = scores.unsqueeze(-1)                          # (B, L, P, 1)
        retained = (tokens * w).sum(dim=2) / (w.sum(dim=2) + 1e-6)   # (B, L, d)

        # 低分摘要：用 (1-score) 加权池化
        w_comp = (1 - scores).unsqueeze(-1)
        compressed = (tokens * w_comp).sum(dim=2) / (w_comp.sum(dim=2) + 1e-6)  # (B, L, d)

        # 门控融合
        g = self.gate(torch.cat([retained, compressed], dim=-1))   # (B, L, d)
        frame_feat = g * retained + (1 - g) * compressed

        return frame_feat, cpi_frame


class CPIBDistill(nn.Module):
    """CPIB-Distill 完整模块：CGU + 因果上下文 + Token 蒸馏。

    插入位置：spatial encoder 之后、temporal model 之前。
    输入：tokens (B,L,P,d)（来自 FeatureAdapter 的中间表示或原始 patch 特征）
    输出：frame_feat (B,L,d), cpi_frame (B,L), cpi_tokens (B,L,P)

    因果上下文模式（context_mode）：
      - "ema"（默认）：用帧级均值特征的因果 EMA，无需额外前向
      - "ssm"：用 EACS 提交状态 h_pi 的投影（设计文档原始要求，因果性更强）
        需提供 ssm_context 参数（由 model 在 EACS 前向后传入）

    用法（在 CSTSSMModel 中）：
        tokens = self.vision.get_tokens(vis_in)     # (B,L,P,d)
        frame_feat, cpi_frame, cpi_tokens = self.cpib(tokens, ssm_context=optional_ctx)
        ms = self.temporal(frame_feat, timestamps, frame_mask)
    """

    def __init__(self, d: int, hidden: int = 256, ema_alpha: float = 0.9, n_heads: int = 8,
                 context_mode: str = "ema", ssm_state_dim: int = 0):
        super().__init__()
        self.context_mode = context_mode
        self.cgu = CGU(d_in=d, d_ctx=d, hidden=hidden)
        self.ema = CausalEMA(d, alpha=ema_alpha)
        self.distiller = TokenDistiller(d, n_heads=n_heads)
        # SSM 状态上下文投影（仅 context_mode="ssm" 时使用）
        self.ssm_ctx: SSMStateContext | None = None
        if context_mode == "ssm" and ssm_state_dim > 0:
            self.ssm_ctx = SSMStateContext(ssm_state_dim, d)

    def forward(self, tokens: Tensor, ssm_context: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """
        tokens: (B, L, P, d)
        ssm_context: 可选 (B, L, d) 来自 EACS 提交状态的投影（context_mode="ssm" 时用）
        返回: (frame_feat (B,L,d), cpi_frame (B,L), cpi_tokens (B,L,P))
        """
        # 因果上下文
        if self.context_mode == "ssm" and ssm_context is not None:
            # 使用真实 SSM 状态作为因果上下文（设计文档原始要求）
            context = ssm_context                            # (B, L, d)
        else:
            # EMA 近似（默认回退）
            frame_mean = tokens.mean(dim=2)                   # (B, L, d)
            context = self.ema(frame_mean)                    # (B, L, d)

        # CGU：逐 token 保留分数
        scores = self.cgu(tokens, context)                # (B, L, P)

        # Token 蒸馏
        frame_feat, cpi_frame = self.distiller(tokens, scores)

        return frame_feat, cpi_frame, scores

    def project_ssm_states(self, h_pi: Tensor) -> Tensor:
        """将 EACS 提交状态投影为 CGU 上下文（供 model 层调用）。

        h_pi: (B, L, H, N) 复数状态（来自 EACS run_with_commits 或前向中间结果）
        返回: (B, L, d) 实数上下文
        """
        if self.ssm_ctx is not None:
            return self.ssm_ctx(h_pi)
        # 回退：取实部均值
        return h_pi.real.mean(dim=-1)  # (B, L, H) → 需进一步投影

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
