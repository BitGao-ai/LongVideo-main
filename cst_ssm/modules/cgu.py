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

from .pooling import single_query_pool


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
        self.d_in = d_in
        self.d_ctx = d_ctx
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
            # 等价于 mlp[0](cat([tokens, ctx_expanded]))，但**不物化** (B,L,P,d+d_ctx)：
            #   Linear(W)([x ; c]) == F.linear(x, W[:, :d]) + F.linear(c, W[:, d:])
            # ctx 只有 (B,L,d_ctx)，先算成 (B,L,hidden) 再按 P 广播即可。
            # 参数结构与 state_dict 完全不变（仍是那一个 nn.Linear），只是换了算法。
            W, bias = self.mlp[0].weight, self.mlp[0].bias
            h = F.linear(tokens, W[:, :d], bias) + F.linear(context, W[:, d:]).unsqueeze(2)
        else:
            h = self.mlp[0](tokens)
        h = self.mlp[1](h)                                # GELU
        return torch.sigmoid(self.mlp[2](h).squeeze(-1))  # (B, L, P)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CausalEMA(nn.Module):
    """因果指数移动平均：为 CGU 提供 c_{1:t-1} 上下文。

    context_t = α·context_{t-1} + (1-α)·feat_{t-1}（因果：只用 t 之前的帧）。
    第 0 帧上下文为零向量。

    数值稳定性：一阶递推等价于幺半群 (g, b) ∘ (g', b') = (g·g', g'·b + b')
    上的独占前缀扫描（Blelloch 并行扫描，O(log L) 层、每层向量化）。
    所有中间量幅值不超过 max‖feat‖（α∈[0,1) 时乘积只减不增），任意 L 与 α
    下均无上溢/NaN。旧版闭式 α^{-j} 前缀和在 α=0.9、L≈840 即溢出 fp32，已废弃。
    """

    def __init__(self, dim: int, alpha: float = 0.9):
        super().__init__()
        self.alpha = alpha
        self.dim = dim

    @staticmethod
    def _exclusive_scan(g: Tensor, b: Tensor) -> Tensor:
        """幺半群 (g, b) 上的独占前缀扫描。

        输入 g, b: (n, ...)，返回 out[t] = e_0 ∘ e_1 ∘ ... ∘ e_{t-1}（out[0]=单位元），
        其中 e_t = (g[t], b[t])，单位元 = (1, 0)。
        经典 Blelloch up/down-sweep（树形索引：段缩减恒存于段末位 k+2^d-1），
        n 补齐到 2 的幂、尾部补单位元（不改变前缀值）。幺半群非交换，顺序严格。
        """
        n = g.shape[0]
        npow = 1 << (n - 1).bit_length()              # ≥ n 的最小 2 的幂（n≥1）
        if npow > n:
            pad_g = (npow - n, *g.shape[1:])
            pad_b = (npow - n, *b.shape[1:])
            g = torch.cat([g, g.new_ones(pad_g)], dim=0)
            b = torch.cat([b, b.new_zeros(pad_b)], dim=0)
        tmp_g, tmp_b = g.clone(), b.clone()
        # up-sweep：相邻段归并（左∘右），缩减写入段末位，逐层折半，总缩减落在 tmp[-1]
        step = 1
        while step < npow:
            lo = torch.arange(step - 1, npow - step, step * 2, device=g.device)
            hi = lo + step
            g_lo, b_lo = tmp_g[lo], tmp_b[lo]
            g_hi, b_hi = tmp_g[hi], tmp_b[hi]
            tmp_g[hi] = g_lo * g_hi
            tmp_b[hi] = b_hi + g_hi * b_lo
            step *= 2
        # down-sweep：根置单位元，逐层展开——左子=父前缀，右子=父前缀∘左段缩减
        tmp_g[-1] = torch.ones_like(tmp_g[-1])
        tmp_b[-1] = torch.zeros_like(tmp_b[-1])
        step = npow // 2
        while step >= 1:
            lo = torch.arange(step - 1, npow - step, step * 2, device=g.device)
            hi = lo + step
            p_g, p_b = tmp_g[hi], tmp_b[hi]             # 父节点前缀
            t_g, t_b = tmp_g[lo], tmp_b[lo]             # 左段缩减（先读后写）
            tmp_g[lo], tmp_b[lo] = p_g, p_b
            tmp_g[hi] = p_g * t_g
            tmp_b[hi] = t_g * p_b + t_b
            step //= 2
        return tmp_b[:n]

    def forward(self, feat: Tensor) -> Tensor:
        """feat: (B, L, d) → context: (B, L, d)，context[:,0]=0。"""
        B, L, d = feat.shape
        if L <= 1:
            return torch.zeros(B, L, d, device=feat.device, dtype=feat.dtype)
        # ctx_t = (1-α)·Σ_{j<t} α^{t-1-j}·feat_j = 幺半群元素 e_j = (α, (1-α)·feat_j)
        # 在索引 t 处的独占前缀（e_0∘...∘e_{t-1}）的 b 分量；t=0 即单位元 0。
        # 低精度（bf16/fp16）升到 fp32 扫描避免累加误差；fp32/fp64 保持原精度
        #（不降精度，保证 double 下与 autograd/数值差分严格一致），输出转回输入 dtype。
        alpha = float(self.alpha)
        scan_dtype = (torch.float32 if feat.dtype in (torch.bfloat16, torch.float16)
                      else feat.dtype)
        x = feat.to(scan_dtype)                         # (B, L, d)
        g = x.new_full((L,), alpha)
        b = (1.0 - alpha) * x
        # 扫描轴 = 时间维：置换成 (L, B, d) 扫描后换回，与递推逐位等价
        s = self._exclusive_scan(
            g.unsqueeze(1).unsqueeze(2), b.permute(1, 0, 2))   # (L, B, d)
        ctx = s.permute(1, 0, 2).to(feat.dtype)
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
      - 高分 token 直接保留（加权均值）
      - 低分 token 信息压缩为一个摘要向量：用**可学习注意力池化**（query + MultiheadAttention），
        以 log(1-s) 作为加性注意力偏置——分数越低越参与摘要（真正用上 attn_pool/query 参数）
      - 最终帧级特征 = 门控融合(保留, 摘要)

    输入：tokens (B,L,P,d), scores (B,L,P)
    输出：frame_feat (B,L,d), cpi_frame (B,L) 帧级 CPI = mean(scores)
    """

    def __init__(self, d: int, n_heads: int = 8):
        super().__init__()
        self.n_heads = n_heads
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

        # 低分摘要：单 query 注意力池化，加性偏置 log(1-s)（低分 token 偏置≈0、高分→-∞）。
        # 走 pooling.single_query_pool：与 nn.MultiheadAttention 数学等价，但不物化
        # (B,L,P,d) 的 K/V（P=64、d=384、L=128 时省 50MB 且不留给反传）。
        # 参数仍是 self.attn_pool 那一套，state_dict 不变。见 pooling.py 的推导。
        compressed = single_query_pool(
            self.attn_pool, self.query.reshape(d), tokens,
            bias=torch.log1p(-scores).clamp_min(-14.0))              # (B,L,d)

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
