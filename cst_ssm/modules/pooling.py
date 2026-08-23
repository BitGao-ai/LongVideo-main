"""单 query 注意力池化：把 P 个 token 压成一个向量，且不物化 K/V。

TokenDistiller（CPIB 的低分摘要）与 FeatureAdapter（特征模式的帧级池化）都是
"一个**学习出来的常量** query 去 attend P 个 token"，两处原本都直接调
nn.MultiheadAttention。那样每帧要付两份 (P, d) → (P, d) 的 K/V 投影，
并把 (B,L,P,d) 的 K 和 V 一起留给反传——P=64、d=384、L=128 时就是 50 MB 白留。

本模块提供等价但省显存的实现，**参数与 state_dict 完全不变**（仍是那个
MultiheadAttention 的 in_proj_weight / out_proj），只是换了计算顺序。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def single_query_pool(mha: nn.MultiheadAttention, query: Tensor, tokens: Tensor,
                      bias: Tensor | None = None) -> Tensor:
    """用 mha 的参数做单 query 注意力池化，等价于 mha(query, tokens, tokens, attn_mask=bias)。

    参数
        mha    : nn.MultiheadAttention（batch_first=True，qkv 同维）
        query  : (d,) 学习到的常量 query（与 token 内容无关，这是本实现成立的前提）
        tokens : (B, L, P, d)
        bias   : 可选 (B, L, P) 加性注意力偏置（会广播到所有头）
    返回
        (B, L, d)

    两处等价变换：
      ① query 是常量 ⇒ 每个头的 logits
             (W_q^h q)·(W_k^h x_i) = x_i · [(W_k^h)^T (W_q^h q)] ≜ x_i · m_h
         m_h 是可预先算好的 d 维向量。于是 logits 用一次 (B·L·P, d)×(d, H) 的大 GEMM 得到，
         代价 O(P·d·H) 而非 K 投影的 O(P·d²)（d=384/H=8 时少算 48 倍），且**不物化 K**。
         偏置项 (W_q q + b_q)·b_k 对所有 i 相同，softmax 平移不变，直接省掉。
      ② V 投影是线性的，可挪到加权求和之后：
             Σ_i α_i (W_v^h x_i + b_v^h) = W_v^h (Σ_i α_i x_i) + b_v^h
         先在 token 维池化成每头一个 d 维向量再投影，代价从 O(P·d²) 降到 O(d²)，
         同样**不物化 V**。
    """
    B, L, P, d = tokens.shape
    H = mha.num_heads
    dh = d // H
    Wi, bi = mha.in_proj_weight, mha.in_proj_bias
    Wq, Wk, Wv = Wi[:d], Wi[d:2 * d], Wi[2 * d:]
    bq = bi[:d] if bi is not None else None
    bv = bi[2 * d:] if bi is not None else None

    qh = (Wq @ query + (bq if bq is not None else 0)).view(H, dh)      # (H, dh)
    m = torch.einsum("hk,hkd->hd", qh, Wk.view(H, dh, d))              # (H, d)
    # 写成大 GEMM 而不是逐头 einsum：einsum 在 (B,L) 上会退化成大量小矩阵乘，实测更慢。
    logits = (tokens.reshape(-1, d) @ m.t()).view(B, L, P, H) * (dh ** -0.5)
    logits = logits.permute(0, 1, 3, 2)                                # (B,L,H,P)
    if bias is not None:
        logits = logits + bias.unsqueeze(2)
    alpha = logits.softmax(dim=-1)                                     # (B,L,H,P)
    pooled = alpha @ tokens                                            # (B,L,H,P)@(B,L,P,d)
    ctx = torch.bmm(pooled.permute(2, 0, 1, 3).reshape(H, B * L, d),
                    Wv.view(H, dh, d).transpose(1, 2))                 # (H, B·L, dh)
    ctx = ctx.view(H, B, L, dh).permute(1, 2, 0, 3)                    # (B,L,H,dh)
    if bv is not None:
        ctx = ctx + bv.view(H, dh)
    return mha.out_proj(ctx.reshape(B, L, d))
