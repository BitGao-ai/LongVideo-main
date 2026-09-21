"""Single-query attention pooling without materializing K/V projections."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def single_query_pool(mha: nn.MultiheadAttention, query: Tensor, tokens: Tensor,
                      bias: Tensor | None = None) -> Tensor:
    """Pool tokens to one vector with mha weights; equivalent to mha(query, tokens, tokens).

    Args:
        mha: MultiheadAttention with batch_first=True and matching qkv dims.
        query: (d,) learned constant query.
        tokens: (B, L, P, d).
        bias: optional (B, L, P) additive attention bias.
    Returns:
        (B, L, d).
    """
    B, L, P, d = tokens.shape
    H = mha.num_heads
    dh = d // H
    Wi, bi = mha.in_proj_weight, mha.in_proj_bias
    Wq, Wk, Wv = Wi[:d], Wi[d:2 * d], Wi[2 * d:]
    bq = bi[:d] if bi is not None else None
    bv = bi[2 * d:] if bi is not None else None

    qh = (Wq @ query + (bq if bq is not None else 0)).view(H, dh)
    m = torch.einsum("hk,hkd->hd", qh, Wk.view(H, dh, d))
    logits = (tokens.reshape(-1, d) @ m.t()).view(B, L, P, H) * (dh ** -0.5)
    logits = logits.permute(0, 1, 3, 2)
    if bias is not None:
        logits = logits + bias.unsqueeze(2)
    alpha = logits.softmax(dim=-1)
    pooled = alpha @ tokens
    ctx = torch.bmm(pooled.permute(2, 0, 1, 3).reshape(H, B * L, d),
                    Wv.view(H, dh, d).transpose(1, 2))
    ctx = ctx.view(H, B, L, dh).permute(1, 2, 0, 3)
    if bv is not None:
        ctx = ctx + bv.view(H, dh)
    return mha.out_proj(ctx.reshape(B, L, d))
