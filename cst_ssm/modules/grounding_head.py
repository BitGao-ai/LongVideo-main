"""可训练定位打分头（把 infer_grounding 的随机占位方向换成真·训练头）。

连续查询给出任意时刻的视觉读出 y(t*)∈R^d；本头把**文本查询**编码成向量，与 y(t*) 做条件打分
score(t*)=f(query, y(t*))，从而输出 query 相关的时间相关度曲线。配 BCE-over-span 损失可端到端训练，
参数随模型 state_dict 保存/加载 → 推理时 `--ckpt` 直接接上真实权重（穿透 δ/4 的亚帧边界仍由
infer_grounding.boundaries_from_scores 从该曲线插值得到）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def masked_mean(x: Tensor, mask: Tensor | None) -> Tensor:
    """(B,T,d) 按 (B,T) 有效位求均值；mask 缺省或整行全假时退化为普通均值。"""
    if mask is None:
        return x.mean(1)
    m = mask.to(x.dtype).unsqueeze(-1)                 # (B,T,1)
    denom = m.sum(1).clamp_min(1.0)
    return (x * m).sum(1) / denom


class GroundingHead(nn.Module):
    """query 条件的时间相关度打分器。

    encode_query: 文本 token 嵌入 (B,T,d_txt) + 掩码 → 查询向量 (B,d_model)
    forward     : 查询向量 (B,d_model) + 读出 (B,Q,d_model) → 相关度 logits (B,Q)
    打分用 [y, q, y⊙q] 拼接过 MLP（含双线性交互项），比纯点积更表达。
    """

    def __init__(self, d_model: int, d_txt: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_model
        self.q_proj = nn.Sequential(
            nn.Linear(d_txt, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.score = nn.Sequential(
            nn.Linear(3 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def encode_query(self, text_emb: Tensor, text_mask: Tensor | None = None) -> Tensor:
        return self.q_proj(masked_mean(text_emb, text_mask))       # (B,d_model)

    def forward(self, query_vec: Tensor, readouts: Tensor) -> Tensor:
        Q = readouts.shape[1]
        q = query_vec.unsqueeze(1).expand(-1, Q, -1)               # (B,Q,d_model)
        feat = torch.cat([readouts, q, readouts * q], dim=-1)      # (B,Q,3d)
        return self.score(feat).squeeze(-1)                        # (B,Q) logits


def span_targets(t_query: Tensor, gt_start: Tensor, gt_end: Tensor) -> Tensor:
    """(B,Q) ∈ {0,1}：查询时刻落在 [gt_start,gt_end] 内为 1。"""
    return ((t_query >= gt_start.unsqueeze(1)) & (t_query <= gt_end.unsqueeze(1))).float()


def grounding_loss(logits: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """BCE-with-logits over 查询时刻；mask 标记有效查询点（如有效帧）。"""
    if mask is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    m = mask.to(logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (loss * m).sum() / m.sum().clamp_min(1.0)
