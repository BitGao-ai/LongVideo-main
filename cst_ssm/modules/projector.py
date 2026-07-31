"""跨模态投影层（设计方案 §3.3）：把连续时序状态投影到 LLM 文本特征空间。

两层 MLP + LayerNorm，保留时间维（连续属性）。投影后视觉状态作为 LLM 交叉注意力的键/值。
"""
from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class CrossModalProjector(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or d_out
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_out)
        )
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.net(x))
