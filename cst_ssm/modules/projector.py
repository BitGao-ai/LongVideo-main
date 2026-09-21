"""Cross-modal projector: temporal states to LLM embedding space."""
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
