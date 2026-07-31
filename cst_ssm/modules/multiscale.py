"""多尺度 EACS：短/中/长三分支 + 输入依赖门控融合（设计方案 §3.2）。

三个分支按谱初始化覆盖不同记忆时长（秒级/分钟级/小时级），各自独立事件门控；
输出用输入依赖的 Softmax 门控权重动态加权融合，实现"不同内容差异化激活不同尺度"。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from ..ops.spectral_init import BranchSpec, DEFAULT_BRANCHES
from .eacs import EACSLayer, EACSOutput
from .event_gate import EventGate


@dataclass
class MultiScaleOutput:
    y: Tensor                        # (B, L, d) 融合后时序表示（已含残差）
    gates: Tensor                    # (B, n_branch, L)
    update_rate: Tensor              # 标量，整体有效更新率
    per_branch_update_rate: Tensor   # (n_branch,) 各分支更新率
    fusion_weights: Tensor           # (B, L, n_branch)
    residual: Tensor                 # (B, n_branch, L) 各分支预测残差


class MultiScaleEACS(nn.Module):
    def __init__(self, d_model: int,
                 branches: tuple[BranchSpec, ...] | list[BranchSpec] = DEFAULT_BRANCHES,
                 gate_kwargs: dict | None = None,
                 dt_init: float = 1.0, chunk_size: int = 0, fused_train: bool = False,
                 robust_guard: bool = False,
                 disc_mode: str = "continuous", use_spectral_init: bool = True):
        super().__init__()
        gate_kwargs = gate_kwargs or {}
        self.branch_specs = list(branches)
        self.branches = nn.ModuleList([
            EACSLayer(d_model, spec, gate=EventGate(**gate_kwargs),
                      dt_init=dt_init, chunk_size=chunk_size, add_residual=False,
                      fused_train=fused_train, robust_guard=robust_guard,
                      disc_mode=disc_mode, use_spectral_init=use_spectral_init)
            for spec in branches
        ])
        self.n_branch = len(self.branches)
        self.fusion = nn.Linear(d_model, self.n_branch)   # 输入依赖门控权重
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, timestamps: Tensor,
                frame_mask: Tensor | None = None) -> MultiScaleOutput:
        outs: list[EACSOutput] = [br(x, timestamps) for br in self.branches]
        deltas = torch.stack([o.y for o in outs], dim=-1)          # (B,L,d,n_branch)
        w = torch.softmax(self.fusion(x), dim=-1)                  # (B,L,n_branch)
        fused = torch.einsum("bldn,bln->bld", deltas, w)
        y = self.out_norm(x + fused)
        gates = torch.stack([o.gates for o in outs], dim=1)        # (B,n_branch,L)
        res = torch.stack([o.residual for o in outs], dim=1)
        if frame_mask is not None:
            # 只在有效帧上统计更新率（避免 padding 稀释更新率正则与效率指标）
            m = frame_mask.to(gates.dtype).unsqueeze(1)            # (B,1,L)
            denom = m.sum().clamp_min(1.0)
            update_rate = (gates * m).sum() / (denom * self.n_branch)
            pbur = (gates * m).sum(dim=(0, 2)) / denom             # (n_branch,)
        else:
            update_rate = gates.mean()
            pbur = torch.stack([o.update_rate for o in outs])
        return MultiScaleOutput(y=y, gates=gates, update_rate=update_rate,
                                per_branch_update_rate=pbur, fusion_weights=w, residual=res)

    def spectral_reg(self) -> Tensor:
        return sum(br.spectral_reg() for br in self.branches) / self.n_branch
