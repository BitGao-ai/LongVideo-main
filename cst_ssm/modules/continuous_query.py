"""任意时刻连续查询算子（设计方案 §4.3，杀手锏 C3 的实现）。

连续时间形式支持在**帧与帧之间**任意 t* 求状态/读出：
    x(t*) = Ā(t*−t_π) x_π + B̄(t*−t_π; B_π) u_π,   y(t*) = Re(C_π · x(t*))
从而突破一切离散模型的"帧栅格时间定位分辨率下界 δ/4"（命题 2）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..ops.discretization import zoh_discretize, effective_dt
from .eacs import EACSLayer


class ContinuousQuery(nn.Module):
    """封装单分支 EACSLayer 的连续查询：先记录逐帧提交，再对任意 t* 演化读出。"""

    def __init__(self, layer: EACSLayer):
        super().__init__()
        self.layer = layer

    @torch.no_grad()
    def query(self, x: Tensor, timestamps: Tensor, t_query: Tensor) -> Tensor:
        """x:(B,L,d) timestamps:(B,L) t_query:(B,Q) → 读出 y_query:(B,Q,d)。"""
        _, commits = self.layer.run_with_commits(x, timestamps)
        return self.readout_at(commits, t_query)

    @torch.no_grad()
    def readout_at(self, commits: dict, t_query: Tensor) -> Tensor:
        lam = commits["lam"]                       # (H,N)
        frame_t = commits["frame_t"].contiguous()  # (B,L) 单调递增
        B, L = frame_t.shape
        Q = t_query.shape[1]

        # 定位每个 t* 所在段：最后一个 frame_t[k] <= t* 的 k
        idx = torch.searchsorted(frame_t, t_query.contiguous(), right=True) - 1
        idx = idx.clamp(0, L - 1)                   # (B,Q)

        def gather(seq: Tensor) -> Tensor:
            # seq: (B,L,*) → (B,Q,*)
            extra = seq.shape[2:]
            index = idx.view(B, Q, *([1] * len(extra))).expand(B, Q, *extra)
            return torch.gather(seq, 1, index)

        h_pi = gather(commits["h"])                # (B,Q,H,N)
        t_pi = gather(commits["t"])                # (B,Q)
        u_pi = gather(commits["u"])                # (B,Q,H)
        B_pi = gather(commits["B"])                # (B,Q,N)
        C_pi = gather(commits["C"])                # (B,Q,N)

        delta = (t_query - t_pi).clamp_min(0.0)     # (B,Q)
        dt_eff = effective_dt(delta.unsqueeze(-1), commits["log_dt_scale"])  # (B,Q,H)
        dA, dB = zoh_discretize(lam, dt_eff, B_pi)  # (B,Q,H,N)
        x_star = dA * h_pi + dB * u_pi.unsqueeze(-1)
        y = torch.einsum("bqn,bqhn->bqh", C_pi, x_star).real          # (B,Q,H)
        y = y + self.layer.D * u_pi
        return self.layer.proj_out(y)               # (B,Q,d)
