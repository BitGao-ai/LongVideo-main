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


def segment_index(frame_t: Tensor, t_query: Tensor) -> Tensor:
    """每个 t* 落在哪一段：最后一个 frame_t[k] <= t* 的 k → (B,Q)。

    索引不需要梯度，先 detach 避免 searchsorted 对带 grad_fn 的输入做无谓处理。
    单独拆出来是为了让多分支读出只算一次（各分支的 frame_t 相同）。
    """
    L = frame_t.shape[1]
    idx = torch.searchsorted(frame_t.detach().contiguous(),
                             t_query.detach().contiguous(), right=True) - 1
    return idx.clamp(0, L - 1)


def readout_from_commits(layer: EACSLayer, commits: dict, t_query: Tensor,
                         idx: Tensor | None = None) -> Tensor:
    """由逐帧提交在任意时刻 t* 求读出 → (B,Q,d)。纯函数，不构造模块。

    梯度由调用方上下文决定；推理请包 `with torch.no_grad():`。
    idx：可选的预算定段索引（见 segment_index），多分支复用时传进来省掉重复 searchsorted。

    显存：Q 与 L 同阶时 gather 出的 (B,Q,H,N) 复数与提交轨迹本身同量级。训练用的
    t_query 常常**就是帧时刻**（grounding_forward 传的是 batch["timestamps"]），此时
    定段索引恒等于 arange(L)，gather 是无意义的整份拷贝——下面对这种情况直接别名复用。
    """
    lam = commits["lam"]                       # (H,N)
    frame_t = commits["frame_t"].contiguous()  # (B,L) 单调递增
    B, L = frame_t.shape
    Q = t_query.shape[1]

    # 恒等定段快路：判定用**对象同一性**而非 torch.equal——后者要把比较结果取回 host，
    # 在 GPU 上是每次调用一次强制同步（定位训练每 step 每分支各一次）。定位路径传的
    # t_query 就是 timestamps 本身（fp32 下 .float() 返回原张量），快路照常命中；
    # 值相等但对象不同时退回 gather，只是多占一份显存，结果完全一致。
    if t_query is frame_t or t_query is commits["frame_t"]:
        h_pi, t_pi = commits["h"], commits["t"]
        u_pi, B_pi, C_pi = commits["u"], commits["B"], commits["C"]
    else:
        if idx is None:
            idx = segment_index(frame_t, t_query)

        def gather(seq: Tensor) -> Tensor:
            # seq: (B,L,*) → (B,Q,*)
            extra = seq.shape[2:]
            index = idx.view(B, Q, *([1] * len(extra))).expand(B, Q, *extra)
            return torch.gather(seq, 1, index)

        h_pi = gather(commits["h"])            # (B,Q,H,N)
        t_pi = gather(commits["t"])            # (B,Q)
        u_pi = gather(commits["u"])            # (B,Q,H)
        B_pi = gather(commits["B"])            # (B,Q,N)
        C_pi = gather(commits["C"])            # (B,Q,N)

    delta = (t_query - t_pi).clamp_min(0.0)     # (B,Q)
    # dt_max 沿用该分支自己的上界（10·τ_max），与 forward/_step 保持一致；
    # 旧 commits 没有这个键时退回全局默认 1e3。
    dt_eff = effective_dt(delta.unsqueeze(-1), commits["log_dt_scale"],
                          dt_max=commits.get("dt_max", 1e3))      # (B,Q,H)
    # 1/λ 只有 (H,N)，而离散化核要吃 (B,Q,H,N)——先算倒数再乘，省掉一次大张量复数除法
    # （本机实测复数除法是复数乘法的 8 倍成本，Q 与 L 同阶时这是整条读出路径最贵的一步）。
    dA, dB = zoh_discretize(lam, dt_eff, B_pi, inv_lam=torch.reciprocal(lam))  # (B,Q,H,N)
    x_star = dA * h_pi + dB * u_pi.unsqueeze(-1)
    y = torch.einsum("bqn,bqhn->bqh", C_pi, x_star).real          # (B,Q,H)
    y = y + layer.D * u_pi
    return layer.proj_out(y)                    # (B,Q,d)


def continuous_query(layer: EACSLayer, x: Tensor, timestamps: Tensor,
                     t_query: Tensor, cpi: Tensor | None = None,
                     use_chunk: bool | None = None) -> Tensor:
    """x:(B,L,d) timestamps:(B,L) t_query:(B,Q) → 读出 (B,Q,d)。

    cpi: 可选 (B,L) 帧级 CPI 信号，与主前向同口径地送进 EventGate（P1.5）。
    use_chunk: 透传给 run_with_commits；外层已包梯度检查点时传 False（见其文档）。
    """
    _, commits = layer.run_with_commits(x, timestamps, cpi=cpi, use_chunk=use_chunk)
    return readout_from_commits(layer, commits, t_query)


class ContinuousQuery(nn.Module):
    """封装单分支 EACSLayer 的连续查询：先记录逐帧提交，再对任意 t* 演化读出。

    薄封装，实际逻辑在模块级纯函数 `continuous_query` / `readout_from_commits`——
    模型内部（CSTSSMModel.continuous_readout）直接调那两个函数，避免在 forward 里
    反复构造 nn.Module（构造本身会把 layer 注册成子模块，是纯粹的浪费）。

    梯度由**调用方上下文**决定（本类不再自带 no_grad）：训练定位头时需要梯度经读出回传
    到 EACS/空间编码；推理请自行包 `with torch.no_grad():`。见 EACSLayer.run_with_commits。
    """

    def __init__(self, layer: EACSLayer):
        super().__init__()
        self.layer = layer

    def query(self, x: Tensor, timestamps: Tensor, t_query: Tensor,
              cpi: Tensor | None = None) -> Tensor:
        """x:(B,L,d) timestamps:(B,L) t_query:(B,Q) → 读出 y_query:(B,Q,d)。"""
        return continuous_query(self.layer, x, timestamps, t_query, cpi=cpi)

    def readout_at(self, commits: dict, t_query: Tensor) -> Tensor:
        return readout_from_commits(self.layer, commits, t_query)
