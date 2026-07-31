"""事件驱动门控（预测编码残差门控），设计方案 §4.1(2)(3)。

预测编码：先用连续状态自由演化预测当前观测，若预测已解释观测（归一化残差 ≤ ε）则跳过状态更新。
- 训练：软门控 sigmoid((r-ε)/T) + 可选直通估计（STE），阈值 ε 可学习、带上下界。
- 推理：硬门控 1[r>ε]，跳过分支零开销快进。
- 观测标准化：用训练集运行统计消除亮度/风格幅值扰动，使阈值跨场景通用。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RunningStandardizer(nn.Module):
    """逐特征运行均值/方差标准化（类 BatchNorm 统计，但不缩放仿射）。

    训练时用 batch 统计更新缓冲；推理时用缓冲。对残差计算前的观测做归一化。
    """

    def __init__(self, dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., dim)
        if self.training:
            flat = x.reshape(-1, x.shape[-1])
            mean = flat.mean(0)
            var = flat.var(0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean.detach())
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * var.detach())
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) / torch.sqrt(var + self.eps)

    def apply_stats(self, x: Tensor) -> Tensor:
        """用当前缓冲统计标准化，但**不更新**（用于预测项，避免污染观测统计）。"""
        return (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)


class EventGate(nn.Module):
    """事件门控：由归一化预测残差决定是否更新状态。

    参数：
        eps_min/eps_max : 可学习阈值 ε 的上下界（保证 ε∈(eps_min,eps_max)，避免退化）
        temperature     : 软门控温度 T（外部退火，见 train）；越小越接近硬门控
        use_ste         : True 时前向硬门控、反向用软门控梯度（直通估计）
    残差沿最后一维（特征/通道）取 L2 范数 → 每(样本,步)一个标量残差（分支级事件决策）。
    """

    def __init__(self, eps_min: float = 0.01, eps_max: float = 0.9,
                 init_eps: float = 0.1, temperature: float = 0.1,
                 use_ste: bool = True, norm_eta: float = 1e-6,
                 gate_kind: str = "event", random_rate: float = 0.5):
        super().__init__()
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.norm_eta = norm_eta
        self.use_ste = use_ste
        self.gate_kind = gate_kind        # "event"(默认) | "random" | "always"（消融）
        self.random_rate = random_rate    # random 门控的目标更新率
        # 反解 raw 使 sigmoid(raw) 映射到 init_eps
        frac = (init_eps - eps_min) / max(eps_max - eps_min, 1e-6)
        frac = min(max(frac, 1e-4), 1 - 1e-4)
        raw = torch.logit(torch.tensor(frac))
        self.eps_raw = nn.Parameter(raw)
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("eps_anneal", torch.tensor(1.0))   # ε 退火乘子（外部调度，默认1=无退火）

    @property
    def eps(self) -> Tensor:
        base = self.eps_min + (self.eps_max - self.eps_min) * torch.sigmoid(self.eps_raw)
        return base * self.eps_anneal

    def set_temperature(self, t: float) -> None:
        self.temperature.fill_(float(t))

    def set_eps_anneal(self, v: float) -> None:
        """ε 退火：前期 <1（ε 小→多更新学动力学），后期→1（压稀疏率）。设计 §4.4。"""
        self.eps_anneal.fill_(float(v))

    def residual(self, obs: Tensor, pred: Tensor) -> Tensor:
        """归一化残差 r = ||obs-pred|| / (||obs||+η)，沿最后一维取范数 → (…,)。"""
        num = torch.linalg.vector_norm(obs - pred, dim=-1)
        den = torch.linalg.vector_norm(obs, dim=-1) + self.norm_eta
        return num / den

    def forward(self, obs: Tensor, pred: Tensor,
                eps_override: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """返回 (gate, r)。gate∈[0,1]（软，训练）或∈{0,1}（硬，推理/STE 前向）。
        eps_override：可选逐样本阈值（鲁棒兜底用），覆盖 self.eps。"""
        r = self.residual(obs, pred)                    # (…,)
        if self.gate_kind == "always":                  # 消融：稠密（无门控）
            return torch.ones_like(r), r
        if self.gate_kind == "random":                  # 消融：随机跳更新（对照残差驱动）
            return (torch.rand_like(r) < self.random_rate).to(r.dtype), r
        eps = self.eps if eps_override is None else eps_override
        soft = torch.sigmoid((r - eps) / self.temperature.clamp_min(1e-4))
        if not self.training:
            return (r > eps).to(r.dtype), r
        if self.use_ste:
            hard = (r > eps).to(r.dtype)
            gate = hard + (soft - soft.detach())        # 前向=hard，反向=soft 梯度
            return gate, r
        return soft, r
