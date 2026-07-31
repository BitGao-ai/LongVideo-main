"""多尺度 HiPPO 谱初始化（设计方案 §4.5，降格为初始化细节，但仍需正确实现）。

时间常数定理（教科书一阶线性系统）：对角模 λ 的有效记忆时长 τ = -1/Re(λ)。
据此把短/中/长三分支的特征值实部按目标记忆时长反推、分频段初始化：
    Re(λ) ∈ [-1/τ_min, -1/τ_max]   （τ 越大记忆越长 → |Re(λ)| 越小）
虚部按 S4D-Lin（HiPPO 对角近似）铺开，建模周期性结构。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class BranchSpec:
    """一个时序分支的谱配置。"""
    name: str
    n_state: int          # 状态维 N
    tau_min: float        # 目标记忆时长下界（秒）
    tau_max: float        # 目标记忆时长上界（秒）
    imag_scale: float = 1.0  # 虚部（频率）整体缩放


# 设计方案默认三尺度（0.1–1s / 10–60s / 10–60min），状态维比例 1:2:4
DEFAULT_BRANCHES: tuple[BranchSpec, ...] = (
    BranchSpec("short", n_state=16, tau_min=0.1, tau_max=1.0, imag_scale=1.0),
    BranchSpec("mid", n_state=32, tau_min=10.0, tau_max=60.0, imag_scale=0.3),
    BranchSpec("long", n_state=64, tau_min=600.0, tau_max=3600.0, imag_scale=0.1),
)


def init_branch_lambda(spec: BranchSpec, n_channel: int,
                       generator: torch.Generator | None = None
                       ) -> tuple[Tensor, Tensor]:
    """初始化单个分支的 λ 参数分量。

    返回 (a_log_neg_real, a_imag)，形状均为 (n_channel, n_state)，供 make_lambda 组装：
        Re(λ) = -exp(a_log_neg_real) = -1/τ ，τ 在 [τ_min, τ_max] 内 log-uniform
        Im(λ) = S4D-Lin 频率 π·n （×imag_scale）
    """
    N = spec.n_state
    H = n_channel
    # 实部：τ log-uniform ∈ [τ_min, τ_max] → 记忆时长常数 m=1/τ → a=log(m)=-log(τ)
    log_tau = torch.empty(H, N)
    if generator is not None:
        log_tau.uniform_(math.log(spec.tau_min), math.log(spec.tau_max), generator=generator)
    else:
        log_tau.uniform_(math.log(spec.tau_min), math.log(spec.tau_max))
    a_log_neg_real = -log_tau  # a = log(1/τ) = -log(τ)

    # 虚部：S4D-Lin，π·n，跨通道相同（也可加轻微抖动）
    n_idx = torch.arange(N, dtype=torch.float32)
    imag = math.pi * n_idx * spec.imag_scale  # (N,)
    a_imag = imag.unsqueeze(0).expand(H, N).contiguous()
    return a_log_neg_real, a_imag


def build_multiscale_lambdas(branches: tuple[BranchSpec, ...] | list[BranchSpec],
                             n_channel: int,
                             generator: torch.Generator | None = None
                             ) -> list[tuple[Tensor, Tensor]]:
    """批量初始化所有分支，返回 [(a_log_neg_real, a_imag), ...]。"""
    return [init_branch_lambda(spec, n_channel, generator) for spec in branches]


def target_real_band(spec: BranchSpec) -> tuple[float, float]:
    """该分支特征值实部的目标频段 [-1/τ_min, -1/τ_max]，供谱正则损失约束。"""
    return (-1.0 / spec.tau_min, -1.0 / spec.tau_max)
