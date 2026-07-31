"""数值算子层：全架构的数值正确性心脏。

- discretization : 连续时间 ZOH 离散化（可变 Δt、复数对角 SSM），设计方案 §3(P2)/§4.1
- spectral_init  : 多尺度 HiPPO 谱初始化（时间常数 τ=-1/Re(λ)），设计方案 §4.5
- scan           : 一阶线性递推的并行 associative scan 与序列 scan，设计方案 §3(P4)
"""
from __future__ import annotations

from .discretization import zoh_discretize, complex_expm1
from .spectral_init import init_branch_lambda, build_multiscale_lambdas, BranchSpec
from .scan import associative_scan_diag, sequential_scan_diag, selective_scan
from .matrix_exp import matrix_exp, matrix_exp_scale_square, zoh_discretize_dense

__all__ = [
    "zoh_discretize",
    "complex_expm1",
    "init_branch_lambda",
    "build_multiscale_lambdas",
    "BranchSpec",
    "associative_scan_diag",
    "sequential_scan_diag",
    "selective_scan",
    "matrix_exp",
    "matrix_exp_scale_square",
    "zoh_discretize_dense",
]
