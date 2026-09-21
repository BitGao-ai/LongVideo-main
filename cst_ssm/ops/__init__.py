"""Numerical operators: ZOH discretization, spectral init, parallel scans."""
from __future__ import annotations

from .discretization import (zoh_discretize, zoh_kernels, zoh_apply_B,
                             complex_expm1, make_lambda, effective_dt)
from .spectral_init import init_branch_lambda, build_multiscale_lambdas, BranchSpec
from .scan import associative_scan_diag, sequential_scan_diag, selective_scan
from .matrix_exp import matrix_exp, matrix_exp_scale_square, zoh_discretize_dense

__all__ = [
    "zoh_discretize",
    "zoh_kernels",
    "zoh_apply_B",
    "complex_expm1",
    "make_lambda",
    "effective_dt",
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
