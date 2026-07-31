"""一阶线性递推的扫描算子：h_k = a_k ⊙ h_{k-1} + b_k（逐元素、复数、可微）。

对角 SSM 的状态演化按维度独立，是一阶线性递推 → 可用 associative scan 以 O(L) 工作量、
O(log L) 并行深度求解（设计方案 §3(P4)，对应 Mamba-2 SSD 的并行扫描）。

- associative_scan_diag : Hillis-Steele 并行前缀扫描（无门控的连续 SSM / 基线 / stage-1 用）
- sequential_scan_diag  : 朴素序列扫描（正确性对拍 + 分块梯度检查点用）

注：事件门控的稀疏更新破坏 associativity（门控依赖状态），其序列 cell 在 modules/eacs.py。
生产可将本文件替换为 Mamba-2 CUDA 内核（可变 Δt 版），接口保持一致。
"""
from __future__ import annotations

import torch
from torch import Tensor


def _shift_right(x: Tensor, d: int, fill: float) -> Tensor:
    """沿 dim=1（L 维）右移 d 步，左侧空位用 fill 填充（作为 monoid 单位元）。"""
    if d == 0:
        return x
    pad = torch.full_like(x[:, :d], fill)
    return torch.cat([pad, x[:, :-d]], dim=1)


def associative_scan_diag(a: Tensor, b: Tensor, h_init: Tensor | None = None) -> Tensor:
    """并行 associative scan，返回状态序列 h（形状同 b）。

    递推：h_k = a_k ⊙ h_{k-1} + b_k，h_{-1}=h_init（缺省 0）。
    a, b : (B, L, H, N) 复数。monoid：(a_l,b_l)∘(a_r,b_r)=(a_l a_r, a_r b_l + b_r)，单位元 (1,0)。
    """
    A = a
    Bp = b
    L = a.shape[1]
    d = 1
    while d < L:
        A_sh = _shift_right(A, d, 1.0)   # 单位元 a=1
        B_sh = _shift_right(Bp, d, 0.0)  # 单位元 b=0
        Bp = A * B_sh + Bp               # new_b = a_r·b_l + b_r
        A = A_sh * A                     # new_a = a_l·a_r
        d *= 2
    if h_init is not None:
        Bp = Bp + A * h_init.unsqueeze(1)
    return Bp


def sequential_scan_diag(a: Tensor, b: Tensor, h_init: Tensor | None = None) -> Tensor:
    """朴素序列扫描（O(L) 步 Python 循环）。用于对拍与分块检查点。"""
    B, L, H, N = a.shape
    h = torch.zeros(B, H, N, dtype=b.dtype, device=b.device) if h_init is None else h_init
    outs = []
    for k in range(L):
        h = a[:, k] * h + b[:, k]
        outs.append(h)
    return torch.stack(outs, dim=1)


def selective_scan(a: Tensor, b: Tensor, h_init: Tensor | None = None,
                   backend: str = "auto") -> Tensor:
    """统一变步长扫描入口：CUDA 内核（可用且在 GPU）否则纯 PyTorch associative scan。

    backend: "auto"（默认，自动选）| "cuda"（强制，不可用则报错）| "pytorch"（强制纯 PyTorch）。
    CUDA 路径不支持 h_init（内核初值为 0）；给了 h_init 自动走 PyTorch。
    """
    if backend != "pytorch" and a.is_cuda and h_init is None:
        try:
            from .scan_cuda import cuda_available, selective_scan_cuda
            if cuda_available():
                return selective_scan_cuda(a, b)
        except Exception:
            pass
    if backend == "cuda":
        raise RuntimeError("CUDA 变步长扫描不可用（需 a 在 GPU、扩展编译成功、且 h_init 为空）。")
    return associative_scan_diag(a, b, h_init)
