"""门控 EACS cell 融合前向内核的 Python 封装（推理专用）。

有 GPU 且能编译 → 用融合内核 `csrc/eacs_cell_cuda.cu`（整段 L 一次跑完，显存 O(B·H·N)）。
否则 → 返回 None，调用方回退 eacs.py 的序列 cell。仅前向/推理；训练走 autograd 序列 cell。
"""
from __future__ import annotations

import os

import torch
from torch import Tensor

_EXT = None
_TRIED = False


def _load():
    global _EXT, _TRIED
    if _TRIED:
        return _EXT
    _TRIED = True
    if not torch.cuda.is_available():
        return None
    try:
        from torch.utils.cpp_extension import load
        here = os.path.dirname(__file__)
        src = os.path.join(here, "csrc", "eacs_cell_cuda.cu")
        _EXT = load(name="eacs_cell_cuda", sources=[src], verbose=False)
    except Exception as e:  # pragma: no cover
        print(f"[eacs_gated_scan] JIT 编译失败，回退序列 cell：{e}")
        _EXT = None
    return _EXT


def gated_kernel_available() -> bool:
    return _load() is not None


@torch.no_grad()
def eacs_cell_forward(lam: Tensor, log_dt_scale: Tensor, u: Tensor, Bc: Tensor,
                      Cc: Tensor, t: Tensor, D: Tensor, mean: Tensor, var: Tensor,
                      eps: float, eta: float, var_eps: float = 1e-5,
                      dt_init: float = 1.0, dt_min: float = 1e-3, dt_max: float = 1e3):
    """融合前向。返回 (ys[B,L,H], gates[B,L], resid[B,L])。张量需在 CUDA、dtype 对齐。"""
    ext = _load()
    if ext is None:
        raise RuntimeError("门控 EACS 融合内核不可用（无 GPU 或编译失败）。")
    return ext.eacs_cell_fwd(
        lam.to(torch.complex64), log_dt_scale.float(), u.float(),
        Bc.to(torch.complex64), Cc.to(torch.complex64), t.float(),
        D.float(), mean.float(), var.float(),
        float(eps), float(eta), float(var_eps), float(dt_init),
        float(dt_min), float(dt_max))
