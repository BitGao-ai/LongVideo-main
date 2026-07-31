"""CST-SSM 安装脚本。

CPU-only 机器：`pip install -e .` 正常安装（跳过 CUDA 扩展，运行时自动回退纯 PyTorch 扫描）。
有 GPU：设 `FORCE_CUDA=1 pip install -e .` 或本机可见 CUDA 时，编译变步长扫描 CUDA 扩展。
也可完全不装扩展——`cst_ssm/ops/scan_cuda.py` 会在首次调用时 JIT 编译。
"""
from __future__ import annotations

import os

from setuptools import find_packages, setup

ext_modules = []
cmdclass = {}

_want_cuda = os.environ.get("FORCE_CUDA", "0") == "1"
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    if _want_cuda or torch.cuda.is_available():
        ext_modules = [
            CUDAExtension(name="cst_ssm_eacs_scan_cuda",
                          sources=["cst_ssm/ops/csrc/eacs_scan_cuda.cu"]),
            CUDAExtension(name="cst_ssm_eacs_cell_cuda",
                          sources=["cst_ssm/ops/csrc/eacs_cell_cuda.cu"]),
        ]
        cmdclass = {"build_ext": BuildExtension}
except Exception as e:  # torch 未装或无 CUDA 工具链
    print(f"[setup] 跳过 CUDA 扩展编译（回退纯 PyTorch）：{e}")

setup(
    name="cst_ssm",
    version="0.1.0",
    description="Event-Adaptive Continuous-Time State Space Models for hour-long video understanding",
    packages=find_packages(include=["cst_ssm", "cst_ssm.*"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.1", "safetensors>=0.4", "numpy>=1.24", "pyyaml>=6.0"],
    extras_require={
        "qwen3vl": ["transformers>=4.57"],
        "train8bit": ["bitsandbytes>=0.43"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
