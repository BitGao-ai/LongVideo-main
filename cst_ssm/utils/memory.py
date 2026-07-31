"""训练期显存优化工具（需求 2）。

- 梯度检查点：对 EACS 序列扫描分块重算（chunk_size），对空间编码/LLM 层用 torch checkpoint。
- 混合精度：bf16 autocast（SSM 复数核心内部仍用 float32，见 EACSLayer）。
- 逐通道时标/门控为小参数，配合 LoRA 冻结基座，可训练参数与显存大幅下降。
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from ..modules.eacs import EACSLayer
from ..modules.event_gate import EventGate


def set_eacs_chunk(model: nn.Module, chunk_size: int) -> int:
    """给所有 EACS 层设置分块梯度检查点大小（0=关闭）。返回被设置的层数。"""
    n = 0
    for m in model.modules():
        if isinstance(m, EACSLayer):
            m.chunk_size = chunk_size
            n += 1
    return n


def set_gate_temperature(model: nn.Module, t: float) -> None:
    """设置所有事件门控温度（退火用；越小越接近硬门控）。"""
    for m in model.modules():
        if isinstance(m, EventGate):
            m.set_temperature(t)


def set_eps_anneal(model: nn.Module, v: float) -> None:
    """设置所有事件门控的 ε 退火乘子（设计 §4.4）。"""
    for m in model.modules():
        if isinstance(m, EventGate):
            m.set_eps_anneal(v)


def eps_anneal_schedule(step: int, total: int,
                        start: float = 1.0, end: float = 1.0) -> float:
    """ε 退火乘子线性调度：前期 start(<1，多更新学动力学)→后期 end(=1，压稀疏率)。
    默认 start=end=1.0 即无退火（保持既有行为）。"""
    if total <= 1:
        return end
    frac = min(max(step / total, 0.0), 1.0)
    return float(start + (end - start) * frac)


@contextmanager
def autocast_ctx(enabled: bool = True, dtype: torch.dtype = torch.bfloat16,
                 device_type: str = "cuda"):
    if enabled and (device_type == "cuda" and torch.cuda.is_available() or device_type == "cpu"):
        with torch.autocast(device_type=device_type, dtype=dtype):
            yield
    else:
        yield


def build_optimizer(model: nn.Module, lr: float = 1e-4, weight_decay: float = 0.05,
                    try_8bit: bool = True):
    """构建 AdamW；try_8bit 且 bitsandbytes 可用时用 8bit Adam 进一步省显存。"""
    params = [p for p in model.parameters() if p.requires_grad]
    if try_8bit:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except Exception:
            pass
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def gate_temperature_schedule(step: int, total: int,
                              t_start: float = 1.0, t_end: float = 0.05) -> float:
    """指数退火温度：训练前期高温(近软门控，稳)，后期低温(近硬门控，压稀疏率)。"""
    if total <= 1:
        return t_end
    frac = min(max(step / total, 0.0), 1.0)
    return float(t_start * (t_end / t_start) ** frac)
