"""输入 dtype 对齐：让"特征按低精度存、按低精度传"在所有路径上都安全。

**为什么需要这个模块**：视觉特征在磁盘上是 float16（extract_features.py 明确
`astype(np.float16)`），此前 dataset 取出后立刻 `.float()` 升到 float32。可训练路径的第一个
算子是 autocast(bf16) 下的 `nn.Linear`——它无论如何都会把输入降到 bf16，所以那份 float32
从生成到被丢弃全程没有任何消费者，只是让**每个环节都翻倍**：worker RSS、pinned buffer、
H2D 带宽、GPU 上常驻的 batch。实测 B=1/L=256/P=9/d=2560 下它占整步留存的 21%，
外推 B=2/L=8192 约 755 MB。

数值上取消这次升精度是**零代价**的：float16→float32 无损，所以
`float32→bf16` 与 `float16→bf16` 的舍入结果逐位相同（见 tests/regression_dtype_paths.py）。

但**不带 autocast 的路径**（评测 / 推理脚本全都不带）不能直接吃 float16 输入：
`F.linear` 要求输入与权重 dtype 严格一致，否则抛
`RuntimeError: expected m1 and m2 to have the same dtype`。所以低精度存储必须配一个
入口处的对齐：**autocast 开着就什么都不做**（交给 autocast，且不能自己先升精度，
否则刚省下的显存又原样占回去）；**autocast 没开就升到参数精度**（结果与旧行为逐位相同）。

本模块**放在顶层包下、且不 import 任何 cst_ssm 内部模块**。位置是刻意的：
`cst_ssm/utils/__init__.py` 会经 config.py 拉起 `models` → `modules`，所以
`modules/spatial_encoder.py` 一旦 `from ..utils.xxx import` 就会撞上
"partially initialized module" 的循环导入；而 `cst_ssm/__init__.py` 只有一个 `__version__`，
从任何层 `from ..dtypes import` 都不可能成环。
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["autocast_dtype", "autocast_active", "align_to_param", "resolve_feat_dtype"]

_HALF = (torch.float16, torch.bfloat16)


def autocast_dtype(device_type: str) -> torch.dtype | None:
    """当前是否处于该设备的 autocast 区域；是则返回 autocast dtype，否则 None。

    跨 torch 版本做兼容：2.4+ 是 `torch.is_autocast_enabled(device_type)`，
    更早的版本只有分设备的 `is_autocast_enabled()` / `is_autocast_cpu_enabled()`。
    任何一条都取不到时**按"没开"处理**——那只会让下面多做一次升精度（回到旧行为），
    是安全的一侧，而不是让前向直接崩。
    """
    try:                                        # torch >= 2.4
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
        return None
    except (TypeError, AttributeError, RuntimeError):
        pass
    try:                                        # 旧签名
        if device_type == "cuda":
            if torch.is_autocast_enabled():
                return torch.get_autocast_gpu_dtype()
        elif device_type == "cpu":
            if torch.is_autocast_cpu_enabled():
                return torch.get_autocast_cpu_dtype()
    except (TypeError, AttributeError, RuntimeError):
        return None
    return None


def autocast_active(device_type: str) -> bool:
    return autocast_dtype(device_type) is not None


def align_to_param(x: Tensor, param: Tensor) -> Tensor:
    """把输入张量对齐到参数 dtype，**但 autocast 区域内原样返回**。

    三种情形：
      · dtype 已相同                → 原样返回（绝大多数调用，零开销）
      · autocast 开着               → 原样返回，交给 autocast 统一降到 bf16
                                      （自己先升 fp32 会白白物化一份 B·L·P·d 的大张量）
      · autocast 没开且 dtype 不同  → 升到参数 dtype，结果与"数据侧就存 fp32"逐位相同
    非浮点输入（不该发生）直接原样返回，让下游报它自己的错，不在这里改语义。
    """
    if x.dtype == param.dtype or not x.is_floating_point():
        return x
    if autocast_active(x.device.type):
        return x
    return x.to(param.dtype)


def resolve_feat_dtype(name: str | None):
    """把 LoaderConfig/DataConfig 的 feat_dtype 字符串解析成 torch dtype 或 None。

    None（"auto"）表示**沿用特征文件自身的 dtype**（fp16 存就 fp16 用），这是默认；
    显式给 "float32" 可恢复旧的"取出即升精度"行为。
    """
    if not name or name == "auto":
        return None
    table = {"float16": torch.float16, "fp16": torch.float16,
             "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
             "float32": torch.float32, "fp32": torch.float32}
    if name not in table:
        raise ValueError(
            f"feat_dtype={name!r} 不认识；可选 auto(默认，沿用特征文件 dtype) / "
            f"float16 / bfloat16 / float32")
    return table[name]
