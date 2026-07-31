"""极简 LoRA（不依赖 peft）：冻结基座权重，只训练低秩增量 A,B → 大幅省显存/存储（需求 2）。

y = W0·x + (α/r)·(B·A)·x ，A 高斯小初始、B 零初始（初始等价基座，稳定）。
apply_lora 就地把命中的 nn.Linear 替换为 LoRALinear。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                     # 冻结基座
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + (self.drop(x) @ self.A.t() @ self.B.t()) * self.scaling


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new)


_DEFAULT_TARGETS = frozenset({"qkv", "q", "kv", "proj_u", "proj_out", "fusion"})


def apply_lora(model: nn.Module, targets: frozenset = _DEFAULT_TARGETS,
               r: int = 16, alpha: int = 32, dropout: float = 0.0) -> list[str]:
    """就地注入 LoRA，返回被替换的模块名列表。"""
    to_wrap = [name for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and name.split(".")[-1] in targets]
    for name in to_wrap:
        _set_submodule(model, name, LoRALinear(model.get_submodule(name), r, alpha, dropout))
    return to_wrap


def apply_qlora(model: nn.Module, targets: frozenset = _DEFAULT_TARGETS,
                r: int = 16, alpha: int = 32, dropout: float = 0.0,
                compute_dtype=None) -> list[str]:
    """QLoRA（设计 §4.4）：bitsandbytes 可用则量化命中 Linear 基座为 4bit 再加 LoRA；
    否则回退普通 LoRA（fp 基座冻结）。返回被替换的模块名列表。"""
    try:
        import bitsandbytes as bnb
    except Exception:
        print("[qlora] bitsandbytes 不可用，回退普通 LoRA（fp 基座冻结）。")
        return apply_lora(model, targets, r, alpha, dropout)
    dtype = compute_dtype or torch.bfloat16
    to_wrap = [name for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and name.split(".")[-1] in targets]
    for name in to_wrap:
        base = model.get_submodule(name)
        q = bnb.nn.Linear4bit(base.in_features, base.out_features,
                              bias=base.bias is not None, compute_dtype=dtype)
        q.weight = bnb.nn.Params4bit(base.weight.data.clone(), requires_grad=False)
        if base.bias is not None:
            q.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)
        _set_submodule(model, name, LoRALinear(q, r, alpha, dropout))
    return to_wrap


def mark_only_lora_trainable(model: nn.Module,
                             frozen_prefixes: tuple[str, ...] = ("llm.",)) -> None:
    """冻结基座：只训 LoRA(A,B) + CST-SSM 各段(非 frozen_prefixes)参数。

    规则（按优先级）：
      1) 参数名以 .A/.B 结尾           → 训练（LoRA 低秩增量）
      2) 名内含 ".base."               → 冻结（LoRALinear 包裹的原基座，任何段内均然）
      3) 名以 frozen_prefixes 开头     → 冻结（如 "llm." 基座的非 LoRA 部分）
      4) 其余（temporal./projector./vision./pred_head 等 CST-SSM 段）→ 训练
    这样避免了子串误匹配（如 "norm"/"gate" 命中 LLM 的 layernorm/gate_proj）。
    """
    for name, p in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            keep = True
        elif ".base." in name:
            keep = False
        elif any(name.startswith(pre) for pre in frozen_prefixes):
            keep = False
        else:
            keep = True
        p.requires_grad_(keep)
