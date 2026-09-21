"""Minimal LoRA without peft: frozen base plus trainable low-rank updates."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class LoRALinear(nn.Module):
    """LoRA wrapper: y = W0·x + (alpha/r)·(B·A)·x with fp32 low-rank factors."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = self.base(x)
        A = self.A if self.A.dtype == x.dtype else self.A.to(x.dtype)
        B = self.B if self.B.dtype == x.dtype else self.B.to(x.dtype)
        delta = (self.drop(x) @ A.t() @ B.t()) * self.scaling
        return out + (delta if delta.dtype == out.dtype else delta.to(out.dtype))


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parts = name.split(".")
    obj = root
    for p in parts[:-1]:
        obj = getattr(obj, p)
    setattr(obj, parts[-1], new)


_DEFAULT_TARGETS = frozenset({"qkv", "q", "kv"})


def apply_lora(model: nn.Module, targets: frozenset = _DEFAULT_TARGETS,
               r: int = 16, alpha: int = 32, dropout: float = 0.0) -> list[str]:
    """Wrap matching linears in place; returns wrapped module names."""
    to_wrap = [name for name, m in model.named_modules()
               if isinstance(m, nn.Linear) and name.split(".")[-1] in targets]
    for name in to_wrap:
        _set_submodule(model, name, LoRALinear(model.get_submodule(name), r, alpha, dropout))
    return to_wrap


def apply_qlora(model: nn.Module, targets: frozenset = _DEFAULT_TARGETS,
                r: int = 16, alpha: int = 32, dropout: float = 0.0,
                compute_dtype=None) -> list[str]:
    """QLoRA: 4-bit quantized base plus LoRA; falls back to plain LoRA."""
    try:
        import bitsandbytes as bnb
    except Exception:
        print("[qlora] bitsandbytes unavailable, falling back to plain LoRA.")
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
    """Freeze bases: train LoRA factors plus non-prefixed CST-SSM segments.

    Priority: *.A/*.B train; *.base.* freeze; frozen_prefixes freeze; rest train.
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
