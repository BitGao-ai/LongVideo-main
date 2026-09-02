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
    """LoRA 包装：y = W0·x + (α/r)·(B·A)·x。

    **A/B 恒为 float32，与被包的基座 dtype 无关**，这是有意的：AdamW 直接在 bf16 参数上
    更新会让小步长在末位下溢，fp32 的低秩增量等价于给这部分参数配了 master weights。
    代价是接 `--dtype bfloat16` 的底座时 x 是 bf16 而 A 是 fp32——训练侧有 autocast 兜着，
    但评测/推理脚本全都不带 autocast，那里会直接抛
    `RuntimeError: expected m1 and m2 to have the same dtype`。

    所以 forward 里做一次 dtype 对齐，且对齐的是**权重侧**：A 是 (r,in)、B 是 (out,r)，
    r=64/in=2560 时只有十几万元素，而 x 是 (B·T,in)（B=2/T=8704 时 4400 万），
    转权重比转激活便宜两个数量级。dtype 本来就一致时两次判断都短路，
    **全 fp32 的既有路径逐位不变**（见 tests/regression_dtype_paths.py）。
    """

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


# 只针对注意力投影。proj_u / proj_out / fusion 是 EACS 的核心投影，一旦被 LoRA 包住，
# mark_only_lora_trainable 就会把它们的基座权重冻上、只留 r 秩的旁路——本项目要训的恰恰
# 是这些时序参数，等于把主干冻死。LLM 段用 apply_lora(model.llm, …) 单独注入。
_DEFAULT_TARGETS = frozenset({"qkv", "q", "kv"})


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
