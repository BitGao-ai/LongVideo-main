"""配置加载：YAML ↔ 数据类，便于脚本化实验。"""
from __future__ import annotations

from typing import Any

import yaml

from ..ops.spectral_init import BranchSpec, DEFAULT_BRANCHES
from ..models.cst_ssm_model import CSTSSMConfig
from ..modules.llm_interface import LLMConfig


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def branches_from_list(items: list[dict[str, Any]] | None) -> tuple[BranchSpec, ...]:
    if not items:
        return DEFAULT_BRANCHES
    return tuple(BranchSpec(**it) for it in items)


def model_config_from_dict(d: dict) -> CSTSSMConfig:
    """把普通字典（如 YAML 解析结果）构造成 CSTSSMConfig。"""
    d = dict(d)
    if "branches" in d:
        d["branches"] = branches_from_list(d["branches"])
    if "llm" in d and isinstance(d["llm"], dict):
        d["llm"] = LLMConfig(**d["llm"])
    known = CSTSSMConfig.__dataclass_fields__.keys()
    return CSTSSMConfig(**{k: v for k, v in d.items() if k in known})
