"""配置加载：YAML ↔ 数据类，便于脚本化实验。"""
from __future__ import annotations

import warnings
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
    # 未知键必须出声：YAML 里写错字段名（typo、或 CSTSSMConfig 没有的字段）此前被静默
    # 丢弃，「改 config 就能改行为」的预期会毫无提示地落空。不 raise——调用方常把整份
    # YAML 传进来，而 YAML 里本来就有 train / loss / data 三段不属于模型配置。
    unknown = sorted(k for k in d if k not in known)
    if unknown:
        warnings.warn(
            "model_config_from_dict 忽略了 CSTSSMConfig 中不存在的键: "
            + ", ".join(unknown)
            + "（若为模型字段请检查拼写；train/loss/data 等非模型段可忽略本告警）",
            stacklevel=2,
        )
    return CSTSSMConfig(**{k: v for k, v in d.items() if k in known})


def _apply_section(obj, d: dict | None, section: str, skip: set[str] | None = None):
    """把 YAML 的一段覆盖到已构造好的 dataclass 实例上（就地 setattr）。

    对未知键告警，理由同 model_config_from_dict：静默丢弃会让「改 config 就能改行为」
    的预期落空。skip 里的键是**由命令行/运行时接管**的字段，属于有意不消费，不告警。
    """
    if not d:
        return obj
    skip = skip or set()
    known = set(getattr(obj, "__dataclass_fields__", {}))
    unknown = sorted(k for k in d if k not in known and k not in skip)
    if unknown:
        warnings.warn(
            f"配置段 [{section}] 中存在 {type(obj).__name__} 没有的键: " + ", ".join(unknown),
            stacklevel=3,
        )
    for k, v in d.items():
        if k in known and k not in skip:
            setattr(obj, k, v)
    return obj


def train_config_from_yaml(y: dict, base=None, skip: set[str] | None = None):
    """YAML 的 [train] 段 → TrainConfig。base 给了就在它上面覆盖。

    存在的理由（D11）：此前脚本只读 ["model"]，YAML 里的 train / loss / data 三段共 20 余键
    无人消费——外部复现者按注释改了 lr / grad_accum / max_steps 却毫无效果，且没有任何提示。
    """
    from ..train.trainer import TrainConfig
    return _apply_section(base or TrainConfig(), y.get("train"), "train", skip)


def loss_weights_from_yaml(y: dict, base=None):
    """YAML 的 [loss] 段 → LossWeights。"""
    from ..train.losses import LossWeights
    return _apply_section(base or LossWeights(), y.get("loss"), "loss")


def loader_config_from_yaml(y: dict, base=None, skip: set[str] | None = None):
    """YAML 的 [data] 段 → LoaderConfig。"""
    from ..data.loaders import LoaderConfig
    return _apply_section(base or LoaderConfig(), y.get("data"), "data", skip)
