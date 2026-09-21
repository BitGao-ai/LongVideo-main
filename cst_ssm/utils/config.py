"""YAML <-> dataclass loading for scripted experiments."""
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
    """Build a CSTSSMConfig from a plain dict (e.g. parsed YAML)."""
    d = dict(d)
    if "branches" in d:
        d["branches"] = branches_from_list(d["branches"])
    if "llm" in d and isinstance(d["llm"], dict):
        d["llm"] = LLMConfig(**d["llm"])
    known = CSTSSMConfig.__dataclass_fields__.keys()
    unknown = sorted(k for k in d if k not in known)
    if unknown:
        warnings.warn(
            "model_config_from_dict ignored unknown keys: "
            + ", ".join(unknown),
            stacklevel=2,
        )
    return CSTSSMConfig(**{k: v for k, v in d.items() if k in known})


def require_model_config(manifest: str | None, config_path: str | None, script: str,
                         allow_default: bool = False,
                         fallback: str = "d_model=96, stand-in LLM dim=128") -> None:
    """Exit when a real manifest would fall back to the smoke-size default config."""
    if not manifest or allow_default:
        return
    why = "no --config given" if not config_path else f"--config {config_path} has no model: section"
    raise SystemExit(
        f"[cfg] real manifest ({manifest}) but {why}; would fall back to smoke-size "
        f"defaults ({fallback}) whose checkpoints cannot warm-start production stages.\n"
        f"[cfg]   Pass a config: python3 {script} --config configs/default.yaml ...\n"
        f"[cfg]   Or pass --allow-default-config to validate the data pipeline only.")


def _apply_section(obj, d: dict | None, section: str, skip: set[str] | None = None):
    """Overlay one YAML section onto a dataclass instance in place."""
    if not d:
        return obj
    skip = skip or set()
    known = set(getattr(obj, "__dataclass_fields__", {}))
    unknown = sorted(k for k in d if k not in known and k not in skip)
    if unknown:
        warnings.warn(
            f"config section [{section}] has unknown keys for {type(obj).__name__}: "
            + ", ".join(unknown),
            stacklevel=3,
        )
    for k, v in d.items():
        if k in known and k not in skip:
            setattr(obj, k, v)
    return obj


def train_config_from_yaml(y: dict, base=None, skip: set[str] | None = None):
    """YAML [train] section -> TrainConfig."""
    from ..train.trainer import TrainConfig
    return _apply_section(base or TrainConfig(), y.get("train"), "train", skip)


def loss_weights_from_yaml(y: dict, base=None):
    """YAML [loss] section -> LossWeights."""
    from ..train.losses import LossWeights
    return _apply_section(base or LossWeights(), y.get("loss"), "loss")


def loader_config_from_yaml(y: dict, base=None, skip: set[str] | None = None):
    """YAML [data] section -> LoaderConfig."""
    from ..data.loaders import LoaderConfig
    return _apply_section(base or LoaderConfig(), y.get("data"), "data", skip)
