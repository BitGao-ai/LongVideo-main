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


def require_model_config(manifest: str | None, config_path: str | None, script: str,
                         allow_default: bool = False,
                         fallback: str = "d_model=96、stand-in LLM dim=128") -> None:
    """真实 manifest 却拿不到 model 配置段时直接退出（除非显式放行）。

    三个训练脚本的 build_cfg 在 YAML 缺 model 段时会退到一份冒烟尺寸的
    CSTSSMConfig(d_model=96, llm.dim=128)。对合成数据这是合理的默认；配上**真实
    manifest** 就是一个几万步之后才暴露的错误：stage-1 会照常收敛、照常写出检查点，
    直到 stage-2 热启才因 d_model 96≠384 把 vision.*/temporal.* 整段跳过，热启完全
    白做（run.md §2 的反例，实际浪费过一次 20000 步的训练）。

    与 align_feat_dim 是同一类入口守卫，只是拦的维度不同，且**补救方式相反**：
    feat_dim 由特征文件唯一确定，能自动校正；d_model 是建模超参、没有唯一正确值，
    只能要求两个阶段用同一份 --config，所以这里只能退出，不能"自动修"。

    必须在入口拦是因为 d_model 错配在运行期没有任何征兆——loss 照常下降、检查点照常
    落盘，报错要等到下一个阶段的 load_checkpoint。只在 build_cfg 的兜底分支调用。
    """
    if not manifest or allow_default:
        return
    why = "没有传 --config" if not config_path else f"--config {config_path} 里没有 model: 段"
    raise SystemExit(
        f"[cfg] 错误: 给了真实 manifest（{manifest}）但{why}，本次将退到脚本内置的冒烟兜底"
        f"配置（{fallback}）。\n"
        f"[cfg]   这一路训得完也存得下检查点，但 d_model 与生产配置不一致，下一阶段 --load "
        f"热启时 vision.*/temporal.* 会因形状不匹配被整段丢弃——这轮训练等于白跑。\n"
        f"[cfg]   请补上配置：\n"
        f"[cfg]     python3 {script} --config configs/default.yaml ...\n"
        f"[cfg]   若确实只想用冒烟尺寸验证数据管线（不打算把检查点用于下一阶段），"
        f"显式加 --allow-default-config 放行。")


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
