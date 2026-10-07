"""Restore the same model, tokenizer and adapters used to create a checkpoint."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace

from .checkpoint import checkpoint_metadata, load_checkpoint, CRITICAL_SHAPE_PREFIXES
from .config import load_yaml, model_config_from_dict


def restoration_metadata(model) -> dict:
    from .lora import LoRALinear
    info = dict(getattr(model, "restoration_config", {}))
    if not info:
        hf = getattr(getattr(model, "llm", None), "hf", None)
        name = getattr(getattr(hf, "config", None), "_name_or_path", None)
        info = dict(backend="qwen3_vl" if hf is not None else "stand_in")
        if name:
            info.update(base_model=name, tokenizer_model=name)
    info["lora_modules"] = {
        name: dict(r=m.r, alpha=m.scaling * m.r, dropout=m.drop.p)
        for name, m in model.named_modules() if isinstance(m, LoRALinear)
    }
    hf = getattr(getattr(model, "llm", None), "hf", None)
    if hasattr(hf, "parameters"):
        first = next(hf.parameters(), None)
        if first is not None:
            info["dtype"] = str(first.dtype).removeprefix("torch.")
    return info


def resolve_model_config(ckpt=None, config=None, feat_dim=None, default=None,
                         allow_config_override=False):
    """Prefer saved effective config; reject conflicting non-derived YAML fields."""
    from ..models import CSTSSMConfig
    saved = checkpoint_metadata(ckpt).get("model_config") if ckpt else None
    path = config
    if path is None and ckpt:
        candidate = os.path.join(os.path.dirname(ckpt.rstrip(os.sep)), "config.yaml")
        if os.path.isfile(candidate):
            path = candidate
    supplied = None
    if path:
        y = load_yaml(path)
        supplied = model_config_from_dict(y.get("model", y))
    cfg = model_config_from_dict(saved) if saved else (supplied or default or CSTSSMConfig())
    if saved and supplied:
        a, b = asdict(cfg), asdict(supplied)
        # Feature alignment and HF-derived embedding widths override the original YAML.
        for d in (a, b):
            d.pop("feat_dim", None)
            d.pop("grounding_head", None)
            for key in ("dim", "vocab_size"):
                d["llm"].pop(key, None)
        different = sorted(k for k in a if a[k] != b[k])
        if different:
            if not allow_config_override:
                raise ValueError("incompatible checkpoint/config fields: " + ", ".join(different))
            cfg = supplied
    if feat_dim is not None and cfg.input_mode == "feature":
        if saved and cfg.feat_dim != feat_dim:
            raise ValueError(f"checkpoint feat_dim={cfg.feat_dim} differs from data feat_dim={feat_dim}")
        cfg = replace(cfg, feat_dim=feat_dim)
    return cfg


def restore_model(ckpt=None, config=None, feat_dim=None, default=None, device="cpu",
                  base_model=None, dtype=None, lora=None, lora_r=None, lora_alpha=None,
                  max_video_tokens=8192, grounding=False, max_missing_ratio=None):
    from ..models import CSTSSMModel
    from ..data import build_hf_tokenizer
    from ..data.dataset import ByteTokenizer
    from .lora import LoRALinear, _set_submodule, mark_only_lora_trainable

    meta = checkpoint_metadata(ckpt) if ckpt else {}
    info = dict(meta.get("restoration", {}))
    if ckpt and os.path.isdir(ckpt) and "backend" not in info:
        index_path = os.path.join(ckpt, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as stream:
                keys = json.load(stream).get("weight_map", {})
            if any(k.startswith("llm.hf.") for k in keys):
                info["backend"] = "qwen3_vl"
    dtype = dtype if dtype is not None else info.get("dtype")
    saved_base = info.get("base_model")
    if base_model and saved_base and base_model != saved_base:
        raise ValueError("base-model differs from the checkpoint's recorded base model")
    base_model = base_model or saved_base
    if info.get("backend") == "stand_in" and base_model:
        raise ValueError("a stand-in checkpoint cannot be restored as a Qwen model")
    if (meta.get("trainable_only") or info.get("backend") == "qwen3_vl") and not base_model:
        raise ValueError("trainable_only/Qwen checkpoint requires --base-model; "
                         "this older checkpoint does not record the base model name")
    cfg = resolve_model_config(ckpt, config, feat_dim, default,
                               allow_config_override=max_missing_ratio is not None)
    if grounding and not cfg.grounding_head:
        if meta.get("model_config"):
            raise ValueError("checkpoint was not trained with a grounding head")
        cfg = replace(cfg, grounding_head=True)
    adapters = info.get("lora_modules")
    if adapters is not None:
        if lora is not None and bool(adapters) != lora:
            raise ValueError("--lora/--no-lora conflicts with checkpoint adapters")
        for spec in adapters.values():
            if lora_r is not None and spec["r"] != lora_r:
                raise ValueError("lora-r conflicts with checkpoint adapters")
            if lora_alpha is not None and spec["alpha"] != lora_alpha:
                raise ValueError("lora-alpha conflicts with checkpoint adapters")
    if base_model:
        from ..integrations.qwen3_vl import build_cstssm_qwen3vl
        model = build_cstssm_qwen3vl(
            model_name=base_model, base_cfg=cfg, dtype=dtype,
            lora=(bool(lora) if lora is not None else True) if adapters is None else False,
            lora_r=lora_r if lora_r is not None else 64,
            lora_alpha=lora_alpha if lora_alpha is not None else 16)
        if (feat_dim is not None or meta.get("model_config")) and model.cfg.feat_dim != cfg.feat_dim:
            raise ValueError("base model visual dimension differs from checkpoint/data features")
        tok = build_hf_tokenizer(info.get("tokenizer_model") or base_model,
                                 max_video_tokens=max_video_tokens)
    else:
        model, tok = CSTSSMModel(cfg), ByteTokenizer()
        if adapters is None and lora:
            from .lora import apply_lora
            apply_lora(model.llm, r=lora_r or 64, alpha=lora_alpha or 16)
    if adapters:
        for name, spec in adapters.items():
            _set_submodule(model, name, LoRALinear(model.get_submodule(name), **spec))
        if base_model:
            mark_only_lora_trainable(model)
    if ckpt:
        relaxed = max_missing_ratio is not None
        if relaxed:
            print("[restore][warn] explicit missing-weight relaxation: results are not reportable")
        ratio = max_missing_ratio if relaxed else (0.99 if meta.get("trainable_only") else 0.02)
        missing, _ = load_checkpoint(model, ckpt, max_missing_ratio=ratio, tag="restore",
                                     critical_prefixes=() if relaxed else CRITICAL_SHAPE_PREFIXES)
        if not relaxed and meta.get("trainable_only"):
            params = dict(model.named_parameters())
            bad = [k for k in missing if not k.startswith("llm.")
                   or k.endswith((".A", ".B"))
                   or (k in params and params[k].requires_grad)]
            if bad:
                raise ValueError("checkpoint is missing non-recoverable weights: " + ", ".join(bad[:8]))
    return model.to(device).eval(), tok
