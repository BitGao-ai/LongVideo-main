#!/usr/bin/env python3
"""Train the GroundingHead for sub-frame temporal localization.

Single-GPU smoke:
    python scripts/train_grounding.py --device cpu --steps 50
8-GPU production:
    torchrun --standalone --nproc_per_node=8 scripts/train_grounding.py \
        --config configs/default.yaml --base-model Qwen/Qwen3-VL-4B-Instruct \
        --dtype bfloat16 --manifest data/manifests/charades_train.jsonl \
        --data-root data --load checkpoints/stage2/final --device cuda --ddp --steps 20000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.data import LoaderConfig, build_dataloader, cycle, align_feat_dim
from cst_ssm.train import Trainer, TrainConfig
from cst_ssm.utils import (model_config_from_dict, load_yaml,
                           train_config_from_yaml, loss_weights_from_yaml,
                           loader_config_from_yaml, add_ddp_args, resolve_ddp,
                           require_model_config)
from scripts._common import pick, setup_seed, warmup_loader, resolve_device_map


def build_model(cfg: CSTSSMConfig, args):
    """Build the LLM segment: real base or stand-in. Returns (model, tokenizer)."""
    if not args.base_model:
        print("[model] no --base-model: stand-in LLM, query embeddings carry no signal.")
        model = CSTSSMModel(cfg)
        if args.lora:
            from cst_ssm.utils import apply_lora, mark_only_lora_trainable
            apply_lora(model.llm, r=args.lora_r, alpha=args.lora_alpha)
            mark_only_lora_trainable(model)
        return model, None

    from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
    from cst_ssm.data import build_hf_tokenizer
    print(f"[model] base: {args.base_model} (lora={args.lora}, r={args.lora_r})")
    model = build_cstssm_qwen3vl(
        model_name=args.base_model, base_cfg=cfg, lora=args.lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, dtype=args.dtype,
        device_map=resolve_device_map(args))
    print(f"[tokenizer] loading: {args.base_model}")
    return model, build_hf_tokenizer(args.base_model, max_video_tokens=args.max_video_tokens)


def build_cfg(y, manifest, data_root, config_path=None, allow_default=False) -> CSTSSMConfig:
    """Resolve the model config with grounding head enabled, then align feat_dim."""
    if y.get("model"):
        d = dict(y["model"])
        d["grounding_head"] = True
        return align_feat_dim(model_config_from_dict(d), manifest, data_root)
    require_model_config(manifest, config_path, "scripts/train_grounding.py", allow_default)
    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96, grounding_head=True,
                       eacs_chunk=16,
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
    return align_feat_dim(cfg, manifest, data_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--load-to-device", action="store_true")
    ap.add_argument("--max-video-tokens", type=int, default=8192)
    ap.add_argument("--lora", action="store_true", default=None)
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--load", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--allow-default-config", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    add_ddp_args(ap)
    args = ap.parse_args()

    y = load_yaml(args.config) if args.config else {}
    ddp, ddp_backend = resolve_ddp(args.ddp, args.ddp_backend, y.get("train"),
                                   "scripts/train_grounding.py", device=args.device)
    ydata = y.get("data") or {}
    manifest = pick(args.manifest, ydata.get("manifest"), None)
    data_root = pick(args.data_root, ydata.get("data_root"), "")
    batch_size = pick(args.batch_size, ydata.get("batch_size"), 2)
    num_workers = pick(args.num_workers, ydata.get("num_workers"), 0)
    args.num_workers = num_workers
    seed = pick(args.seed, (y.get("train") or {}).get("seed"), 0)
    setup_seed(seed)
    if args.lora is None:
        args.lora = bool(args.base_model)

    cfg = build_cfg(y, manifest, data_root, args.config, args.allow_default_config)
    model, tokenizer = build_model(cfg, args)
    if args.base_model and model.cfg.feat_dim != cfg.feat_dim:
        sys.exit(
            f"[model] base visual dim {model.cfg.feat_dim} != manifest dim {cfg.feat_dim}; "
            f"re-extract features with {args.base_model}.")
    if args.load:
        from cst_ssm.utils import load_checkpoint
        load_checkpoint(model, args.load, tag="grounding-load",
                        max_missing_ratio=0.99 if args.base_model else 0.9)
    if args.lora and not args.base_model:
        for p in model.grounding_head.parameters():
            p.requires_grad_(True)

    tc = train_config_from_yaml(y, skip={"stage", "device", "bf16", "eacs_chunk"})
    weights = loss_weights_from_yaml(y)
    steps = pick(args.steps, None, tc.max_steps if y.get("train") else 50)
    tc.stage = "grounding"
    tc.device = args.device
    tc.bf16 = (args.device != "cpu")
    tc.max_steps = steps
    tc.lr = pick(args.lr, y.get("train", {}).get("lr"), 1e-3)
    tc.ckpt_dir = pick(args.ckpt, None,
                       tc.ckpt_dir if y.get("train") else "checkpoints/grounding")
    tc.eacs_chunk = None
    tc.ddp, tc.ddp_backend = ddp, ddp_backend
    tc.seed = seed

    lc = loader_config_from_yaml(
        y, skip={"manifest", "val_manifest", "data_root", "mode", "batch_size",
                 "num_workers", "pin_memory", "drop_last", "persistent_workers"})
    lc.manifest, lc.data_root = manifest, data_root
    lc.batch_size, lc.num_workers = batch_size, num_workers
    lc.persistent_workers = num_workers > 0
    lc.feat_dim = cfg.feat_dim
    lc.seed = seed
    lc.synth_grounding = (manifest is None)
    lc.drop_last = bool(manifest)
    lc.synth_n = steps * batch_size + 8
    tc._eff_batch = batch_size * (int(os.environ.get("WORLD_SIZE", 1)) if ddp else 1) * tc.grad_accum
    if y:
        print(f"[cfg] YAML consumed: model/train/loss/data "
              f"(lr={tc.lr}, grad_accum={tc.grad_accum}, max_steps={tc.max_steps}, batch={lc.batch_size})")
    loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    if manifest and lc.drop_last and len(ds) < batch_size:
        print(f"[data] {len(ds)} samples < batch_size {batch_size}; retrying with drop_last=False")
        lc.drop_last = False
        loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    print(f"[data] {('manifest: ' + manifest) if manifest else 'synthetic'}"
          f" ({len(ds)} samples, grounding_head={cfg.grounding_head})")

    warmup_loader(loader, None, num_workers, args.device)
    from cst_ssm.utils import cleanup_distributed
    tr = Trainer(model, tc, weights)
    try:
        tr.fit(cycle(loader))
        tr.save("final")
    finally:
        cleanup_distributed()
    print(f"[done] grounding training complete -> {tc.ckpt_dir}/final")


if __name__ == "__main__":
    main()
