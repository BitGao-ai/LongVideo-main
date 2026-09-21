#!/usr/bin/env python3
"""Stage-2: end-to-end finetuning with optional Qwen3-VL base model and LoRA.

Single-GPU smoke (stand-in LLM):
    python scripts/train_stage2.py --device cpu --steps 20
8-GPU production:
    torchrun --standalone --nproc_per_node=8 scripts/train_stage2.py \
        --config configs/default.yaml --base-model Qwen/Qwen3-VL-4B-Instruct \
        --dtype bfloat16 --manifest data/manifests/lvb_train.jsonl --data-root data \
        --load checkpoints/stage1/final --device cuda --ddp \
        --num-workers 8 --batch-size 4 --steps 5000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.data import LoaderConfig, build_dataloader, cycle, align_feat_dim
from cst_ssm.train import Trainer, TrainConfig, LossWeights
from cst_ssm.utils import (apply_lora, mark_only_lora_trainable,
                           model_config_from_dict, load_yaml,
                           train_config_from_yaml, loss_weights_from_yaml,
                           loader_config_from_yaml, add_ddp_args, resolve_ddp,
                           require_model_config)
from scripts._common import pick, setup_seed, warmup_loader, build_val_loader, resolve_device_map


def build_model(cfg: CSTSSMConfig, args):
    """Build the LLM segment: real Qwen3-VL base or stand-in. Returns (model, tokenizer)."""
    if not args.base_model:
        print("[model] no --base-model: using stand-in LLM (pipeline check only).")
        model = CSTSSMModel(cfg)
        if args.lora:
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
    tok = build_hf_tokenizer(args.base_model, max_video_tokens=args.max_video_tokens)
    return model, tok


def preflight_video_tokens(ds, tok, tag: str = "data") -> None:
    """Check that <video> placeholders match frame count for the real-base path."""
    s = ds[0]
    n_frames = int(s["features"].shape[0])
    n_vid = int((s["input_ids"] == tok.video_token_id).sum())
    if n_vid == 0:
        sys.exit(f"[{tag}] prompt has no <video> placeholder: the LLM would see no video.")
    if n_vid != n_frames:
        print(f"[{tag}] video tokens {n_vid} != frames {n_frames}; trailing frames are dropped.")
    else:
        print(f"[{tag}] video tokens == frames == {n_frames}")


def build_cfg(y: dict, manifest: str | None, data_root: str,
              config_path: str | None = None, allow_default: bool = False) -> CSTSSMConfig:
    """Resolve the model config (YAML or smoke default), then align feat_dim."""
    if y.get("model"):
        return align_feat_dim(model_config_from_dict(y["model"]), manifest, data_root)
    require_model_config(manifest, config_path, "scripts/train_stage2.py", allow_default)
    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96, eacs_chunk=16,
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=4, n_head=4, max_len=64))
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
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--mode", default=None, choices=["feature", "pixel"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--allow-default-config", action="store_true")
    add_ddp_args(ap)
    args = ap.parse_args()

    y = load_yaml(args.config) if args.config else {}
    ddp, ddp_backend = resolve_ddp(args.ddp, args.ddp_backend, y.get("train"),
                                   "scripts/train_stage2.py", device=args.device)
    ydata = y.get("data") or {}
    val_manifest = pick(args.val_manifest, ydata.get("val_manifest"), None)
    manifest = pick(args.manifest, ydata.get("manifest"), None)
    data_root = pick(args.data_root, ydata.get("data_root"), "")
    mode = pick(args.mode, ydata.get("mode"), "feature")
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
        load_checkpoint(model, args.load, tag="stage2-load",
                        max_missing_ratio=0.99 if args.base_model else 0.5)

    tc = train_config_from_yaml(y, skip={"stage", "device", "bf16", "eacs_chunk"})
    weights = loss_weights_from_yaml(y)
    steps = pick(args.steps, None, tc.max_steps if y.get("train") else 20)
    tc.stage = "finetune"
    tc.device = args.device
    tc.bf16 = (args.device != "cpu")
    tc.max_steps = steps
    tc.ckpt_dir = pick(args.ckpt, None, tc.ckpt_dir if y.get("train") else "checkpoints/stage2")
    tc.eacs_chunk = None
    tc.ddp, tc.ddp_backend = ddp, ddp_backend
    tc.seed = seed

    lc = loader_config_from_yaml(
        y, skip={"manifest", "val_manifest", "data_root", "mode",
                 "batch_size", "num_workers", "pin_memory", "drop_last",
                 "persistent_workers"})
    lc.manifest, lc.data_root, lc.mode = manifest, data_root, mode
    lc.batch_size, lc.num_workers = batch_size, num_workers
    lc.pin_memory = (args.device != "cpu")
    lc.drop_last = bool(manifest)
    lc.persistent_workers = num_workers > 0
    lc.feat_dim = cfg.feat_dim
    lc.seed = seed
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
    val_loader = build_val_loader(LoaderConfig, build_dataloader, val_manifest, data_root,
                                  mode, batch_size, num_workers, lc.max_frames,
                                  lc.max_text_len, cfg.feat_dim, tokenizer)
    print(f"[data] {('manifest: ' + manifest) if manifest else 'synthetic'}"
          f" ({len(ds)} samples, batch={batch_size}, workers={num_workers})")
    if tokenizer is not None and manifest:
        preflight_video_tokens(ds, tokenizer)

    warmup_loader(loader, val_loader, num_workers, args.device)
    if val_manifest and not tc.val_every:
        tc.val_every = 500
    from cst_ssm.utils import cleanup_distributed
    tr = Trainer(model, tc, weights)
    print("[train] starting...")
    try:
        tr.fit(cycle(loader), val_loader=val_loader)
        tr.save("final")
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
