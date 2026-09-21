#!/usr/bin/env python3
"""End-to-end training on the Qwen3-VL base model.

Production (8-GPU):
    torchrun --standalone --nproc_per_node=8 scripts/train_qwen3vl.py \
        --model Qwen/Qwen3-VL-4B-Instruct --manifest data/manifests/lvb_train.jsonl \
        --data-root data --lora --steps 5000 --device cuda --ddp
Pipeline check (no GPU/weights needed):
    python scripts/train_qwen3vl.py --stand-in --steps 20 --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts._common import setup_seed, warmup_loader, build_val_loader, resolve_device_map


def build_model_real(args):
    """Load the Qwen3-VL base with CST-SSM temporal layers and LoRA."""
    from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
    print(f"[model] base: {args.model} (lora={args.lora}, r={args.lora_r})")
    return build_cstssm_qwen3vl(
        model_name=args.model,
        d_model=args.d_model,
        gate_init_eps=args.gate_eps,
        eacs_chunk=args.eacs_chunk,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dtype=args.dtype,
        device_map=resolve_device_map(args),
    )


def build_model_standin(args):
    """Self-contained small model for pipeline checks (no language ability)."""
    from cst_ssm.models import CSTSSMModel, CSTSSMConfig
    from cst_ssm.modules.llm_interface import LLMConfig
    print("[model] stand-in mode (pipeline check only)")
    cfg = CSTSSMConfig(
        input_mode="feature", feat_dim=args.feat_dim, d_model=args.d_model,
        llm=LLMConfig(vocab_size=259, dim=128, n_layer=4, n_head=4, max_len=512))
    model = CSTSSMModel(cfg)
    if args.lora:
        from cst_ssm.utils import apply_lora, mark_only_lora_trainable
        apply_lora(model.llm, r=args.lora_r, alpha=args.lora_alpha)
        mark_only_lora_trainable(model)
    return model


def build_tokenizer(args):
    """HF tokenizer for the real path; None (ByteTokenizer) for stand-in."""
    if args.stand_in:
        return None
    from cst_ssm.data import build_hf_tokenizer
    print(f"[tokenizer] loading: {args.model}")
    return build_hf_tokenizer(args.model, max_video_tokens=args.max_video_tokens)


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL + CST-SSM end-to-end training")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--d-model", type=int, default=768)
    ap.add_argument("--gate-eps", type=float, default=0.1)
    ap.add_argument("--eacs-chunk", type=int, default=32)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--load-to-device", action="store_true")
    ap.add_argument("--lora", action="store_true", default=True)
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--val-manifest", default=None)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=8192)
    ap.add_argument("--max-text-len", type=int, default=8704)
    ap.add_argument("--max-video-tokens", type=int, default=8192)
    ap.add_argument("--feat-dim", type=int, default=64)
    ap.add_argument("--stage", default="finetune", choices=["finetune", "pretrain"])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="checkpoints/qwen3vl")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--load", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stand-in", action="store_true")
    from cst_ssm.utils import add_ddp_args, resolve_ddp
    add_ddp_args(ap)
    args = ap.parse_args()

    ddp, ddp_backend = resolve_ddp(args.ddp, args.ddp_backend, None,
                                   "scripts/train_qwen3vl.py", device=args.device)
    setup_seed(args.seed)

    model = build_model_standin(args) if args.stand_in else build_model_real(args)
    if args.load:
        from cst_ssm.utils import load_checkpoint
        print(f"[model] warm start: {args.load}")
        load_checkpoint(model, args.load, max_missing_ratio=0.95, tag="qwen3vl-load")

    tokenizer = build_tokenizer(args)
    from cst_ssm.data import LoaderConfig, build_dataloader, cycle, infer_feat_dim
    feat_dim = args.feat_dim
    if args.manifest and not args.stand_in:
        d = infer_feat_dim(args.manifest, args.data_root)
        if d:
            feat_dim = d
            print(f"[data] inferred feat_dim={d}")
    lc = LoaderConfig(
        manifest=args.manifest, data_root=args.data_root, mode="feature",
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"), drop_last=bool(args.manifest),
        persistent_workers=args.num_workers > 0,
        max_frames=args.max_frames, max_text_len=args.max_text_len,
        feat_dim=feat_dim, seed=args.seed,
        synth_n=args.steps * args.batch_size + 8)
    loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    if args.manifest and lc.drop_last and len(ds) < args.batch_size:
        print(f"[data] {len(ds)} samples < batch_size {args.batch_size}; retrying with drop_last=False")
        lc.drop_last = False
        loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    val_loader = build_val_loader(LoaderConfig, build_dataloader, args.val_manifest,
                                  args.data_root, "feature", args.batch_size,
                                  args.num_workers, args.max_frames, args.max_text_len,
                                  feat_dim, tokenizer)
    print(f"[data] {('manifest: ' + args.manifest) if args.manifest else 'synthetic'}"
          f" ({len(ds)} samples, batch={args.batch_size})")

    warmup_loader(loader, val_loader, args.num_workers, args.device)

    from cst_ssm.train import Trainer, TrainConfig, LossWeights
    tc = TrainConfig(
        stage=args.stage, lr=args.lr, device=args.device, max_steps=args.steps,
        grad_accum=args.grad_accum, eacs_chunk=args.eacs_chunk,
        bf16=(args.device != "cpu"), ckpt_dir=args.ckpt, ckpt_every=args.ckpt_every,
        val_every=500 if args.val_manifest else 0, seed=args.seed,
        ddp=ddp, ddp_backend=ddp_backend)
    tc._eff_batch = args.batch_size * (int(os.environ.get("WORLD_SIZE", 1)) if ddp else 1) * args.grad_accum
    weights = LossWeights(task=1.0, pred=0.5, update_rate=0.1, spectral=0.05)
    from cst_ssm.utils import cleanup_distributed
    tr = Trainer(model, tc, weights)
    print(f"[train] stage={args.stage}, steps={args.steps}, lr={args.lr}, device={args.device}")
    try:
        tr.fit(cycle(loader), val_loader=val_loader)
        tr.save("final")
    finally:
        cleanup_distributed()
    print(f"[done] training complete -> {args.ckpt}/final")


if __name__ == "__main__":
    main()
