#!/usr/bin/env python3
"""Stage-1: self-supervised temporal pretraining.

Single-GPU:
    python scripts/train_stage1.py --device cpu --steps 20
8-GPU (one node):
    torchrun --standalone --nproc_per_node=8 scripts/train_stage1.py \
        --config configs/default.yaml --manifest data/manifests/pretrain.jsonl \
        --data-root data --device cuda --ddp --batch-size 8 --num-workers 8 --steps 20000
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
from cst_ssm.utils import (model_config_from_dict, load_yaml,
                           train_config_from_yaml, loss_weights_from_yaml,
                           loader_config_from_yaml, add_ddp_args, resolve_ddp,
                           require_model_config)
from scripts._common import pick, setup_seed, warmup_loader, build_val_loader


def build_cfg(y: dict, manifest: str | None, data_root: str,
              config_path: str | None = None, allow_default: bool = False) -> CSTSSMConfig:
    """Resolve the model config (YAML or smoke default), then align feat_dim."""
    if y.get("model"):
        return align_feat_dim(model_config_from_dict(y["model"]), manifest, data_root)
    require_model_config(manifest, config_path, "scripts/train_stage1.py", allow_default)
    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96, eacs_chunk=16,
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=4, n_head=4, max_len=64))
    return align_feat_dim(cfg, manifest, data_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
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
                                   "scripts/train_stage1.py", device=args.device)
    ydata = y.get("data") or {}
    val_manifest = pick(args.val_manifest, ydata.get("val_manifest"), None)
    manifest = pick(args.manifest, ydata.get("manifest"), None)
    data_root = pick(args.data_root, ydata.get("data_root"), "")
    mode = pick(args.mode, ydata.get("mode"), "feature")
    seed = pick(args.seed, (y.get("train") or {}).get("seed"), 0)
    setup_seed(seed)

    cfg = build_cfg(y, manifest, data_root, args.config, args.allow_default_config)
    model = CSTSSMModel(cfg)

    tc = train_config_from_yaml(y, skip={"stage", "device", "bf16", "eacs_chunk"})
    weights = loss_weights_from_yaml(y)
    steps = pick(args.steps, None, tc.max_steps if y.get("train") else 20)
    tc.stage = "pretrain"
    tc.device = args.device
    tc.bf16 = (args.device != "cpu")
    tc.max_steps = steps
    tc.ckpt_dir = pick(args.ckpt, None, tc.ckpt_dir if y.get("train") else "checkpoints/stage1")
    tc.eacs_chunk = None
    tc.ddp, tc.ddp_backend = ddp, ddp_backend
    tc.seed = seed

    batch_size = pick(args.batch_size, ydata.get("batch_size"), 2)
    num_workers = pick(args.num_workers, ydata.get("num_workers"), 0)
    lc = loader_config_from_yaml(
        y, skip={"manifest", "val_manifest", "data_root", "mode", "batch_size",
                 "num_workers", "pin_memory", "drop_last", "persistent_workers"})
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
    loader, ds = build_dataloader(lc)
    if manifest and lc.drop_last and len(ds) < batch_size:
        print(f"[data] {len(ds)} samples < batch_size {batch_size}; retrying with drop_last=False")
        lc.drop_last = False
        loader, ds = build_dataloader(lc)
    val_loader = build_val_loader(LoaderConfig, build_dataloader, val_manifest, data_root,
                                  mode, batch_size, num_workers, lc.max_frames,
                                  lc.max_text_len, cfg.feat_dim)
    print(f"[data] {('manifest: ' + manifest) if manifest else 'synthetic'}"
          f" ({len(ds)} samples, batch={batch_size}, workers={num_workers})")

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
