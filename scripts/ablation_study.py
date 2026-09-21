#!/usr/bin/env python3
"""Unified ablation experiments across six module groups.

Metrics are mean task loss on a held-out synthetic set (different seed from training).

Usage:
    python scripts/ablation_study.py --steps 20 --device cpu
    python scripts/ablation_study.py --real --group 1 2 --steps 500 --device cuda --bf16
    python scripts/ablation_study.py --steps 20 --output results/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.models.cst_ssm_model import assert_config_applied
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.train import Trainer, TrainConfig, LossWeights
from cst_ssm.train.contrastive import CPIBWeights
from cst_ssm.train.trainer import move_to
from cst_ssm.data import (LoaderConfig, build_dataloader, cycle,
                          SyntheticVideoDataset, collate_fn)


def make_base_config(args) -> CSTSSMConfig:
    """Base config shared by stand-in and --real paths (sole carrier of ablation switches)."""
    return CSTSSMConfig(
        input_mode="feature", feat_dim=args.feat_dim, d_model=args.d_model,
        cpib_distill=True, cpib_hidden=128, cpi_modulation=0.5,
        diff_kv=True, diff_kv_rank=4, eacs_chunk=16,
        llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=256))


def apply_overrides(cfg: CSTSSMConfig, **kw) -> CSTSSMConfig:
    return replace(cfg, **kw)


def build_model(cfg: CSTSSMConfig, args):
    """Build the model; switches must enter cfg before construction."""
    if not args.real:
        model = CSTSSMModel(cfg)
    else:
        from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
        model = build_cstssm_qwen3vl(model_name=args.model, base_cfg=cfg,
                                     lora=True, lora_r=args.lora_r)
        for p in model.llm.parameters():
            if not (p.requires_grad and p.dim() == 2):
                p.requires_grad_(False)
    assert_config_applied(model, cfg)
    return model


def make_loader(args, seed: int, shuffle: bool):
    lc = LoaderConfig(
        mode="feature", batch_size=args.batch_size,
        feat_dim=args.feat_dim, shuffle=shuffle,
        synth_n=args.synth_n, max_frames=args.max_frames, num_workers=0)
    loader, _ = build_dataloader(lc)
    return loader


def make_eval_loader(args):
    """Held-out synthetic set with a different seed from training."""
    ds = SyntheticVideoDataset(n=args.eval_batches * args.batch_size,
                               L=args.max_frames, P=4, d=args.feat_dim,
                               seed=args.seed + 10_000)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      collate_fn=partial(collate_fn, pad_id=256))


@torch.no_grad()
def evaluate(model, loader, device, weights) -> dict:
    """Mean task loss and update rate on the held-out set."""
    model.eval()
    tot_loss, tot_ur, n = 0.0, 0.0, 0
    for batch in loader:
        batch = move_to(batch, device)
        out = model(batch)
        tot_loss += float(out["loss"].item())
        tot_ur += float(out.get("update_rate", torch.zeros(())).item())
        n += 1
    model.train()
    return {"loss": tot_loss / max(n, 1), "update_rate": tot_ur / max(n, 1), "n_batches": n}


def run_one(name: str, cfg: CSTSSMConfig, weights: LossWeights, args) -> dict:
    """Train one ablation arm and report held-out metrics."""
    model = build_model(cfg, args)
    tc = TrainConfig(stage="finetune", lr=args.lr, device=args.device,
                     max_steps=args.steps, bf16=args.bf16, log_every=args.steps + 1)
    tr = Trainer(model, tc, weights)
    train_loader = make_loader(args, args.seed, shuffle=True)
    eval_loader = make_eval_loader(args)

    t0 = time.time()
    tr.fit(cycle(train_loader))
    elapsed = time.time() - t0

    m = evaluate(tr.model, eval_loader, args.device, weights)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {
        "name": name,
        "heldout_loss": round(m["loss"], 5),
        "update_rate": round(m["update_rate"], 5),
        "eval_batches": m["n_batches"],
        "params": n_params,
        "trainable": n_train,
        "train_time_s": round(elapsed, 2),
    }
    print(f"  [{name:>30}] heldout_loss={m['loss']:.4f} ur={m['update_rate']:.4f} "
          f"trainable={n_train:,}/{n_params:,} time={elapsed:.1f}s")
    return result


def group1_innovation1(args) -> list[dict]:
    """Group 1: CPIB-Distill components."""
    print("\n[Group 1] CPIB-Distill")
    base = make_base_config(args)
    results = []

    w_full = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))
    results.append(run_one("full_cpib", base, w_full, args))

    w_no_nce = LossWeights(cpib=CPIBWeights(nce=0.0, kl=0.01, cf=0.05))
    results.append(run_one("no_nce", base, w_no_nce, args))

    w_no_cf = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.0))
    results.append(run_one("no_cf", base, w_no_cf, args))

    w_no_kl = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.0, cf=0.05))
    results.append(run_one("no_ib_kl", base, w_no_kl, args))

    results.append(run_one("no_cpib_baseline",
                           apply_overrides(base, cpib_distill=False, cpi_modulation=0.0),
                           LossWeights(), args))

    return results


def group2_gate_kind(args) -> list[dict]:
    """Group 2: gate types."""
    print("\n[Group 2] gate types")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    base = make_base_config(args)
    for gate_kind in ["event", "random", "always"]:
        results.append(run_one(f"gate_{gate_kind}",
                               apply_overrides(base, eacs_gate_kind=gate_kind), w, args))

    results.append(run_one("event_no_cpi_mod",
                           apply_overrides(base, cpi_modulation=0.0), w, args))

    return results


def group3_disc_mode(args) -> list[dict]:
    """Group 3: discretization modes."""
    print("\n[Group 3] discretization modes")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    base = make_base_config(args)
    for disc in ["continuous", "fixed", "learned"]:
        results.append(run_one(f"disc_{disc}",
                               apply_overrides(base, eacs_disc_mode=disc), w, args))

    return results


def group4_diff_kv(args) -> list[dict]:
    """Group 4: differential KV cache (adaptive vs fixed intervals)."""
    print("\n[Group 4] differential KV")
    base = make_base_config(args)
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))
    results = [
        run_one("diff_kv_adaptive_G", apply_overrides(base, diff_kv_interval=0), w, args),
        run_one("diff_kv_fixed_G8", apply_overrides(base, diff_kv_interval=8), w, args),
        run_one("diff_kv_fixed_G16", apply_overrides(base, diff_kv_interval=16), w, args),
        run_one("no_diff_kv", apply_overrides(base, diff_kv=False), w, args),
    ]
    _report_kv_compression(base, args)
    return results


@torch.no_grad()
def _report_kv_compression(base: CSTSSMConfig, args) -> None:
    """Report compression ratios with reconstruction errors on the held-out set."""
    loader = make_eval_loader(args)
    batch = move_to(next(iter(loader)), args.device)
    print(f"  {'config':>20} {'bases':>7} {'resid':>7} {'ratio':>8} {'reconMSE':>10}")
    for tag, G in (("adaptive_G", 0), ("fixed_G8", 8), ("fixed_G16", 16)):
        model = build_model(apply_overrides(base, diff_kv_interval=G), args).to(args.device).eval()
        st = model.measure_kv_compression(batch)
        print(f"  {tag:>20} {st['n_base']:>7} {st['n_residual']:>7} "
              f"{st['compression_ratio']:>8.3f} {st['recon_mse']:>10.4f}")
    if not args.real:
        print("  synthetic data has no redundancy; ratios are 1.0 by construction")


def group5_unified_cpi(args) -> list[dict]:
    """Group 5: unified vs separate CPI signals."""
    print("\n[Group 5] unified CPI")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    base = make_base_config(args)
    results.append(run_one("cpi_unified", base, w, args))
    results.append(run_one("cpi_separate",
                           apply_overrides(base, cpi_modulation=0.0), w, args))

    return results


def group6_spectral_init(args) -> list[dict]:
    """Group 6: spectral vs random init."""
    print("\n[Group 6] spectral init")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    base = make_base_config(args)
    results.append(run_one("spectral_init",
                           apply_overrides(base, eacs_use_spectral_init=True), w, args))
    results.append(run_one("random_init",
                           apply_overrides(base, eacs_use_spectral_init=False), w, args))

    return results


ALL_GROUPS = {
    1: group1_innovation1,
    2: group2_gate_kind,
    3: group3_disc_mode,
    4: group4_diff_kv,
    5: group5_unified_cpi,
    6: group6_spectral_init,
}


def main():
    ap = argparse.ArgumentParser(description="CST-SSM ablation experiments")
    ap.add_argument("--group", type=int, nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--feat-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--synth-n", type=int, default=256)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    groups = args.group or list(ALL_GROUPS.keys())
    all_results = {}
    print(f"{'='*60}")
    print(f"ablation | groups={groups} steps={args.steps} device={args.device}")
    print(f"model: {'real Qwen3-VL base (LLM frozen, LoRA)' if args.real else f'stand-in d_model={args.d_model}'}")
    print(f"metric: held-out mean task loss (seed={args.seed + 10000}, {args.eval_batches} batches)")
    if not args.real:
        print("stand-in results validate plumbing only, not module effectiveness")
    print(f"{'='*60}")

    for g in groups:
        if g in ALL_GROUPS:
            all_results[f"group_{g}"] = ALL_GROUPS[g](args)

    print(f"\n{'='*60}")
    print("ablation complete")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"results saved -> {args.output}")


if __name__ == "__main__":
    main()
