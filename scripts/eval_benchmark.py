#!/usr/bin/env python3
"""Benchmark eval: QA accuracy and efficiency metrics.

Usage:
    python scripts/eval_benchmark.py --ckpt checkpoints/qwen3vl/final \
        --manifest data/benchmarks/mlvu_test.jsonl --data-root data --device cuda
    python scripts/eval_benchmark.py --ckpt checkpoints/qwen3vl/final --mode efficiency --device cuda
    python scripts/eval_benchmark.py --stand-in --mode qa --steps 10
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def _build_cfg(args):
    """Restore the model config from config.yaml next to the checkpoint, else defaults."""
    from cst_ssm.models import CSTSSMConfig
    from cst_ssm.utils import model_config_from_dict, load_yaml
    cfg_path = args.config or os.path.join(os.path.dirname(args.ckpt or "."), "config.yaml")
    if cfg_path and os.path.exists(cfg_path):
        y = load_yaml(cfg_path)
        cfg_dict = y.get("model") if isinstance(y.get("model"), dict) else y
        print(f"[eval] config from {cfg_path}")
        return model_config_from_dict(cfg_dict)
    print("[eval][warn] no config.yaml found; using default config")
    return CSTSSMConfig(input_mode="feature", feat_dim=768, d_model=768)


def _preflight_ckpt(args):
    """Refuse to score a real-base checkpoint without --base-model."""
    from cst_ssm.utils import checkpoint_metadata
    if args.base_model or not args.ckpt:
        return
    meta = checkpoint_metadata(args.ckpt)
    if meta.get("trainable_only"):
        sys.exit(
            f"[eval] {args.ckpt} stores trainable weights only "
            f"(trainable_only=True); it was trained on a real base. "
            f"Add --base-model Qwen/Qwen3-VL-4B-Instruct (same as training).")


def load_model(args):
    """Load the trained model; returns (model, tokenizer or None)."""
    if args.stand_in:
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.modules.llm_interface import LLMConfig
        cfg = CSTSSMConfig(
            input_mode="feature", feat_dim=64, d_model=96,
            cpib_distill=True, cpi_modulation=0.5,
            llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=256))
        print("[eval] stand-in model")
        return CSTSSMModel(cfg).to(args.device).eval(), None

    from cst_ssm.utils import load_checkpoint
    from cst_ssm.utils.checkpoint import CRITICAL_SHAPE_PREFIXES
    _DEFAULT_MAX_MISSING_RATIO = 0.02
    cfg = _build_cfg(args)
    _preflight_ckpt(args)

    if args.base_model:
        from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
        from cst_ssm.data import build_hf_tokenizer
        from cst_ssm.models import assert_config_applied
        print(f"[eval] base: {args.base_model} (lora={args.lora}, r={args.lora_r})")
        model = build_cstssm_qwen3vl(
            model_name=args.base_model, base_cfg=cfg, lora=args.lora,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, dtype=args.dtype)
        assert_config_applied(model, cfg)
        tok = build_hf_tokenizer(args.base_model, max_video_tokens=args.max_video_tokens)
        if args.ckpt:
            load_checkpoint(model, args.ckpt, max_missing_ratio=0.99, tag="eval")
        else:
            print("[eval][warn] no --ckpt: random weights, metrics are meaningless")
        return model.to(args.device).eval(), tok

    from cst_ssm.models import CSTSSMModel
    model = CSTSSMModel(cfg)
    if args.ckpt:
        _forced = args.max_missing_ratio is not None
        if _forced:
            print(f"[eval][warn] explicit max_missing_ratio={args.max_missing_ratio}: "
                  f"results are not reportable")
        load_checkpoint(model, args.ckpt,
                        max_missing_ratio=(args.max_missing_ratio
                                           if _forced else _DEFAULT_MAX_MISSING_RATIO),
                        tag="eval",
                        critical_prefixes=() if _forced else CRITICAL_SHAPE_PREFIXES)
    print(f"[eval] model loaded: {args.ckpt}")
    return model.to(args.device).eval(), None


def _mcq_answer_index(row: dict):
    """Answer index within options; None when unresolvable."""
    opts = row.get("options")
    ans = row.get("answer")
    if not opts or ans is None:
        return None
    if isinstance(ans, int):
        return ans if 0 <= ans < len(opts) else None
    s = str(ans).strip()
    if len(s) == 1 and s.isalpha():
        idx = ord(s.upper()) - ord("A")
        return idx if 0 <= idx < len(opts) else None
    for j, o in enumerate(opts):
        if str(o).strip() == s:
            return j
    return None


def _letter_token_id(tok, letter: str):
    """Token id for one letter, compatible with both tokenizers."""
    ids = tok.encode(letter)
    bos = getattr(tok, "BOS", None)
    if bos is None:
        bos = getattr(tok, "bos_id", None)
    ids = [i for i in ids if i != bos]
    return ids[0] if ids else None


def _eval_autocast(args):
    """Eval autocast context; enabled only with --bf16."""
    from cst_ssm.utils import autocast_ctx
    dev = "cuda" if str(args.device).startswith("cuda") else "cpu"
    return autocast_ctx(bool(getattr(args, "bf16", False)), device_type=dev)


def eval_qa(model, args, tokenizer=None):
    """QA eval: true MCQ accuracy when rows carry options, else token hit rate."""
    from cst_ssm.data import LoaderConfig, build_dataloader

    lc = LoaderConfig(
        manifest=args.manifest, data_root=args.data_root, mode="feature",
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False,
        max_frames=args.max_frames, max_text_len=args.max_text_len,
        feat_dim=model.cfg.feat_dim, synth_n=args.steps * args.batch_size)
    loader, ds = build_dataloader(lc, tokenizer=tokenizer)

    rows = getattr(ds, "rows", None)
    mcq = bool(rows) and all(r.get("options") for r in rows)

    correct, total = 0, 0
    n_batches = 0
    n_unaligned = 0
    letter_cache: dict = {}
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            if n_batches >= args.steps:
                break
            n_batches += 1
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with _eval_autocast(args):
                out = model(batch)
            logits = out.get("logits")
            labels = batch.get("labels")
            if logits is None or labels is None:
                continue
            B = labels.shape[0]
            if mcq:
                rid = batch.get("row_index")
                for i in range(B):
                    row_idx = int(rid[i]) if rid is not None else -1
                    if row_idx < 0 or row_idx >= len(rows):
                        n_unaligned += 1
                        continue
                    gt = _mcq_answer_index(rows[row_idx])
                    if gt is None:
                        continue
                    valid_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if valid_pos.numel() == 0 or valid_pos[0] == 0:
                        continue
                    pred_pos = int(valid_pos[0].item()) - 1
                    n_opt = len(rows[row_idx]["options"])
                    if n_opt not in letter_cache:
                        ids = [_letter_token_id(ds.tok, chr(ord("A") + j)) for j in range(n_opt)]
                        letter_cache[n_opt] = ids if all(t is not None for t in ids) else None
                    ids = letter_cache[n_opt]
                    if ids is None:
                        continue
                    sub = logits[i, pred_pos, ids]
                    if int(sub.argmax().item()) == gt:
                        correct += 1
                    total += 1
            else:
                valid = labels != -100
                if valid.any():
                    preds = logits.argmax(dim=-1)
                    correct += (preds[valid] == labels[valid]).sum().item()
                    total += valid.sum().item()

    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    metric_name = "MCQ accuracy" if mcq else "teacher-forcing token hit rate (not QA accuracy)"
    fs = getattr(ds, "failure_stats", None)
    print(f"\n{'='*50}")
    print("[QA results]")
    print(f"  metric: {metric_name}")
    print(f"  result: {acc:.4f} ({correct}/{total})")
    print(f"  elapsed: {elapsed:.1f}s")
    if n_unaligned:
        print(f"  {n_unaligned} samples excluded (unmapped manifest rows)")
    if fs and fs["failures"]:
        print(f"  data failures {fs['failures']} (fallbacks {fs['fallbacks']}) / "
              f"{fs['total_rows']} rows - fix data before trusting metrics")
    print(f"{'='*50}")
    key = "qa_acc" if mcq else "token_acc"
    res = {key: acc, "total_counted": total, "elapsed": elapsed, "unaligned": n_unaligned}
    if fs:
        res["data_failures"] = fs["failures"]
        res["data_fallbacks"] = fs["fallbacks"]
    return res


def eval_efficiency(model, args):
    """Efficiency metrics: update rate, latency, and peak memory vs frame count."""
    from cst_ssm.data import SyntheticVideoDataset, collate_fn
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"[efficiency] device={args.device}")
    print(f"{'frames':>8} {'upd_rate':>8} {'lat(ms)':>10} {'mem(MB)':>10}")
    print(f"{'-'*60}")

    results = []
    for n_frames in [32, 64, 128, 256, 512]:
        ds = SyntheticVideoDataset(n=4, L=n_frames, P=4, d=model.cfg.feat_dim)
        loader = DataLoader(ds, batch_size=1, collate_fn=lambda b: collate_fn(b, pad_id=256))

        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad(), _eval_autocast(args):
                _ = model(batch)
            break

        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        update_rates = []
        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad(), _eval_autocast(args):
                out = model(batch)
            update_rates.append(out["update_rate"].item())
        elapsed = (time.time() - t0) * 1000 / len(loader)

        mem = 0
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        avg_ur = sum(update_rates) / len(update_rates)
        print(f"{n_frames:>8} {avg_ur:>8.4f} {elapsed:>10.1f} {mem:>10.1f}")
        results.append({"frames": n_frames, "update_rate": avg_ur,
                        "latency_ms": elapsed, "peak_mem_mb": mem})

    print(f"{'='*60}")
    return results


def main():
    ap = argparse.ArgumentParser(description="CST-SSM benchmark eval")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--data-root", default="")
    ap.add_argument("--mode", default="qa", choices=["qa", "efficiency"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=8192)
    ap.add_argument("--max-text-len", type=int, default=8704)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--stand-in", action="store_true")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--lora", action="store_true", default=None)
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--max-video-tokens", type=int, default=8192)
    ap.add_argument("--config", default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-missing-ratio", type=float, default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        args.device = "cpu"
    if args.lora is None:
        args.lora = bool(args.base_model)
    if args.bf16 and args.device == "cpu":
        print("[eval][warn] CPU bf16 autocast validates plumbing only")

    model, tokenizer = load_model(args)

    if args.mode == "qa":
        results = eval_qa(model, args, tokenizer=tokenizer)
    else:
        results = eval_efficiency(model, args)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[saved] -> {args.output}")


if __name__ == "__main__":
    main()
