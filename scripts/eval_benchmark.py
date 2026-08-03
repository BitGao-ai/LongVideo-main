#!/usr/bin/env python3
"""真实 benchmark 评测脚本（P3：对齐设计文档阶段4 GATE-4）。

支持评测：
  - 视频问答准确率（MLVU / LVBench / VideoMME / EgoSchema 格式）
  - 效率指标：更新率、推理延迟、显存峰值
  - 连续查询/定位指标（CSG/VFR 已有，此处补 QA 准确率）

用法：
    # 评测 QA 准确率（manifest 格式：每行 {video_id, prompt, answer, options?}）
    python scripts/eval_benchmark.py --ckpt checkpoints/qwen3vl/final \
        --manifest data/benchmarks/mlvu_test.jsonl --data-root data \
        --device cuda

    # 评测效率（不同帧数下的更新率 + 延迟）
    python scripts/eval_benchmark.py --ckpt checkpoints/qwen3vl/final \
        --mode efficiency --device cuda

    # stand-in 模式（验证评测管线逻辑）
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


def load_model(args):
    """加载训练好的模型。"""
    if args.stand_in:
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.modules.llm_interface import LLMConfig
        cfg = CSTSSMConfig(
            input_mode="feature", feat_dim=64, d_model=96,
            cpib_distill=True, cpi_modulation=0.5,
            llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=256))
        model = CSTSSMModel(cfg)
        print("[eval] stand-in 模型")
    else:
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.utils import load_sharded, model_config_from_dict, load_yaml
        # 从检查点加载
        state = load_sharded(args.ckpt)
        # 尝试从同目录读 config
        cfg_path = os.path.join(os.path.dirname(args.ckpt), "config.yaml")
        if os.path.exists(cfg_path):
            cfg = model_config_from_dict(load_yaml(cfg_path))
        else:
            cfg = CSTSSMConfig(input_mode="feature", feat_dim=768, d_model=768)
        model = CSTSSMModel(cfg)
        model.load_state_dict(state, strict=False)
        print(f"[eval] 加载模型: {args.ckpt}")
    return model.to(args.device).eval()


def eval_qa(model, args):
    """评测 QA 准确率（multiple-choice 或 open-ended）。"""
    from cst_ssm.data import LoaderConfig, build_dataloader

    lc = LoaderConfig(
        manifest=args.manifest, data_root=args.data_root, mode="feature",
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False,
        max_frames=args.max_frames, max_text_len=args.max_text_len,
        feat_dim=64 if args.stand_in else 768, synth_n=args.steps * args.batch_size)
    loader, ds = build_dataloader(lc)

    correct, total = 0, 0
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch)
            # 取 logits 的 argmax 作为预测
            logits = out.get("logits")
            if logits is not None and "labels" in batch:
                labels = batch["labels"]
                # 只看非 -100 位置
                valid = labels != -100
                if valid.any():
                    preds = logits.argmax(dim=-1)
                    correct += (preds[valid] == labels[valid]).sum().item()
                    total += valid.sum().item()
            if total >= args.steps * args.batch_size * 10:
                break

    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    print(f"\n{'='*50}")
    print(f"[QA 评测结果]")
    print(f"  准确率: {acc:.4f} ({correct}/{total})")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  吞吐: {total/max(elapsed,0.01):.1f} tokens/s")
    print(f"{'='*50}")
    return {"accuracy": acc, "total_tokens": total, "elapsed": elapsed}


def eval_efficiency(model, args):
    """评测效率指标：不同帧数下的更新率 + 推理延迟 + 显存。"""
    from cst_ssm.data import SyntheticVideoDataset, collate_fn
    from torch.utils.data import DataLoader

    print(f"\n{'='*60}")
    print(f"[效率评测] 设备={args.device}")
    print(f"{'帧数':>8} {'更新率':>8} {'延迟(ms)':>10} {'显存(MB)':>10}")
    print(f"{'-'*60}")

    results = []
    for n_frames in [32, 64, 128, 256, 512]:
        ds = SyntheticVideoDataset(n=4, L=n_frames, P=4, d=64 if args.stand_in else 768)
        loader = DataLoader(ds, batch_size=1, collate_fn=lambda b: collate_fn(b, pad_id=256))

        # 预热
        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                _ = model(batch)
            break

        # 计时
        if args.device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        update_rates = []
        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad():
                out = model(batch)
            update_rates.append(out["update_rate"].item())
        elapsed = (time.time() - t0) * 1000 / len(loader)

        mem = 0
        if args.device == "cuda":
            mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        avg_ur = sum(update_rates) / len(update_rates)
        print(f"{n_frames:>8} {avg_ur:>8.4f} {elapsed:>10.1f} {mem:>10.1f}")
        results.append({"frames": n_frames, "update_rate": avg_ur,
                        "latency_ms": elapsed, "peak_mem_mb": mem})

    print(f"{'='*60}")
    return results


def main():
    ap = argparse.ArgumentParser(description="CST-SSM Benchmark 评测")
    ap.add_argument("--ckpt", default=None, help="检查点目录")
    ap.add_argument("--manifest", default=None, help="评测数据 manifest")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--mode", default="qa", choices=["qa", "efficiency"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=512)
    ap.add_argument("--max-text-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--stand-in", action="store_true")
    ap.add_argument("--output", default=None, help="结果输出 JSON 路径")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退 CPU")
        args.device = "cpu"

    model = load_model(args)

    if args.mode == "qa":
        results = eval_qa(model, args)
    else:
        results = eval_efficiency(model, args)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[saved] → {args.output}")


if __name__ == "__main__":
    main()
