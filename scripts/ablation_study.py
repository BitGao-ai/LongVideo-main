#!/usr/bin/env python3
"""统一消融实验脚本（P4：对齐设计文档 GATE 门逻辑 + 实验执行清单）。

消融组（设计§6.3 + 实验清单）：
  组1 - 创新点1生死线：Full vs 去L_NCE vs 去L_cf vs 无CPIB(baseline)
  组2 - 创新点2门控：event vs random vs always(稠密) vs CPI调制 vs 无CPI调制
  组3 - 创新点2离散化：continuous vs fixed vs learned
  组4 - 创新点3：diff_kv vs 无diff_kv；自适应G vs 固定G
  组5 - 统一主线：CPI贯通三处 vs 各独立信号
  组6 - 谱初始化：spectral_init vs random_init

用法：
    # 跑全部消融（stand-in 快速验证）
    python scripts/ablation_study.py --stand-in --steps 20 --device cpu

    # 跑指定组
    python scripts/ablation_study.py --group 1 2 --steps 500 --device cuda

    # 输出 JSON 结果
    python scripts/ablation_study.py --stand-in --steps 20 --output results/ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.train import Trainer, TrainConfig, LossWeights
from cst_ssm.train.contrastive import CPIBWeights
from cst_ssm.data import LoaderConfig, build_dataloader, cycle


def make_base_config(args) -> CSTSSMConfig:
    """基础配置（stand-in 或生产）。"""
    if args.stand_in:
        return CSTSSMConfig(
            input_mode="feature", feat_dim=64, d_model=96,
            cpib_distill=True, cpib_hidden=128, cpi_modulation=0.5,
            diff_kv=True, diff_kv_rank=4,
            llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=256))
    return CSTSSMConfig(
        input_mode="feature", feat_dim=768, d_model=768,
        cpib_distill=True, cpi_modulation=0.5,
        diff_kv=True, diff_kv_rank=8,
        llm=LLMConfig(vocab_size=151936, dim=3584, n_layer=28, n_head=28, max_len=4096))


def make_loader(args):
    lc = LoaderConfig(
        mode="feature", batch_size=args.batch_size,
        feat_dim=64 if args.stand_in else 768,
        synth_n=args.steps * args.batch_size + 8,
        max_frames=32, num_workers=0)
    loader, _ = build_dataloader(lc)
    return loader


def run_one(name: str, cfg: CSTSSMConfig, weights: LossWeights, args) -> dict:
    """训练一个消融配置，返回最终 loss 和指标。"""
    model = CSTSSMModel(cfg)
    tc = TrainConfig(stage="finetune", lr=1e-3, device=args.device,
                     max_steps=args.steps, bf16=False, log_every=args.steps + 1)
    tr = Trainer(model, tc, weights)
    loader = make_loader(args)

    t0 = time.time()
    tr.fit(cycle(loader))
    elapsed = time.time() - t0

    # 最后一步的 loss
    batch = next(cycle(loader))
    from cst_ssm.train.trainer import move_to
    batch = move_to(batch, args.device)
    model.eval()
    with torch.no_grad():
        out = model(batch)
    final_loss = out["loss"].item()
    update_rate = out.get("update_rate", torch.tensor(0.0)).item()

    n_params = sum(p.numel() for p in model.parameters())
    result = {
        "name": name,
        "final_task_loss": final_loss,
        "update_rate": update_rate,
        "params": n_params,
        "train_time_s": round(elapsed, 2),
    }
    print(f"  [{name:>30}] loss={final_loss:.4f} ur={update_rate:.4f} "
          f"params={n_params:,} time={elapsed:.1f}s")
    return result


# ==================== 消融组定义 ====================

def group1_innovation1(args) -> list[dict]:
    """组1：创新点1生死线（CPIB-Distill 各组件消融）。"""
    print("\n[组1] 创新点1 CPIB-Distill 消融")
    base = make_base_config(args)
    results = []

    # Full（完整 CPIB）
    w_full = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))
    results.append(run_one("full_cpib", base, w_full, args))

    # 去 L_NCE
    w_no_nce = LossWeights(cpib=CPIBWeights(nce=0.0, kl=0.01, cf=0.05))
    results.append(run_one("no_nce", base, w_no_nce, args))

    # 去 L_cf
    w_no_cf = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.0))
    results.append(run_one("no_cf", base, w_no_cf, args))

    # 去 IB-KL
    w_no_kl = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.0, cf=0.05))
    results.append(run_one("no_ib_kl", base, w_no_kl, args))

    # Baseline（无 CPIB）
    cfg_no_cpib = make_base_config(args)
    cfg_no_cpib.cpib_distill = False
    cfg_no_cpib.cpi_modulation = 0.0
    w_base = LossWeights()
    results.append(run_one("no_cpib_baseline", cfg_no_cpib, w_base, args))

    return results


def group2_gate_kind(args) -> list[dict]:
    """组2：创新点2 门控类型消融。"""
    print("\n[组2] 创新点2 门控类型消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    for gate_kind in ["event", "random", "always"]:
        cfg = make_base_config(args)
        cfg.eacs_gate_kind = gate_kind
        results.append(run_one(f"gate_{gate_kind}", cfg, w, args))

    # CPI 调制 vs 无调制
    cfg_no_mod = make_base_config(args)
    cfg_no_mod.cpi_modulation = 0.0
    results.append(run_one("event_no_cpi_mod", cfg_no_mod, w, args))

    return results


def group3_disc_mode(args) -> list[dict]:
    """组3：创新点2 离散化模式消融。"""
    print("\n[组3] 创新点2 离散化模式消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    for disc in ["continuous", "fixed", "learned"]:
        cfg = make_base_config(args)
        cfg.eacs_disc_mode = disc
        results.append(run_one(f"disc_{disc}", cfg, w, args))

    return results


def group4_diff_kv(args) -> list[dict]:
    """组4：创新点3 差分KV缓存消融。"""
    print("\n[组4] 创新点3 差分KV消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    # 有 diff_kv
    cfg_with = make_base_config(args)
    results.append(run_one("with_diff_kv", cfg_with, w, args))

    # 无 diff_kv
    cfg_without = make_base_config(args)
    cfg_without.diff_kv = False
    results.append(run_one("no_diff_kv", cfg_without, w, args))

    return results


def group5_unified_cpi(args) -> list[dict]:
    """组5：统一主线（CPI贯通 vs 独立信号）。"""
    print("\n[组5] 统一主线消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    # CPI 贯通（完整）
    cfg_full = make_base_config(args)
    results.append(run_one("cpi_unified", cfg_full, w, args))

    # CPI 不贯通 EventGate（独立残差门控）
    cfg_sep = make_base_config(args)
    cfg_sep.cpi_modulation = 0.0
    results.append(run_one("cpi_separate", cfg_sep, w, args))

    return results


def group6_spectral_init(args) -> list[dict]:
    """组6：谱初始化消融。"""
    print("\n[组6] 谱初始化消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    cfg_spec = make_base_config(args)
    cfg_spec.eacs_use_spectral_init = True
    results.append(run_one("spectral_init", cfg_spec, w, args))

    cfg_rand = make_base_config(args)
    cfg_rand.eacs_use_spectral_init = False
    results.append(run_one("random_init", cfg_rand, w, args))

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
    ap = argparse.ArgumentParser(description="CST-SSM 消融实验")
    ap.add_argument("--group", type=int, nargs="*", default=None,
                    help="指定消融组编号（1-6），缺省跑全部")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stand-in", action="store_true", default=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    groups = args.group or list(ALL_GROUPS.keys())
    all_results = {}
    print(f"{'='*60}")
    print(f"消融实验 | groups={groups} steps={args.steps} device={args.device}")
    print(f"{'='*60}")

    for g in groups:
        if g in ALL_GROUPS:
            all_results[f"group_{g}"] = ALL_GROUPS[g](args)

    # 汇总
    print(f"\n{'='*60}")
    print("消融实验完成")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"结果已保存 → {args.output}")


if __name__ == "__main__":
    main()
