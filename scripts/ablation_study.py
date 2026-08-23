#!/usr/bin/env python3
"""统一消融实验脚本（P4：对齐设计文档 GATE 门逻辑 + 实验执行清单）。

消融组（设计§6.3 + 实验清单）：
  组1 - 创新点1生死线：Full vs 去L_NCE vs 去L_cf vs 无CPIB(baseline)
  组2 - 创新点2门控：event vs random vs always(稠密) vs CPI调制 vs 无CPI调制
  组3 - 创新点2离散化：continuous vs fixed vs learned
  组4 - 创新点3：diff_kv vs 无diff_kv；自适应G vs 固定G(8/16)，附压缩率与重构误差
  组5 - 统一主线：CPI贯通三处 vs 各独立信号
  组6 - 谱初始化：spectral_init vs random_init

指标口径：所有组都在**独立留出集**（与训练集不同 seed 的合成数据）上报平均 task loss，
不是训练集单批的拟合程度。

用法：
    # 验证消融管线可跑通（自包含小模型，CPU 即可）
    python scripts/ablation_study.py --steps 20 --device cpu

    # 论文用：真实 Qwen3-VL 底座 + LoRA + 冻结 LLM，组间差异才归因于被消融的模块
    python scripts/ablation_study.py --real --group 1 2 --steps 500 --device cuda --bf16

    # 输出 JSON 结果
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
    """基础配置——**消融开关的唯一载体**，stand-in 与 --real 共用同一份。

    注意这里不再构造 dim=3584/n_layer=28 的自研 VisualConditionedLM：那是一个约 49 亿
    参数、权重全随机的 stand-in（不是 Qwen3-VL），15 个消融配置各造一个既跑不动、
    组间 loss 差异也支撑不了任何关于创新点的结论。

    --real 时本函数返回的 cfg 会原样传给 build_cstssm_qwen3vl(base_cfg=…)，工厂只覆盖
    input_mode / feat_dim / llm 三个由底座决定的字段，全部消融开关照常生效；
    d_model 由 --d-model 控制（接 Qwen 底座时建议 768）。
    """
    return CSTSSMConfig(
        input_mode="feature", feat_dim=args.feat_dim, d_model=args.d_model,
        cpib_distill=True, cpib_hidden=128, cpi_modulation=0.5,
        diff_kv=True, diff_kv_rank=4, eacs_chunk=16,
        llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=256))


def apply_overrides(cfg: CSTSSMConfig, **kw) -> CSTSSMConfig:
    """在基础配置上打消融开关（dataclass 浅拷贝，互不干扰）。"""
    return replace(cfg, **kw)


def build_model(cfg: CSTSSMConfig, args):
    """按配置建模型。--real 时用真实 Qwen3-VL 底座 + 冻结 LLM，只训 CST-SSM 各段。

    冻结 LLM 是消融的必要条件：组间差异必须归因于被消融的模块，而不是一个几十亿参数的
    语言模型在几百步里的随机漂移。

    **开关必须在构造前进入 cfg**：CPIB / diff_kv / 门控类型 / 离散化模式 / 谱初始化都是在
    CSTSSMModel.__init__ 里一次性建好的。早先这里是先调工厂建模型、再 setattr(model.cfg, …)，
    那样一个模块都不会重建——组 2/3/5/6 各臂拿到的其实是同一个模型，组 1 因 model.cpib is None
    而各权重臂也相同，组 4 直接 AssertionError。返回前用 assert_config_applied 把这条路堵死。
    """
    if not args.real:
        model = CSTSSMModel(cfg)
    else:
        from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
        model = build_cstssm_qwen3vl(model_name=args.model, base_cfg=cfg,
                                     lora=True, lora_r=args.lora_r)
        for p in model.llm.parameters():
            if not (p.requires_grad and p.dim() == 2):   # 保留 LoRA A/B 可训
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
    """独立留出集：与训练集用不同 seed 的合成数据，且不打乱。

    此前的 `next(cycle(train_loader))` 每次都返回**训练用 loader 的第一批**，报的是
    训练集单批拟合程度而不是泛化性能——组间比较会被过拟合速度而非模块效果主导。
    """
    ds = SyntheticVideoDataset(n=args.eval_batches * args.batch_size,
                               L=args.max_frames, P=4, d=args.feat_dim,
                               seed=args.seed + 10_000)      # 与训练集不同 seed
    return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                      collate_fn=partial(collate_fn, pad_id=256))


@torch.no_grad()
def evaluate(model, loader, device, weights) -> dict:
    """在留出集上求平均 task loss 与更新率。"""
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
    """训练一个消融配置，在**独立留出集**上报指标。"""
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
    results.append(run_one("no_cpib_baseline",
                           apply_overrides(base, cpib_distill=False, cpi_modulation=0.0),
                           LossWeights(), args))

    return results


def group2_gate_kind(args) -> list[dict]:
    """组2：创新点2 门控类型消融。"""
    print("\n[组2] 创新点2 门控类型消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    base = make_base_config(args)
    for gate_kind in ["event", "random", "always"]:
        results.append(run_one(f"gate_{gate_kind}",
                               apply_overrides(base, eacs_gate_kind=gate_kind), w, args))

    # CPI 调制 vs 无调制
    results.append(run_one("event_no_cpi_mod",
                           apply_overrides(base, cpi_modulation=0.0), w, args))

    return results


def group3_disc_mode(args) -> list[dict]:
    """组3：创新点2 离散化模式消融。"""
    print("\n[组3] 创新点2 离散化模式消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    base = make_base_config(args)
    for disc in ["continuous", "fixed", "learned"]:
        results.append(run_one(f"disc_{disc}",
                               apply_overrides(base, eacs_disc_mode=disc), w, args))

    return results


def group4_diff_kv(args) -> list[dict]:
    """组4：创新点3 差分KV缓存消融（含自适应 G vs 固定 G）。"""
    print("\n[组4] 创新点3 差分KV消融")
    base = make_base_config(args)
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))
    results = [
        # 自适应 G：只由相对残差能量决定何时插基准（base_interval=0 = 不设上界）
        run_one("diff_kv_adaptive_G", apply_overrides(base, diff_kv_interval=0), w, args),
        # 固定 G：每 G 帧强制插一个基准帧，作为"自适应"的对照臂
        run_one("diff_kv_fixed_G8", apply_overrides(base, diff_kv_interval=8), w, args),
        run_one("diff_kv_fixed_G16", apply_overrides(base, diff_kv_interval=16), w, args),
        run_one("no_diff_kv", apply_overrides(base, diff_kv=False), w, args),
    ]
    # 压缩率/重构误差要一起报：只报压缩率的话，靠丢信息就能刷好看
    _report_kv_compression(base, args)
    return results


@torch.no_grad()
def _report_kv_compression(base: CSTSSMConfig, args) -> None:
    """在留出集上实测各 G 设置的压缩率与重构误差（配合组4 的 loss 一起解读）。

    压缩率必须和重构误差一起看：只报压缩率的话，靠丢信息就能刷好看。
    另外合成数据是**独立随机噪声、帧间毫无冗余**，相对残差恒超阈值 → 每帧都成基准帧、
    压缩率必然是 1.0。这个数字只有在真实视频特征（--real + 真实 manifest）上才有意义。
    """
    loader = make_eval_loader(args)
    batch = move_to(next(iter(loader)), args.device)
    print(f"  {'配置':>20} {'基准帧':>7} {'中间帧':>7} {'压缩率':>8} {'重构MSE':>10}")
    for tag, G in (("adaptive_G", 0), ("fixed_G8", 8), ("fixed_G16", 16)):
        model = build_model(apply_overrides(base, diff_kv_interval=G), args).to(args.device).eval()
        st = model.measure_kv_compression(batch)
        print(f"  {tag:>20} {st['n_base']:>7} {st['n_residual']:>7} "
              f"{st['compression_ratio']:>8.3f} {st['recon_mse']:>10.4f}")
    if not args.real:
        print("  ⚠ 合成数据帧间无冗余 → 压缩率必为 1.0；需在真实视频特征上测才有意义")


def group5_unified_cpi(args) -> list[dict]:
    """组5：统一主线（CPI贯通 vs 独立信号）。"""
    print("\n[组5] 统一主线消融")
    results = []
    w = LossWeights(cpib=CPIBWeights(nce=0.1, kl=0.01, cf=0.05))

    base = make_base_config(args)
    results.append(run_one("cpi_unified", base, w, args))
    results.append(run_one("cpi_separate",
                           apply_overrides(base, cpi_modulation=0.0), w, args))

    return results


def group6_spectral_init(args) -> list[dict]:
    """组6：谱初始化消融。"""
    print("\n[组6] 谱初始化消融")
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
    ap = argparse.ArgumentParser(description="CST-SSM 消融实验")
    ap.add_argument("--group", type=int, nargs="*", default=None,
                    help="指定消融组编号（1-6），缺省跑全部")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast（GPU 建议开）")
    # 模型规模
    ap.add_argument("--feat-dim", type=int, default=64, help="输入特征维（合成数据）")
    ap.add_argument("--d-model", type=int, default=96, help="CST-SSM 时序建模维")
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--synth-n", type=int, default=256, help="合成训练集样本数")
    ap.add_argument("--eval-batches", type=int, default=8, help="留出集 batch 数")
    # 真实底座
    ap.add_argument("--real", action="store_true",
                    help="用真实 Qwen3-VL 底座 + LoRA + 冻结 LLM（需 transformers 与权重）。"
                         "缺省用自包含小模型验证管线——注意小模型的消融结论只说明管线通，"
                         "不构成论文证据")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="--real 时的底座")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    groups = args.group or list(ALL_GROUPS.keys())
    all_results = {}
    print(f"{'='*60}")
    print(f"消融实验 | groups={groups} steps={args.steps} device={args.device}")
    print(f"模型: {'真实 Qwen3-VL 底座 (LLM 冻结, LoRA)' if args.real else f'自包含小模型 d_model={args.d_model}'}")
    print(f"指标: 独立留出集（seed={args.seed + 10000}，{args.eval_batches} batch）平均 task loss")
    if not args.real:
        print("⚠ 未加 --real：以下数字仅验证消融管线可跑通，不能作为创新点有效性的证据")
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
