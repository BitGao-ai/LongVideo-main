#!/usr/bin/env python3
"""Stage-2：端到端任务微调（设计方案 §4.1 阶段②）+ LoRA 省显存。

数据加载走统一工厂 cst_ssm.data.build_dataloader（合成/真实 manifest 同口径）：
    # 合成快跑（无数据，smoke）
    python scripts/train_stage2.py --device cpu --steps 20 --lora
    # 真实数据（如 LongVideoBench 转出的 manifest；feat_dim 自动从特征推断）
    python scripts/train_stage2.py --manifest data/manifests/lvb_val.jsonl \
        --data-root data --num-workers 8 --batch-size 4 --steps 5000 --lora --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.data import LoaderConfig, build_dataloader, cycle, infer_feat_dim
from cst_ssm.train import Trainer, TrainConfig, LossWeights
from cst_ssm.utils import (apply_lora, mark_only_lora_trainable,
                           model_config_from_dict, load_yaml)


def build_cfg(path: str | None, manifest: str | None, data_root: str) -> CSTSSMConfig:
    if path:
        return model_config_from_dict(load_yaml(path)["model"])
    feat_dim = 64
    if manifest:                                       # 真实数据：自动对齐特征维，防 feat_dim 错配
        d = infer_feat_dim(manifest, data_root)
        if d:
            feat_dim = d
            print(f"[cfg] 从特征自动推断 feat_dim={d}（如需自定义模型维请用 --config）")
    return CSTSSMConfig(input_mode="feature", feat_dim=feat_dim, d_model=96,
                        llm=LLMConfig(vocab_size=259, dim=128, n_layer=4, n_head=4, max_len=64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ckpt", default="checkpoints/stage2")
    ap.add_argument("--lora", action="store_true", help="LoRA 冻结基座省显存")
    ap.add_argument("--load", default=None, help="从 stage-1 分片检查点热启")
    # 数据加载
    ap.add_argument("--manifest", default=None, help="真实数据 manifest jsonl；缺省用合成数据")
    ap.add_argument("--val-manifest", default=None, help="验证集 manifest jsonl；缺省跳过验证")
    ap.add_argument("--data-root", default="", help="feature_ref 的根")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--mode", default="feature", choices=["feature", "pixel"])
    args = ap.parse_args()

    cfg = build_cfg(args.config, args.manifest, args.data_root)
    model = CSTSSMModel(cfg)
    if args.load:
        from cst_ssm.utils import load_sharded
        model.load_state_dict(load_sharded(args.load), strict=False)
    if args.lora:
        apply_lora(model, r=16, alpha=32)
        mark_only_lora_trainable(model)

    lc = LoaderConfig(manifest=args.manifest, data_root=args.data_root, mode=args.mode,
                      batch_size=args.batch_size, num_workers=args.num_workers,
                      pin_memory=(args.device != "cpu"), drop_last=bool(args.manifest),
                      persistent_workers=args.num_workers > 0,
                      feat_dim=cfg.feat_dim, synth_n=args.steps * args.batch_size + 8)
    loader, ds = build_dataloader(lc)
    # 验证集（可选）
    val_loader = None
    if args.val_manifest:
        vlc = LoaderConfig(manifest=args.val_manifest, data_root=args.data_root, mode=args.mode,
                           batch_size=args.batch_size, num_workers=args.num_workers,
                           shuffle=False, feat_dim=cfg.feat_dim)
        val_loader, _ = build_dataloader(vlc)
    print(f"[data] {'真实 manifest: %s' % args.manifest if args.manifest else '合成数据'}"
          f"（{len(ds)} 样本, batch={args.batch_size}, workers={args.num_workers}）"
          f"{' + val: %s' % args.val_manifest if args.val_manifest else ''}")

    tr = Trainer(model, TrainConfig(stage="finetune", device=args.device, max_steps=args.steps,
                                    eacs_chunk=16, bf16=(args.device != "cpu"), ckpt_dir=args.ckpt,
                                    val_every=500 if args.val_manifest else 0),
                 LossWeights(update_rate=0.1, spectral=0.05))
    tr.fit(cycle(loader), val_loader=val_loader)        # cycle：max_steps 超一个 epoch 也不提前停
    tr.save("final")


if __name__ == "__main__":
    main()
