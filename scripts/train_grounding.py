#!/usr/bin/env python3
"""训练定位打分头 GroundingHead（供 CSG/VFR 亚帧定位；产出 infer_grounding 加载的 --ckpt）。

在帧时刻用连续读出 + query 条件头打分，对 [gt_start,gt_end] 内/外做 BCE 学习相关度曲线。
数据走统一工厂：--manifest（含 grounding 标注 gt_start/gt_end 的 jsonl）或合成定位数据兜底。

    # 合成快跑（无数据，验证可训）
    python scripts/train_grounding.py --device cpu --steps 50
    # 真实定位数据（Charades-STA/ActivityNet 转成 VideoSample+gt 的 manifest）
    python scripts/train_grounding.py --manifest data/manifests/charades_train.jsonl \
        --data-root data --steps 20000 --config configs/default.yaml --lora
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.data import LoaderConfig, build_dataloader, cycle, infer_feat_dim
from cst_ssm.train import Trainer, TrainConfig
from cst_ssm.utils import model_config_from_dict, load_yaml


def build_cfg(path, manifest, data_root) -> CSTSSMConfig:
    if path:
        d = load_yaml(path)["model"]
        d["grounding_head"] = True                     # 定位训练必须开头
        return model_config_from_dict(d)
    feat_dim = 64
    if manifest:
        fd = infer_feat_dim(manifest, data_root)
        if fd:
            feat_dim = fd
            print(f"[cfg] 从特征自动推断 feat_dim={fd}")
    return CSTSSMConfig(input_mode="feature", feat_dim=feat_dim, d_model=96, grounding_head=True,
                        llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--ckpt", default="checkpoints/grounding")
    ap.add_argument("--lora", action="store_true", help="LoRA 冻结基座，只训头+投影")
    ap.add_argument("--load", default=None, help="从 stage-2 检查点热启（复用视觉/时序/LLM）")
    ap.add_argument("--manifest", default=None, help="含 gt_start/gt_end 的定位 manifest；缺省合成")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = build_cfg(args.config, args.manifest, args.data_root)
    model = CSTSSMModel(cfg)
    if args.load:
        from cst_ssm.utils import load_sharded
        model.load_state_dict(load_sharded(args.load), strict=False)  # 头是新参数，strict=False
    if args.lora:
        from cst_ssm.utils import apply_lora, mark_only_lora_trainable
        apply_lora(model, r=16, alpha=32)
        mark_only_lora_trainable(model)
        for p in model.grounding_head.parameters():    # 头始终可训
            p.requires_grad_(True)

    lc = LoaderConfig(manifest=args.manifest, data_root=args.data_root,
                      batch_size=args.batch_size, num_workers=args.num_workers,
                      feat_dim=cfg.feat_dim, synth_grounding=(args.manifest is None),
                      drop_last=bool(args.manifest), synth_n=args.steps * args.batch_size + 8)
    loader, ds = build_dataloader(lc)
    print(f"[data] {'真实定位 manifest: %s' % args.manifest if args.manifest else '合成定位数据'}"
          f"（{len(ds)} 样本, grounding_head={cfg.grounding_head}）")

    tr = Trainer(model, TrainConfig(stage="grounding", lr=args.lr, device=args.device,
                                    max_steps=args.steps, bf16=(args.device != "cpu"),
                                    ckpt_dir=args.ckpt))
    tr.fit(cycle(loader))
    tr.save("final")
    print(f"[done] 定位头训练完成 → {args.ckpt}/final（infer_grounding --ckpt 加载）")


if __name__ == "__main__":
    main()
