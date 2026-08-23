#!/usr/bin/env python3
"""Stage-1：时序预测自监督预训练（设计方案 §4.1 阶段①）。

数据加载走统一工厂 cst_ssm.data.build_dataloader（合成/真实 manifest 同口径）：
    # 合成快跑（无数据，smoke）
    python scripts/train_stage1.py --device cpu --steps 20
    # 真实无标注长视频 manifest（自监督只用 features+timestamps，answer 可空）
    python scripts/train_stage1.py --manifest data/manifests/pretrain.jsonl \
        --data-root data --num-workers 8 --batch-size 8 --steps 20000 --config configs/default.yaml
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
                           loader_config_from_yaml)


def _pick(cli, yaml_val, fallback):
    """三级优先级：命令行 > YAML > 脚本内置默认（详见 train_stage2._pick）。"""
    if cli is not None:
        return cli
    return yaml_val if yaml_val is not None else fallback


def build_cfg(y: dict, manifest: str | None, data_root: str) -> CSTSSMConfig:
    # 无论配置来自 YAML 还是脚本内置默认，都统一过一次 align_feat_dim：feat_dim 由抽特征
    # 的视觉塔唯一决定，不是可自由设定的建模超参。此前 YAML 带 model 段时直接 return，
    # 把特征维校验整段短路了（见 align_feat_dim 的 docstring）。
    if y.get("model"):
        return align_feat_dim(model_config_from_dict(y["model"]), manifest, data_root)
    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96, eacs_chunk=16,
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=4, n_head=4, max_len=64))
    return align_feat_dim(cfg, manifest, data_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 YAML train.max_steps")
    ap.add_argument("--ckpt", default=None, help="覆盖 YAML train.ckpt_dir")
    # 数据加载
    ap.add_argument("--manifest", default=None, help="真实数据 manifest jsonl；缺省用合成数据")
    ap.add_argument("--val-manifest", default=None, help="验证集 manifest jsonl；缺省跳过验证")
    ap.add_argument("--data-root", default=None, help="feature_ref 的根")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--mode", default=None, choices=["feature", "pixel"])
    args = ap.parse_args()

    # ---- 配置解析：YAML 四段全部消费（model / train / loss / data），命令行覆盖 YAML ----
    y = load_yaml(args.config) if args.config else {}
    ydata = y.get("data") or {}
    val_manifest = _pick(args.val_manifest, ydata.get("val_manifest"), None)
    manifest = _pick(args.manifest, ydata.get("manifest"), None)
    data_root = _pick(args.data_root, ydata.get("data_root"), "")
    mode = _pick(args.mode, ydata.get("mode"), "feature")

    cfg = build_cfg(y, manifest, data_root)
    model = CSTSSMModel(cfg)

    tc = train_config_from_yaml(y, skip={"stage", "device", "bf16", "eacs_chunk"})
    weights = loss_weights_from_yaml(y)
    steps = _pick(args.steps, None, tc.max_steps if y.get("train") else 20)
    tc.stage = "pretrain"
    tc.device = args.device
    tc.bf16 = (args.device != "cpu")
    tc.max_steps = steps
    tc.ckpt_dir = _pick(args.ckpt, None, tc.ckpt_dir if y.get("train") else "checkpoints/stage1")
    tc.eacs_chunk = None            # 沿用模型配置，理由同 train_stage2

    batch_size = _pick(args.batch_size, ydata.get("batch_size"), 2)
    num_workers = _pick(args.num_workers, ydata.get("num_workers"), 0)
    lc = loader_config_from_yaml(
        y, skip={"manifest", "val_manifest", "data_root", "mode", "batch_size",
                 "num_workers", "pin_memory", "drop_last", "persistent_workers"})
    lc.manifest, lc.data_root, lc.mode = manifest, data_root, mode
    lc.batch_size, lc.num_workers = batch_size, num_workers
    lc.pin_memory = (args.device != "cpu")
    lc.drop_last = bool(manifest)
    lc.persistent_workers = num_workers > 0
    lc.feat_dim = cfg.feat_dim
    lc.synth_n = steps * batch_size + 8
    if y:
        print(f"[cfg] YAML 四段已全部消费：model/train/loss/data "
              f"(lr={tc.lr}, grad_accum={tc.grad_accum}, max_steps={tc.max_steps}, "
              f"pred={weights.pred}, recon={weights.recon}, batch={lc.batch_size})")
    loader, ds = build_dataloader(lc)
    # 防护：真实 manifest + drop_last=True 时，若样本数不足一个 batch，loader 为空（StopIteration）
    if manifest and lc.drop_last and len(ds) < batch_size:
        print(f"[data] 警告: 样本数({len(ds)}) < batch_size({batch_size})，"
              f"drop_last=True 会丢弃所有样本导致 loader 为空；已自动改为 drop_last=False 重建")
        lc.drop_last = False
        loader, ds = build_dataloader(lc)
    # 验证集（可选）
    val_loader = None
    if val_manifest:
        vlc = LoaderConfig(manifest=val_manifest, data_root=data_root, mode=mode,
                           batch_size=batch_size, num_workers=num_workers,
                           shuffle=False, persistent_workers=num_workers > 0,
                           max_frames=lc.max_frames, max_text_len=lc.max_text_len,
                           feat_dim=cfg.feat_dim)
        val_loader, _ = build_dataloader(vlc)
    print(f"[data] {'真实 manifest: %s' % manifest if manifest else '合成数据'}"
          f"（{len(ds)} 样本, batch={batch_size}, workers={num_workers}）"
          f"{' + val: %s' % val_manifest if val_manifest else ''}")

    # persistent_workers=True 使 worker 保持存活，后续 fit()/validate() 不再重复 fork。
    if num_workers > 0 and args.device != "cpu":
        print("[data] 预热 DataLoader worker（CUDA 初始化前派生子进程）...")
        try:
            _warm = iter(loader)
            next(_warm)
            del _warm
            if val_loader is not None:                  # 验证集同样预热，防止验证阶段再 fork 死锁
                _warm_v = iter(val_loader)
                next(_warm_v)
                del _warm_v
        except StopIteration:
            sys.exit(f"[data] 错误: loader 为空，无法预热。请检查 manifest 是否非空、"
                     f"batch_size({batch_size}) 是否超过样本数、特征文件是否可读")
        print("[data] worker 就绪，首个 batch 加载成功")

    if val_manifest and not tc.val_every:
        tc.val_every = 500
    tr = Trainer(model, tc, weights)
    print("[train] 开始训练（每 10 步打印一次日志）...")
    tr.fit(cycle(loader), val_loader=val_loader)        # cycle：按 step 训练不因 epoch 边界提前停
    tr.save("final")


if __name__ == "__main__":
    main()
