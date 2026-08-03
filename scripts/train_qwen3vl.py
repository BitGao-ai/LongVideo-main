#!/usr/bin/env python3
"""接入 Qwen3-VL 底座的端到端训练脚本（P0：生产路径入口）。

把 build_cstssm_qwen3vl() 工厂（已有但此前无脚本调用）串通真实数据流 + HF 分词器 + Trainer，
产出真正可做视频问答/定位的模型。设计文档§4 要求：基于开源 Video-LLM 底座 + LoRA 微调。

用法：
    # 真实训练（需 transformers + Qwen3-VL 权重 + GPU）
    python scripts/train_qwen3vl.py --model Qwen/Qwen3-VL-4B-Instruct \
        --manifest data/manifests/lvb_train.jsonl --data-root data \
        --lora --lora-r 64 --steps 5000 --device cuda

    # 从 stage-1 检查点热启（复用视觉/时序权重）
    python scripts/train_qwen3vl.py --model Qwen/Qwen3-VL-4B-Instruct \
        --manifest data/manifests/lvb_train.jsonl --data-root data \
        --load checkpoints/stage1/final --lora --steps 5000

    # stand-in 模式（验证脚本逻辑，无需 GPU/transformers/权重）
    python scripts/train_qwen3vl.py --stand-in --steps 20 --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_model_real(args):
    """真实路径：加载 Qwen3-VL 底座 + CST-SSM 时序 + LoRA。"""
    from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
    print(f"[model] 加载 Qwen3-VL 底座: {args.model} (lora={args.lora}, r={args.lora_r})")
    model = build_cstssm_qwen3vl(
        model_name=args.model,
        d_model=args.d_model,
        gate_init_eps=args.gate_eps,
        eacs_chunk=args.eacs_chunk,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        dtype=args.dtype,
    )
    return model


def build_model_standin(args):
    """stand-in 路径：自包含小模型，验证训练管线逻辑（无需 GPU/transformers）。"""
    from cst_ssm.models import CSTSSMModel, CSTSSMConfig
    from cst_ssm.modules.llm_interface import LLMConfig
    print("[model] stand-in 模式：自包含小模型（验证管线用，非生产）")
    cfg = CSTSSMConfig(
        input_mode="feature", feat_dim=args.feat_dim, d_model=args.d_model,
        llm=LLMConfig(vocab_size=259, dim=128, n_layer=4, n_head=4, max_len=512))
    model = CSTSSMModel(cfg)
    if args.lora:
        from cst_ssm.utils import apply_lora, mark_only_lora_trainable
        apply_lora(model, r=args.lora_r, alpha=args.lora_alpha)
        mark_only_lora_trainable(model)
    return model


def build_tokenizer(args):
    """构造分词器：真实路径用 HFTokenizer，stand-in 用 ByteTokenizer。"""
    if args.stand_in:
        return None  # build_dataloader 默认 ByteTokenizer
    from cst_ssm.data import build_hf_tokenizer
    print(f"[tokenizer] 加载 Qwen3-VL 分词器: {args.model}")
    return build_hf_tokenizer(args.model, max_video_tokens=args.max_video_tokens)


def main():
    ap = argparse.ArgumentParser(description="Qwen3-VL + CST-SSM 端到端训练")
    # 模型
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="HF 模型名")
    ap.add_argument("--d-model", type=int, default=768, help="CST-SSM 时序建模维")
    ap.add_argument("--gate-eps", type=float, default=0.1, help="事件门控初始阈值")
    ap.add_argument("--eacs-chunk", type=int, default=32, help="EACS 分块检查点大小")
    ap.add_argument("--dtype", default=None, help="模型 dtype（如 bfloat16），缺省自动")
    # LoRA
    ap.add_argument("--lora", action="store_true", default=True, help="LoRA 微调（默认开）")
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    # 数据
    ap.add_argument("--manifest", default=None, help="真实数据 manifest jsonl")
    ap.add_argument("--val-manifest", default=None, help="验证集 manifest jsonl；缺省跳过验证")
    ap.add_argument("--data-root", default="", help="feature_ref 的根目录")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=512)
    ap.add_argument("--max-text-len", type=int, default=2048)
    ap.add_argument("--max-video-tokens", type=int, default=512)
    ap.add_argument("--feat-dim", type=int, default=64, help="stand-in 模式特征维")
    # 训练
    ap.add_argument("--stage", default="finetune", choices=["finetune", "pretrain"])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="checkpoints/qwen3vl")
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--load", default=None, help="从检查点热启（strict=False）")
    # stand-in
    ap.add_argument("--stand-in", action="store_true",
                    help="用自包含小模型验证管线（无需 GPU/transformers/Qwen3-VL 权重）")
    args = ap.parse_args()

    # ---- 模型 ----
    if args.stand_in:
        model = build_model_standin(args)
    else:
        model = build_model_real(args)
    if args.load:
        from cst_ssm.utils import load_sharded
        print(f"[model] 热启: {args.load}")
        model.load_state_dict(load_sharded(args.load), strict=False)

    # ---- 分词器 + 数据 ----
    tokenizer = build_tokenizer(args)
    from cst_ssm.data import LoaderConfig, build_dataloader, cycle, infer_feat_dim
    feat_dim = args.feat_dim
    if args.manifest and not args.stand_in:
        d = infer_feat_dim(args.manifest, args.data_root)
        if d:
            feat_dim = d
            print(f"[data] 从特征自动推断 feat_dim={d}")
    lc = LoaderConfig(
        manifest=args.manifest, data_root=args.data_root, mode="feature",
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=(args.device != "cpu"), drop_last=bool(args.manifest),
        persistent_workers=args.num_workers > 0,
        max_frames=args.max_frames, max_text_len=args.max_text_len,
        feat_dim=feat_dim, synth_n=args.steps * args.batch_size + 8)
    loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    # 验证集（可选）
    val_loader = None
    if args.val_manifest:
        vlc = LoaderConfig(
            manifest=args.val_manifest, data_root=args.data_root, mode="feature",
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, max_frames=args.max_frames, max_text_len=args.max_text_len,
            feat_dim=feat_dim)
        val_loader, _ = build_dataloader(vlc, tokenizer=tokenizer)
    print(f"[data] {'manifest: ' + args.manifest if args.manifest else '合成数据'}"
          f"（{len(ds)} 样本, batch={args.batch_size}）"
          f"{' + val: %s' % args.val_manifest if args.val_manifest else ''}")

    # ---- 训练 ----
    from cst_ssm.train import Trainer, TrainConfig, LossWeights
    tc = TrainConfig(
        stage=args.stage, lr=args.lr, device=args.device, max_steps=args.steps,
        grad_accum=args.grad_accum, eacs_chunk=args.eacs_chunk,
        bf16=(args.device != "cpu"), ckpt_dir=args.ckpt, ckpt_every=args.ckpt_every,
        val_every=500 if args.val_manifest else 0)
    weights = LossWeights(task=1.0, pred=0.5, update_rate=0.1, spectral=0.05)
    tr = Trainer(model, tc, weights)
    print(f"[train] stage={args.stage}, steps={args.steps}, lr={args.lr}, "
          f"grad_accum={args.grad_accum}, device={args.device}")
    tr.fit(cycle(loader), val_loader=val_loader)
    tr.save("final")
    print(f"[done] 训练完成 → {args.ckpt}/final")


if __name__ == "__main__":
    main()
