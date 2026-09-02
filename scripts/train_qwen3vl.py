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


def resolve_device_map(args):
    """--load-to-device 的解析与互锁。返回给 build_cstssm_qwen3vl 的 device_map 或 None。

    互锁的理由不是保守：本脚本刻意把「fork DataLoader worker」排在「CUDA 初始化」之前
    （见下方预热那一段），而 device_map 会在加载底座时就建起 CUDA 上下文，fork 出的子进程
    会继承它。本仓库的 worker 只产出 CPU 张量，实测不会崩，但那是 PyTorch 明确警告的用法，
    所以 num_workers>0 时要出声，让人知道自己在用哪条路。
    """
    if not getattr(args, "load_to_device", False):
        return None
    dev = args.device
    if not str(dev).startswith("cuda"):
        print(f"[model] --load-to-device 对 device={dev!r} 无意义，已忽略")
        return None
    import torch
    if not torch.cuda.is_available():
        print("[model] --load-to-device: CUDA 不可用，已忽略")
        return None
    from cst_ssm.utils import resolve_device
    dev = resolve_device(dev)                 # "cuda" -> "cuda:{LOCAL_RANK}"
    nw = getattr(args, "num_workers", 0) or 0
    if nw > 0:
        print(f"[model] ⚠ --load-to-device 与 --num-workers {nw} 同时开启：底座直接落 {dev} 会在 "
              f"fork worker 之前初始化 CUDA。本仓库的 worker 只产出 CPU 张量，实践中可用，"
              f"但这是 PyTorch 警告过的组合——若出现 worker 卡死/CUDA 重初始化报错，"
              f"请去掉 --load-to-device 或改用 --num-workers 0。")
    print(f"[model] 底座权重直接加载到 {dev}（device_map）")
    return {"": dev}


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
        device_map=resolve_device_map(args),
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
        # 只对 LLM 段注入，与 train_stage2 / train_grounding 对齐：LoRA 的用途是省住
        # 语言模型的显存，CST-SSM 时序段本来就要全参训练，包上 LoRA 会被
        # mark_only_lora_trainable 冻住基座。
        apply_lora(model.llm, r=args.lora_r, alpha=args.lora_alpha)
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
    ap.add_argument("--load-to-device", action="store_true",
                    help="直接把底座权重加载到本 rank 的卡（HF device_map），省掉「先在 CPU 物化 8.8GB 再 .to(cuda)」这一趟。⚠ 它会在 fork DataLoader worker 之前建起 CUDA 上下文，与本脚本刻意安排的顺序冲突，故默认关闭；配 --num-workers 0 时最安全")
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
    ap.add_argument("--max-frames", type=int, default=8192)
    ap.add_argument("--max-text-len", type=int, default=8704)
    ap.add_argument("--max-video-tokens", type=int, default=8192,
                    help="必须 ≥ max-frames，否则 video 占位符被截断、尾部帧特征静默丢弃")
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
    # DDP（本脚本不吃 --config，故只有命令行入口）
    from cst_ssm.utils import add_ddp_args, resolve_ddp
    add_ddp_args(ap)
    args = ap.parse_args()

    # 尽早做：这一步在加载 4B 底座之前。此前本脚本根本没有 ddp 入口，用 torchrun 起它
    # 会让 8 个进程全部落到 cuda:0 而 OOM（run.md §8.5）；现在要么正确分卡，要么当场报错。
    ddp, ddp_backend = resolve_ddp(args.ddp, args.ddp_backend, None,
                                   "scripts/train_qwen3vl.py", device=args.device)

    # ---- 模型 ----
    if args.stand_in:
        model = build_model_standin(args)
    else:
        model = build_model_real(args)
    if args.load:
        from cst_ssm.utils import load_checkpoint
        print(f"[model] 热启: {args.load}")
        # 只热启 CST-SSM 各段是常见用法（LLM 权重来自 HF），故放宽缺失阈值
        load_checkpoint(model, args.load, max_missing_ratio=0.95, tag="qwen3vl-load")

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
    # 防护：真实 manifest + drop_last=True 时，若样本数不足一个 batch，loader 为空（StopIteration）
    if args.manifest and lc.drop_last and len(ds) < args.batch_size:
        print(f"[data] 警告: 样本数({len(ds)}) < batch_size({args.batch_size})，"
              f"drop_last=True 会丢弃所有样本导致 loader 为空；已自动改为 drop_last=False 重建")
        lc.drop_last = False
        loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    # 验证集（可选）
    val_loader = None
    if args.val_manifest:
        vlc = LoaderConfig(
            manifest=args.val_manifest, data_root=args.data_root, mode="feature",
            batch_size=args.batch_size, num_workers=args.num_workers,
            shuffle=False, persistent_workers=args.num_workers > 0,
            max_frames=args.max_frames, max_text_len=args.max_text_len,
            feat_dim=feat_dim)
        val_loader, _ = build_dataloader(vlc, tokenizer=tokenizer)
    print(f"[data] {'manifest: ' + args.manifest if args.manifest else '合成数据'}"
          f"（{len(ds)} 样本, batch={args.batch_size}）"
          f"{' + val: %s' % args.val_manifest if args.val_manifest else ''}")

    # 关键顺序：CUDA 初始化（Trainer 内 model.to(device)）前先 fork DataLoader worker，
    # 避免子进程继承 CUDA 上下文导致死锁；persistent_workers 使 worker 保持存活不再重复 fork。
    if args.num_workers > 0 and args.device != "cpu":
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
                     f"batch_size({args.batch_size}) 是否超过样本数、特征文件是否可读")
        print("[data] worker 就绪，首个 batch 加载成功")

    # ---- 训练 ----
    from cst_ssm.train import Trainer, TrainConfig, LossWeights
    tc = TrainConfig(
        stage=args.stage, lr=args.lr, device=args.device, max_steps=args.steps,
        grad_accum=args.grad_accum, eacs_chunk=args.eacs_chunk,
        bf16=(args.device != "cpu"), ckpt_dir=args.ckpt, ckpt_every=args.ckpt_every,
        val_every=500 if args.val_manifest else 0,
        ddp=ddp, ddp_backend=ddp_backend)
    weights = LossWeights(task=1.0, pred=0.5, update_rate=0.1, spectral=0.05)
    tr = Trainer(model, tc, weights)
    print(f"[train] stage={args.stage}, steps={args.steps}, lr={args.lr}, "
          f"grad_accum={args.grad_accum}, device={args.device}")
    tr.fit(cycle(loader), val_loader=val_loader)
    tr.save("final")
    print(f"[done] 训练完成 → {args.ckpt}/final")


if __name__ == "__main__":
    main()
