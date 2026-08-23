#!/usr/bin/env python3
"""训练定位打分头 GroundingHead（供 CSG/VFR 亚帧定位；产出 infer_grounding 加载的 --ckpt）。

在帧时刻用连续读出 + query 条件头打分，对 [gt_start,gt_end] 内/外做 BCE 学习相关度曲线。
数据走统一工厂：--manifest（含 grounding 标注 gt_start/gt_end 的 jsonl）或合成定位数据兜底。

    # 合成快跑（无数据，验证可训；stand-in LLM）
    python scripts/train_grounding.py --device cpu --steps 50
    # 生产训练：真实底座 + 真实定位数据（Charades-STA/ActivityNet 转成 VideoSample+gt 的 manifest）
    python scripts/train_grounding.py --config configs/default.yaml \
        --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
        --manifest data/manifests/charades_train.jsonl --data-root data \
        --load checkpoints/stage2/final --device cuda --steps 20000
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.data import LoaderConfig, build_dataloader, cycle, align_feat_dim
from cst_ssm.train import Trainer, TrainConfig
from cst_ssm.utils import (model_config_from_dict, load_yaml,
                           train_config_from_yaml, loss_weights_from_yaml,
                           loader_config_from_yaml)


def _pick(cli, yaml_val, fallback):
    """三级优先级：命令行 > YAML > 脚本内置默认（详见 train_stage2._pick）。"""
    if cli is not None:
        return cli
    return yaml_val if yaml_val is not None else fallback


def build_model(cfg: CSTSSMConfig, args):
    """按 --base-model 决定 LLM 段。返回 (model, tokenizer)；tokenizer=None 表示用 ByteTokenizer。

    定位任务同样依赖 LLM：query 文本经 _embed_text → GroundingHead.encode_query 才成为
    条件向量（cst_ssm_model.py:474）。stand-in 是随机初始化的字节级模型，其嵌入是噪声，
    "query 条件"这一半会退化成常数——BCE 照样降，CSG/VFR 指标却拿不到 query 区分度。
    """
    if not args.base_model:
        print("[model] 警告: 未指定 --base-model，LLM 段使用自包含 stand-in，"
              "query 嵌入为随机噪声，定位头学不到 query 区分度。\n"
              "[model]   仅供管线验证；生产训练请加 --base-model Qwen/Qwen3-VL-4B-Instruct。")
        model = CSTSSMModel(cfg)
        if args.lora:
            from cst_ssm.utils import apply_lora, mark_only_lora_trainable
            # 只给 LLM 挂 LoRA：CST-SSM 各段（proj_u/proj_out/fusion 等）是**随机初始化的
            # 新模块**，冻结其基座只训 r=16 低秩增量是无谓的容量限制——LoRA 的前提是
            # 基座已预训练。默认 targets 会命中这些层，必须显式限定作用域。
            apply_lora(model.llm, r=args.lora_r, alpha=args.lora_alpha)
            mark_only_lora_trainable(model)
        return model, None

    from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
    from cst_ssm.data import build_hf_tokenizer
    print(f"[model] 接入真实底座: {args.base_model}（lora={args.lora}, r={args.lora_r}）")
    # 工厂内部已按 QWEN3_LORA_TARGETS 施加 LoRA，且 mark_only_lora_trainable 只冻 "llm."
    # 前缀——grounding_head / vision / temporal / projector 仍可训，无需再手工放开。
    model = build_cstssm_qwen3vl(
        model_name=args.base_model, base_cfg=cfg, lora=args.lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, dtype=args.dtype)
    print(f"[tokenizer] 加载 Qwen3-VL 分词器: {args.base_model}")
    return model, build_hf_tokenizer(args.base_model, max_video_tokens=args.max_video_tokens)


def build_cfg(y, manifest, data_root) -> CSTSSMConfig:
    # 两条分支都过 align_feat_dim，理由同 train_stage1.build_cfg：feat_dim 由抽特征的
    # 视觉塔唯一决定，YAML 带 model 段时也不能跳过校验。
    if y.get("model"):
        d = dict(y["model"])
        d["grounding_head"] = True                     # 定位训练必须开头
        return align_feat_dim(model_config_from_dict(d), manifest, data_root)
    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96, grounding_head=True,
                       eacs_chunk=16,   # 定位训练走 run_with_commits，同样按块做梯度检查点
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
    return align_feat_dim(cfg, manifest, data_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 YAML train.max_steps")
    ap.add_argument("--ckpt", default=None, help="覆盖 YAML train.ckpt_dir")
    # LLM 段：给了 --base-model 走真实底座（生产），否则用自包含 stand-in（仅管线验证）
    ap.add_argument("--base-model", default=None,
                    help="HF Video-LLM 底座名（如 Qwen/Qwen3-VL-4B-Instruct）。"
                         "不给则 query 嵌入来自随机初始化的 stand-in，定位头学不到 query 区分度")
    ap.add_argument("--dtype", default=None, help="底座 dtype（如 bfloat16），缺省自动")
    ap.add_argument("--max-video-tokens", type=int, default=8192)
    ap.add_argument("--lora", action="store_true", default=None,
                    help="LoRA 冻结基座，只训头+投影（真底座默认开）")
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--load", default=None, help="从 stage-2 检查点热启（复用视觉/时序/LLM）")
    ap.add_argument("--manifest", default=None, help="含 gt_start/gt_end 的定位 manifest；缺省合成")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="覆盖 YAML train.lr")
    args = ap.parse_args()

    # ---- 配置解析：YAML 四段全部消费（model / train / loss / data），命令行覆盖 YAML ----
    y = load_yaml(args.config) if args.config else {}
    ydata = y.get("data") or {}
    manifest = _pick(args.manifest, ydata.get("manifest"), None)
    data_root = _pick(args.data_root, ydata.get("data_root"), "")
    if args.lora is None:      # 未显式指定：真底座默认开 LoRA，stand-in 默认关
        args.lora = bool(args.base_model)

    cfg = build_cfg(y, manifest, data_root)
    model, tokenizer = build_model(cfg, args)
    if args.base_model and model.cfg.feat_dim != cfg.feat_dim:
        sys.exit(
            f"[model] 错误: --base-model 的视觉塔输出维 {model.cfg.feat_dim} 与 manifest 特征维 "
            f"{cfg.feat_dim} 不一致，说明特征缓存不是用 {args.base_model} 抽的。请用同一底座重抽：\n"
            f"  python -m data_pipeline.src.extract_features --model {args.base_model} ...")
    if args.load:
        from cst_ssm.utils import load_checkpoint
        # 定位头是新参数，本来就会缺失；接真底座时 LLM 权重来自 HF，缺失比例更高
        load_checkpoint(model, args.load, tag="grounding-load",
                        max_missing_ratio=0.99 if args.base_model else 0.9)
    if args.lora and not args.base_model:
        for p in model.grounding_head.parameters():    # 头始终可训
            p.requires_grad_(True)

    tc = train_config_from_yaml(y, skip={"stage", "device", "bf16", "eacs_chunk"})
    weights = loss_weights_from_yaml(y)
    steps = _pick(args.steps, None, tc.max_steps if y.get("train") else 50)
    tc.stage = "grounding"
    tc.device = args.device
    tc.bf16 = (args.device != "cpu")
    tc.max_steps = steps
    # 定位头是新参数，默认 lr 比 stage-2 大一档；YAML 的 train.lr 若存在则以它为准
    tc.lr = _pick(args.lr, y.get("train", {}).get("lr"), 1e-3)
    tc.ckpt_dir = _pick(args.ckpt, None,
                        tc.ckpt_dir if y.get("train") else "checkpoints/grounding")
    tc.eacs_chunk = None            # 沿用模型配置，理由同 train_stage2

    batch_size = _pick(args.batch_size, ydata.get("batch_size"), 2)
    num_workers = _pick(args.num_workers, ydata.get("num_workers"), 0)
    lc = loader_config_from_yaml(
        y, skip={"manifest", "val_manifest", "data_root", "mode", "batch_size",
                 "num_workers", "pin_memory", "drop_last", "persistent_workers"})
    lc.manifest, lc.data_root = manifest, data_root
    lc.batch_size, lc.num_workers = batch_size, num_workers
    lc.persistent_workers = num_workers > 0
    lc.feat_dim = cfg.feat_dim
    lc.synth_grounding = (manifest is None)
    lc.drop_last = bool(manifest)
    lc.synth_n = steps * batch_size + 8
    if y:
        print(f"[cfg] YAML 四段已全部消费：model/train/loss/data "
              f"(lr={tc.lr}, grad_accum={tc.grad_accum}, max_steps={tc.max_steps}, "
              f"batch={lc.batch_size})")
    loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    # 防护：真实 manifest + drop_last=True 时，若样本数不足一个 batch，loader 为空（StopIteration）
    if manifest and lc.drop_last and len(ds) < batch_size:
        print(f"[data] 警告: 样本数({len(ds)}) < batch_size({batch_size})，"
              f"drop_last=True 会丢弃所有样本导致 loader 为空；已自动改为 drop_last=False 重建")
        lc.drop_last = False
        loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    print(f"[data] {'真实定位 manifest: %s' % manifest if manifest else '合成定位数据'}"
          f"（{len(ds)} 样本, grounding_head={cfg.grounding_head}）")

    # 关键顺序：CUDA 初始化前先 fork DataLoader worker，避免子进程继承 CUDA 上下文导致死锁。
    # persistent_workers=True 使 worker 保持存活，后续 fit() 不再重复 fork。
    if num_workers > 0 and args.device != "cpu":
        print("[data] 预热 DataLoader worker（CUDA 初始化前派生子进程）...")
        try:
            _warm = iter(loader)
            next(_warm)
            del _warm
        except StopIteration:
            sys.exit(f"[data] 错误: loader 为空，无法预热。请检查 manifest 是否非空、"
                     f"batch_size({batch_size}) 是否超过样本数、特征文件是否可读")
        print("[data] worker 就绪，首个 batch 加载成功")

    tr = Trainer(model, tc, weights)
    tr.fit(cycle(loader))
    tr.save("final")
    print(f"[done] 定位头训练完成 → {tc.ckpt_dir}/final（infer_grounding --ckpt 加载）")


if __name__ == "__main__":
    main()
