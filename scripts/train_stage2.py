#!/usr/bin/env python3
"""Stage-2：端到端任务微调（设计方案 §4.1 阶段②）+ LoRA 省显存。

LLM 段有两条通路，由 --base-model 决定：
  · 给了 --base-model → 接真实 Video-LLM 底座（Qwen3-VL 解码器 + LoRA），**生产路径**，
    同时自动换用 Qwen3-VL 分词器（prompt 里的 <video> 展开成 Lv 个占位符，帧特征注入其中）。
  · 没给          → 自包含 stand-in（VisualConditionedLM，随机初始化的字节级小模型），
    只能验证管线跑通，训不出可用的视频问答模型；脚本会显式告警。

数据加载走统一工厂 cst_ssm.data.build_dataloader（合成/真实 manifest 同口径）：
    # 合成快跑（无数据，smoke；stand-in LLM）
    python scripts/train_stage2.py --device cpu --steps 20
    # 生产训练：真实底座 + 真实数据 + stage-1 热启
    python scripts/train_stage2.py --config configs/default.yaml \
        --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
        --manifest data/manifests/lvb_train.jsonl --data-root data \
        --load checkpoints/stage1/final --device cuda \
        --num-workers 8 --batch-size 4 --steps 5000
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
from cst_ssm.utils import (apply_lora, mark_only_lora_trainable,
                           model_config_from_dict, load_yaml,
                           train_config_from_yaml, loss_weights_from_yaml,
                           loader_config_from_yaml)


def build_model(cfg: CSTSSMConfig, args):
    """按 --base-model 决定 LLM 段。返回 (model, tokenizer)；tokenizer=None 表示用 ByteTokenizer。

    真底座路径把 LLM 段换成 Qwen3-VL 解码器：cfg 的 input_mode / feat_dim / llm 三个
    由底座决定的字段被工厂接管，其余（d_model / branches / 全部消融开关）沿用 YAML。
    """
    if not args.base_model:
        print("[model] 警告: 未指定 --base-model，LLM 段使用自包含 stand-in"
              "（VisualConditionedLM，随机初始化的字节级小模型，vocab=259）。\n"
              "[model]   它不含任何语言知识，只能验证训练管线是否跑通，"
              "**训不出可用的视频问答模型**。\n"
              "[model]   生产训练请加 --base-model Qwen/Qwen3-VL-4B-Instruct。")
        model = CSTSSMModel(cfg)
        if args.lora:
            # 只给 LLM 挂 LoRA：CST-SSM 各段（proj_u/proj_out/fusion 等）是**随机初始化的
            # 新模块**，冻结其基座只训 r=16 低秩增量是无谓的容量限制——LoRA 的前提是
            # 基座已预训练。默认 targets 会命中这些层，必须显式限定作用域。
            apply_lora(model.llm, r=args.lora_r, alpha=args.lora_alpha)
            mark_only_lora_trainable(model)
        return model, None

    from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
    from cst_ssm.data import build_hf_tokenizer
    print(f"[model] 接入真实底座: {args.base_model}"
          f"（lora={args.lora}, r={args.lora_r}, dtype={args.dtype or 'auto'}）")
    # base_cfg 必须传：CPIB / diff_kv / 门控类型等开关都在 CSTSSMModel.__init__ 里生效，
    # 拿到模型后再改 cfg 不会重建任何模块（见工厂 docstring）。LoRA 由工厂内部按
    # QWEN3_LORA_TARGETS 施加，这里不能再调一次 apply_lora，否则重复包裹。
    model = build_cstssm_qwen3vl(
        model_name=args.base_model, base_cfg=cfg, lora=args.lora,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, dtype=args.dtype)
    # 换了 LLM 就必须换分词器：Qwen3VLLanguageModel 把每帧视觉状态注入 video 占位符位置，
    # ByteTokenizer 根本不产出这种 token，沿用它等于 LLM 一帧视频都看不到。
    print(f"[tokenizer] 加载 Qwen3-VL 分词器: {args.base_model}")
    tok = build_hf_tokenizer(args.base_model, max_video_tokens=args.max_video_tokens)
    return model, tok


def preflight_video_tokens(ds, tok, tag: str = "data") -> None:
    """真底座路径自检：prompt 必须含 <video>，且展开后的占位符数 == 帧数。

    这是接底座后最容易静默失效的一环：占位符缺失时 CST-SSM 的视觉状态没有落点，
    LM loss 照样下降（模型退化成纯语言先验），从 loss 曲线上完全看不出来。
    """
    s = ds[0]
    n_frames = int(s["features"].shape[0])
    n_vid = int((s["input_ids"] == tok.video_token_id).sum())
    if n_vid == 0:
        sys.exit(
            f"[{tag}] 错误: manifest 的 prompt 里没有 <video> 占位符。接真实底座时，"
            f"CST-SSM 的逐帧视觉状态要注入到 video 占位符位置，缺了它 LLM 完全看不到视频，"
            f"却仍会正常收敛出一个纯语言模型。请在 prompt 中加入 <video>"
            f"（如 \"<video>\\n请回答：...\"），或用 data_pipeline/src/convert_benchmarks.py 重建 manifest。")
    if n_vid != n_frames:
        print(f"[{tag}] 警告: 首样本 video 占位符 {n_vid} 个 ≠ 帧数 {n_frames}，"
              f"尾部帧特征会被丢弃。多半是 max_text_len / --max-video-tokens 截断，请调大。")
    else:
        print(f"[{tag}] 自检通过: video 占位符数 == 帧数 == {n_frames}")


def build_cfg(y: dict, manifest: str | None, data_root: str) -> CSTSSMConfig:
    # 统一过 align_feat_dim，理由同 train_stage1.build_cfg：feat_dim 由抽特征的视觉塔
    # 唯一决定，YAML 带 model 段时也必须校验，不能直接 return 短路掉。
    if y.get("model"):
        return align_feat_dim(model_config_from_dict(y["model"]), manifest, data_root)
    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96, eacs_chunk=16,
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=4, n_head=4, max_len=64))
    return align_feat_dim(cfg, manifest, data_root)


def _pick(cli, yaml_val, fallback):
    """三级优先级：命令行显式给了就用它，否则用 YAML，最后用脚本内置默认。

    需要这个是因为 argparse 的默认值无法与「用户显式传了同一个值」区分——若直接给
    --steps 一个非 None 默认，YAML 里的 train.max_steps 永远会被它覆盖，等于没接上。
    """
    if cli is not None:
        return cli
    return yaml_val if yaml_val is not None else fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=None, help="覆盖 YAML train.max_steps")
    ap.add_argument("--ckpt", default=None, help="覆盖 YAML train.ckpt_dir")
    # LLM 段：给了 --base-model 走真实底座（生产），否则用自包含 stand-in（仅管线验证）
    ap.add_argument("--base-model", default=None,
                    help="HF Video-LLM 底座名（如 Qwen/Qwen3-VL-4B-Instruct）。"
                         "不给则用 stand-in 小模型，只能验证管线、训不出可用模型")
    ap.add_argument("--dtype", default=None, help="底座 dtype（如 bfloat16），缺省自动")
    ap.add_argument("--max-video-tokens", type=int, default=8192,
                    help="须 ≥ 单样本帧数，否则 video 占位符被截断、尾部帧特征静默丢弃")
    # LoRA：真底座默认开（4B 全参微调不现实），stand-in 默认关（其基座本就是随机初始化的）
    ap.add_argument("--lora", action="store_true", default=None, help="LoRA 冻结基座省显存")
    ap.add_argument("--no-lora", dest="lora", action="store_false", help="关闭 LoRA，全参微调")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--load", default=None, help="从 stage-1 分片检查点热启")
    # 数据加载（均可由 YAML 的 [data] 段提供，命令行优先）
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
    # LoaderConfig 里没有 val_manifest（它是脚本级概念），单独取，避免被当成未知键告警
    val_manifest = _pick(args.val_manifest, ydata.get("val_manifest"), None)
    manifest = _pick(args.manifest, ydata.get("manifest"), None)
    data_root = _pick(args.data_root, ydata.get("data_root"), "")
    mode = _pick(args.mode, ydata.get("mode"), "feature")
    if args.lora is None:      # 未显式指定：真底座默认开 LoRA，stand-in 默认关
        args.lora = bool(args.base_model)

    cfg = build_cfg(y, manifest, data_root)
    model, tokenizer = build_model(cfg, args)
    if args.base_model and model.cfg.feat_dim != cfg.feat_dim:
        # 工厂会用 --base-model 视觉塔的 out_hidden_size 覆盖 feat_dim。它与 manifest 推断出的
        # 维度不一致，只能说明 .npz 不是这个底座抽的——继续跑必然在第一个 Linear 崩，
        # 且崩在 CUDA 上更难读，不如在这里就说清楚。
        sys.exit(
            f"[model] 错误: --base-model 的视觉塔输出维 {model.cfg.feat_dim} 与 manifest 特征维 "
            f"{cfg.feat_dim} 不一致，说明特征缓存不是用 {args.base_model} 抽的。请用同一底座重抽：\n"
            f"  python -m data_pipeline.src.extract_features --model {args.base_model} ...")
    if args.load:
        from cst_ssm.utils import load_checkpoint
        # 真底座路径下 LLM 权重来自 HF、LoRA 是新增参数，stage-1 检查点只覆盖 CST-SSM 各段，
        # 缺失比例天然极高（4B 底座下 >95%），用默认阈值会误判成"架构不匹配"直接 raise。
        load_checkpoint(model, args.load, tag="stage2-load",
                        max_missing_ratio=0.99 if args.base_model else 0.5)

    # TrainConfig / LossWeights：先吃 YAML 的 [train] / [loss]，再让命令行覆盖。
    # stage / device / bf16 由脚本与命令行决定，不从 YAML 取（放进 skip 免告警）。
    tc = train_config_from_yaml(y, skip={"stage", "device", "bf16", "eacs_chunk"})
    weights = loss_weights_from_yaml(y)
    steps = _pick(args.steps, None, tc.max_steps if y.get("train") else 20)
    tc.stage = "finetune"
    tc.device = args.device
    tc.bf16 = (args.device != "cpu")
    tc.max_steps = steps
    tc.ckpt_dir = _pick(args.ckpt, None, tc.ckpt_dir if y.get("train") else "checkpoints/stage2")
    # eacs_chunk 保持 None：沿用模型配置（YAML model.eacs_chunk 或 build_cfg 的兜底 16）。
    # YAML 的 train.eacs_chunk 是历史遗留的重复键，接上会与 model 段互相打架，故列入 skip。
    tc.eacs_chunk = None

    batch_size = _pick(args.batch_size, ydata.get("batch_size"), 2)
    num_workers = _pick(args.num_workers, ydata.get("num_workers"), 0)
    lc = loader_config_from_yaml(
        y, skip={"manifest", "val_manifest", "data_root", "mode",
                 "batch_size", "num_workers", "pin_memory", "drop_last",
                 "persistent_workers"})
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
              f"task={weights.task}, pred={weights.pred}, batch={lc.batch_size})")
    loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    # 防护：真实 manifest + drop_last=True 时，若样本数不足一个 batch，loader 为空（StopIteration）
    if manifest and lc.drop_last and len(ds) < batch_size:
        print(f"[data] 警告: 样本数({len(ds)}) < batch_size({batch_size})，"
              f"drop_last=True 会丢弃所有样本导致 loader 为空；已自动改为 drop_last=False 重建")
        lc.drop_last = False
        loader, ds = build_dataloader(lc, tokenizer=tokenizer)
    # 验证集（可选）
    val_loader = None
    if val_manifest:
        vlc = LoaderConfig(manifest=val_manifest, data_root=data_root, mode=mode,
                           batch_size=batch_size, num_workers=num_workers,
                           shuffle=False, persistent_workers=num_workers > 0,
                           max_frames=lc.max_frames, max_text_len=lc.max_text_len,
                           feat_dim=cfg.feat_dim)
        val_loader, _ = build_dataloader(vlc, tokenizer=tokenizer)
    print(f"[data] {'真实 manifest: %s' % manifest if manifest else '合成数据'}"
          f"（{len(ds)} 样本, batch={batch_size}, workers={num_workers}）"
          f"{' + val: %s' % val_manifest if val_manifest else ''}")
    if tokenizer is not None and manifest:
        preflight_video_tokens(ds, tokenizer)

    # 关键顺序：CUDA 初始化前先 fork DataLoader worker，避免子进程继承 CUDA 上下文导致死锁。
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
    tr.fit(cycle(loader), val_loader=val_loader)        # cycle：max_steps 超一个 epoch 也不提前停
    tr.save("final")


if __name__ == "__main__":
    main()
