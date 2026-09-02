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


def _build_cfg(args):
    """从检查点同目录的 config.yaml 还原架构；没有就用默认（并告警）。"""
    from cst_ssm.models import CSTSSMConfig
    from cst_ssm.utils import model_config_from_dict, load_yaml
    # 兼容两种保存风格：嵌套 model: 字段 / 扁平模型字段。
    # 若直接把整个 yaml 根传给 model_config_from_dict，嵌套风格下所有字段会静默回落默认值，
    # 导致架构与权重不匹配（strict=False 静默加载空壳）。
    cfg_path = args.config or os.path.join(os.path.dirname(args.ckpt or "."), "config.yaml")
    if cfg_path and os.path.exists(cfg_path):
        y = load_yaml(cfg_path)
        cfg_dict = y.get("model") if isinstance(y.get("model"), dict) else y
        print(f"[eval] 架构来自 {cfg_path}")
        return model_config_from_dict(cfg_dict)
    print("[eval][warn] 未找到 config.yaml，使用默认架构——若与训练不一致将静默失配")
    return CSTSSMConfig(input_mode="feature", feat_dim=768, d_model=768)


def _preflight_ckpt(args):
    """加载前判定：这份检查点是不是"接了真实底座训出来的"，而调用方却没给 --base-model。

    这是本脚本此前最严重的一处静默失效：load_model 只会建 stand-in
    （VisualConditionedLM，随机初始化的字节级小模型），把 Qwen3-VL 训出来的 stage-2
    检查点丢进来，LLM 段的键会全部落成 unexpected，模型带着一个**随机初始化的语言模型**
    照常出分，只打一行 warn。MCQ 准确率于是变成纯噪声，而曲线看不出任何异常。

    两个判据，命中任一就直接退出（宁可崩，也别报一个假数）：
      ① index.metadata.trainable_only —— 保存时省掉了可从 HF 取回的冻结底座；
      ② 检查点里 llm.* 的键少得离谱，但模型架构却期待一整个 LLM 段。
    """
    from cst_ssm.utils import checkpoint_metadata
    if args.base_model or not args.ckpt:
        return
    meta = checkpoint_metadata(args.ckpt)
    if meta.get("trainable_only"):
        sys.exit(
            f"[eval] 错误: {args.ckpt} 是**只存可训练权重**的检查点"
            f"（index.metadata.trainable_only=True），说明它是接真实底座训出来的，"
            f"冻结的 LLM 权重要由 from_pretrained 取回。\n"
            f"[eval]   现在没给 --base-model，本脚本会建一个随机初始化的 stand-in LLM，"
            f"然后照常算出一个**毫无意义的准确率**。\n"
            f"[eval]   修法: 加 --base-model Qwen/Qwen3-VL-4B-Instruct（与训练时同一个）。")


def load_model(args):
    """加载训练好的模型。返回 (model, tokenizer)；tokenizer=None 表示用 ByteTokenizer。"""
    if args.stand_in:
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.modules.llm_interface import LLMConfig
        cfg = CSTSSMConfig(
            input_mode="feature", feat_dim=64, d_model=96,
            cpib_distill=True, cpi_modulation=0.5,
            llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=256))
        print("[eval] stand-in 模型")
        return CSTSSMModel(cfg).to(args.device).eval(), None

    from cst_ssm.utils import load_checkpoint
    from cst_ssm.utils.checkpoint import CRITICAL_SHAPE_PREFIXES
    # 评测的缺失比例默认值：卡到 2%，理由见 --max-missing-ratio 的 help。
    # 放在这里而不是 argparse 的 default，是为了让 default=None 能区分"没给"与"给了 0.02"。
    _DEFAULT_MAX_MISSING_RATIO = 0.02
    cfg = _build_cfg(args)
    _preflight_ckpt(args)

    if args.base_model:
        # 生产路径：与 train_stage2 / train_grounding 同一个工厂，架构逐字一致。
        # base_cfg 必须传：CPIB / diff_kv / 门控类型等开关都在 CSTSSMModel.__init__ 里生效。
        from cst_ssm.integrations.qwen3_vl import build_cstssm_qwen3vl
        from cst_ssm.data import build_hf_tokenizer
        from cst_ssm.models import assert_config_applied
        print(f"[eval] 接入真实底座: {args.base_model}（lora={args.lora}, r={args.lora_r}）")
        model = build_cstssm_qwen3vl(
            model_name=args.base_model, base_cfg=cfg, lora=args.lora,
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, dtype=args.dtype)
        assert_config_applied(model, cfg)     # 核对消融开关真的建进了模块
        tok = build_hf_tokenizer(args.base_model, max_video_tokens=args.max_video_tokens)
        if args.ckpt:
            # 冻结底座来自 HF、LoRA 是新增参数，缺失比例天然极高（4B 下 >95%）
            load_checkpoint(model, args.ckpt, max_missing_ratio=0.99, tag="eval")
        else:
            print("[eval][warn] 未给 --ckpt：CST-SSM 各段是随机初始值，指标不可采信")
        return model.to(args.device).eval(), tok

    from cst_ssm.models import CSTSSMModel
    model = CSTSSMModel(cfg)
    if args.ckpt:
        # --max-missing-ratio 未显式给（None）→ 用默认 2% 并保留关键参数形状守卫；
        # 显式给了 → 用户已经声明"我知道这份检查点和配置不是一套，强跑"，两道守卫一起放开。
        _forced = args.max_missing_ratio is not None
        if _forced:
            print(f"[eval][warn] 显式放宽 max_missing_ratio={args.max_missing_ratio} "
                  f"→ 关键参数（vision.*）形状不匹配也一并放行；本次结果不可作为上报数据")
        load_checkpoint(model, args.ckpt,
                        max_missing_ratio=(args.max_missing_ratio
                                           if _forced else _DEFAULT_MAX_MISSING_RATIO),
                        tag="eval",
                        critical_prefixes=() if _forced else CRITICAL_SHAPE_PREFIXES)
    print(f"[eval] 加载模型: {args.ckpt}")
    return model.to(args.device).eval(), None


def _mcq_answer_index(row: dict):
    """解析 manifest 行的正确答案在 options 中的下标；无法解析返回 None。"""
    opts = row.get("options")
    ans = row.get("answer")
    if not opts or ans is None:
        return None
    if isinstance(ans, int):
        return ans if 0 <= ans < len(opts) else None
    s = str(ans).strip()
    if len(s) == 1 and s.isalpha():                    # "A"/"B"/... 字母答案
        idx = ord(s.upper()) - ord("A")
        return idx if 0 <= idx < len(opts) else None
    for j, o in enumerate(opts):                       # 答案文本与选项完全匹配
        if str(o).strip() == s:
            return j
    return None


def _letter_token_id(tok, letter: str):
    """取字母的 token id（兼容 ByteTokenizer 与 HF tokenizer）。"""
    ids = tok.encode(letter)
    # 用 `is not None` 而非 `or`：BOS 合法取 0（多数 HF 分词器就是 0），`or` 会把它当假值
    # 吞掉并落到 bos_id（可能是 None），下面的过滤就删不掉 BOS，MCQ 的字母 token id 取错。
    bos = getattr(tok, "BOS", None)
    if bos is None:
        bos = getattr(tok, "bos_id", None)
    ids = [i for i in ids if i != bos]
    return ids[0] if ids else None


def _eval_autocast(args):
    """评测用的 autocast 上下文。--bf16 才开，默认关 = 既有数值行为不变。

    接 `--dtype bfloat16` 的底座时**不开也能跑**（LoRALinear 会做 dtype 对齐），
    开了主要是省显存与提速；stand-in / fp32 检查点上开它会改变数值，故不设默认。
    """
    from cst_ssm.utils import autocast_ctx
    dev = "cuda" if str(args.device).startswith("cuda") else "cpu"
    return autocast_ctx(bool(getattr(args, "bf16", False)), device_type=dev)


def eval_qa(model, args, tokenizer=None):
    """评测 QA：manifest 含 options+answer 时算真 MCQ 准确率；否则回退 teacher-forcing
    逐 token 命中率并显式标注为 token_acc（非 QA 准确率）。"""
    from cst_ssm.data import LoaderConfig, build_dataloader

    # feat_dim 必须跟着**模型实际架构**走：写死 768 时，接 Qwen3-VL（视觉塔 2560）
    # 只会在第一个 Linear 崩，而合成数据路径下更糟——维度对不上却不报错，喂进去的是噪声。
    lc = LoaderConfig(
        manifest=args.manifest, data_root=args.data_root, mode="feature",
        batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, drop_last=False,
        max_frames=args.max_frames, max_text_len=args.max_text_len,
        feat_dim=model.cfg.feat_dim, synth_n=args.steps * args.batch_size)
    loader, ds = build_dataloader(lc, tokenizer=tokenizer)

    # MCQ 模式判定：真实 manifest 且每行都有 options（shuffle=False 保证 batch 顺序=行序）
    rows = getattr(ds, "rows", None)
    mcq = bool(rows) and all(r.get("options") for r in rows)

    correct, total = 0, 0
    n_batches = 0
    n_unaligned = 0     # 无法定位到 manifest 行的样本（占位/替换样本）
    letter_cache: dict = {}
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            if n_batches >= args.steps:
                break
            n_batches += 1
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with _eval_autocast(args):
                out = model(batch)
            logits = out.get("logits")
            labels = batch.get("labels")
            if logits is None or labels is None:
                continue
            B = labels.shape[0]
            if mcq:
                # 真 MCQ：在答案首 token 位置比较各选项字母的概率质量。
                # 用样本自带的 row_index 定位 manifest 行，**不能**按 batch 顺序推算——
                # VideoTemporalDataset 在取样失败时会随机替换成别的行，按顺序推算就会
                # 拿 A 的预测去配 B 的答案，准确率变成噪声且毫无迹象。
                rid = batch.get("row_index")
                for i in range(B):
                    row_idx = int(rid[i]) if rid is not None else -1
                    if row_idx < 0 or row_idx >= len(rows):
                        n_unaligned += 1
                        continue
                    gt = _mcq_answer_index(rows[row_idx])
                    if gt is None:
                        continue
                    valid_pos = (labels[i] != -100).nonzero(as_tuple=True)[0]
                    if valid_pos.numel() == 0 or valid_pos[0] == 0:
                        continue
                    pred_pos = int(valid_pos[0].item()) - 1     # 下一 token 预测位置
                    n_opt = len(rows[row_idx]["options"])
                    if n_opt not in letter_cache:               # 字母 token id 只解一次
                        ids = [_letter_token_id(ds.tok, chr(ord("A") + j)) for j in range(n_opt)]
                        letter_cache[n_opt] = ids if all(t is not None for t in ids) else None
                    ids = letter_cache[n_opt]
                    if ids is None:
                        continue
                    sub = logits[i, pred_pos, ids]
                    if int(sub.argmax().item()) == gt:
                        correct += 1
                    total += 1
            else:
                # 回退：teacher-forcing 逐 token 命中率（注意：这不是 QA 准确率）
                valid = labels != -100
                if valid.any():
                    preds = logits.argmax(dim=-1)
                    correct += (preds[valid] == labels[valid]).sum().item()
                    total += valid.sum().item()

    elapsed = time.time() - t0
    acc = correct / max(total, 1)
    metric_name = "MCQ 准确率" if mcq else "teacher-forcing token 命中率（非 QA 准确率！）"
    fs = getattr(ds, "failure_stats", None)
    print(f"\n{'='*50}")
    print("[QA 评测结果]")
    print(f"  指标类型: {metric_name}")
    print(f"  结果: {acc:.4f} ({correct}/{total})")
    print(f"  耗时: {elapsed:.1f}s")
    if n_unaligned:
        print(f"  ⚠ {n_unaligned} 个样本无法定位 manifest 行（取样失败被替换/占位），已排除计数")
    if fs and fs["failures"]:
        print(f"  ⚠ 数据取样失败 {fs['failures']} 次（占位样本 {fs['fallbacks']} 次）/ "
              f"共 {fs['total_rows']} 行 —— 上述指标基于**不完整**的数据，请先修数据再采信")
    print(f"{'='*50}")
    key = "qa_acc" if mcq else "token_acc"
    res = {key: acc, "total_counted": total, "elapsed": elapsed, "unaligned": n_unaligned}
    if fs:
        res["data_failures"] = fs["failures"]
        res["data_fallbacks"] = fs["fallbacks"]
    return res


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
        ds = SyntheticVideoDataset(n=4, L=n_frames, P=4, d=model.cfg.feat_dim)
        loader = DataLoader(ds, batch_size=1, collate_fn=lambda b: collate_fn(b, pad_id=256))

        # 预热
        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad(), _eval_autocast(args):
                _ = model(batch)
            break

        # 计时
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        update_rates = []
        for batch in loader:
            batch = {k: (v.to(args.device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            with torch.no_grad(), _eval_autocast(args):
                out = model(batch)
            update_rates.append(out["update_rate"].item())
        elapsed = (time.time() - t0) * 1000 / len(loader)

        mem = 0
        if str(args.device).startswith("cuda"):
            torch.cuda.synchronize()
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
    ap.add_argument("--max-frames", type=int, default=8192,
                    help="与 DataConfig.max_frames 一致；调小会在评测时二次抽稀，"
                         "静默降低时间分辨率并直接压低定位指标")
    ap.add_argument("--max-text-len", type=int, default=8704)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--stand-in", action="store_true")
    # ---- 真实底座（生产模型的唯一评测通路）----
    # 不给 --base-model 时本脚本只会建 stand-in LLM（随机初始化），把接 Qwen3-VL 训出来的
    # 检查点丢进来会算出一个毫无意义却看不出异常的准确率。见 _preflight_ckpt 的硬拦截。
    ap.add_argument("--base-model", default=None,
                    help="HF Video-LLM 底座名，必须与训练时**同一个**"
                         "（如 Qwen/Qwen3-VL-4B-Instruct）")
    ap.add_argument("--dtype", default=None, help="底座 dtype（如 bfloat16），缺省自动")
    ap.add_argument("--lora", action="store_true", default=None,
                    help="底座挂 LoRA（须与训练时一致；给了 --base-model 时默认开）")
    ap.add_argument("--no-lora", dest="lora", action="store_false")
    ap.add_argument("--lora-r", type=int, default=64)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--max-video-tokens", type=int, default=8192,
                    help="须 ≥ 单样本帧数，否则 video 占位符被截断、尾部帧特征静默丢弃")
    ap.add_argument("--config", default=None,
                    help="模型架构 YAML；缺省找 --ckpt 同目录的 config.yaml")
    ap.add_argument("--bf16", action="store_true",
                    help="评测走 bf16 autocast（省显存/提速）。默认关 = 数值行为与既往一致")
    ap.add_argument("--max-missing-ratio", type=float, default=None,
                    help="不接底座时允许的检查点缺失比例上限（默认 2%%）。评测与热启不同："
                         "热启缺一部分是常态，评测缺权重就意味着在用随机初始值出分，"
                         "所以这里卡得很紧；确有意为之时再放宽。"
                         "**显式传本参数即视为强跑**：连 vision.* 这类关键参数的形状不匹配"
                         "也一并放行（等价 load_checkpoint(critical_prefixes=())），"
                         "此时出的分只能自用，不能作为结果上报")
    ap.add_argument("--output", default=None, help="结果输出 JSON 路径")
    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，回退 CPU")
        args.device = "cpu"
    if args.lora is None:                 # 未显式指定：真底座默认开，其余默认关
        args.lora = bool(args.base_model)
    if args.bf16 and args.device == "cpu":
        print("[eval][warn] CPU 上的 bf16 autocast 只用于验证逻辑，不代表 GPU 数值")

    model, tokenizer = load_model(args)

    if args.mode == "qa":
        results = eval_qa(model, args, tokenizer=tokenizer)
    else:
        results = eval_efficiency(model, args)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[saved] → {args.output}")


if __name__ == "__main__":
    main()
