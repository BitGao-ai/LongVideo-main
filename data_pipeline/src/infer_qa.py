"""四基准 QA 推理：读 convert_benchmarks 的 manifest + 特征 → pred.jsonl（闭合 convert→infer→qa_eval）。

打分：**逐选项似然**——对每个候选字母 L(A/B/..)，把 (prompt, L) 喂模型（视觉状态经 EACS 压缩后由
LLM 交叉注意力条件化），取答案位的 -loss 作分数，argmax 选项即预测。等价于多选题的 loglikelihood
判据（lmms-eval 惯例），确定性、无需生成解码。

诚实边界：默认构建**未训练** CST-SSM（LLM stand-in），pred 无语义、仅验证链路/格式；真实数值
需 --ckpt 注入训练权重。打分/选择逻辑（score_options/pick_option）与模型解耦，可单测。

用法：
  python -m data_pipeline.src.infer_qa --manifest videomme.jsonl --data-root data \
      --ckpt ckpt_final.pt --out cstssm_videomme.jsonl
  # 评测：python -m data_pipeline.src.qa_eval --manifest videomme.jsonl --pred cstssm_videomme.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import string

import numpy as np

from .validate_dataset import _load_feat_ts

LETTERS = string.ascii_uppercase


# ------------------------------ 纯函数（可单测）------------------------------
def pick_option(scores) -> tuple:
    """分数列表 → (index, letter)，argmax（并列取首）。"""
    if not len(scores):
        return -1, ""
    idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return idx, LETTERS[idx]


def build_example(tok, prompt: str, answer: str, max_len: int):
    """拼 (prompt, answer)，labels 对 prompt 段打 -100；**超长截 prompt 头以保住 answer 被打分**。"""
    p = [tok.BOS] + list(prompt.encode("utf-8"))
    a = list(answer.encode("utf-8")) + [tok.EOS]
    if len(p) + len(a) > max_len:
        p = p[: max(1, max_len - len(a))]
    ids = p + a
    labels = [-100] * len(p) + a
    return ids, labels


# ------------------------------ 模型打分 ------------------------------
def _make_batch(feats, ts, ids, labels, device="cpu"):
    import torch
    f = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(device)  # (1,L,P,d)
    t = torch.from_numpy(np.ascontiguousarray(ts)).float().unsqueeze(0).to(device)      # (1,L)
    L = f.shape[1]
    return dict(
        features=f, timestamps=t,
        frame_mask=torch.ones(1, L, dtype=torch.bool, device=device),
        input_ids=torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0),
        labels=torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0),
        attention_mask=torch.ones(1, len(ids), dtype=torch.bool, device=device),
    )


def score_options(model, tok, feats, ts, prompt: str, options, max_len: int = 1024,
                  score_mode: str = "letter", device: str = "cpu"):
    """对每个选项算似然分数（-loss，越大越可能）。score_mode: letter=打分字母 / text=打分选项原文。

    视觉侧与候选选项无关：若 model 支持 encode_visual（CSTSSMModel），只跑**一次** O(L)
    的视觉+时序扫描，之后只对 LLM 循环选项；否则回退成逐选项整体前向（兼容注入的 stub）。
    4–5 个选项 × 四个基准下，这是 4–5 倍的重复扫描。
    """
    import torch
    n = len(options)
    answers = [LETTERS[i] if score_mode == "letter" else str(options[i]) for i in range(n)]
    examples = [build_example(tok, prompt, a, max_len) for a in answers]

    fast = hasattr(model, "encode_visual") and hasattr(model, "llm")
    if not fast:                                        # 通用回退：逐选项整体前向
        scores = []
        for ids, labels in examples:
            batch = _make_batch(feats, ts, ids, labels, device)
            with torch.no_grad():
                out = model(batch)
            loss = out["loss"] if isinstance(out, dict) else out
            scores.append(-float(loss))
        return scores

    # 快路：视觉只算一次
    probe = _make_batch(feats, ts, examples[0][0], examples[0][1], device)
    with torch.no_grad():
        visual_states, _ = model.encode_visual(probe)
    scores = []
    for ids, labels in examples:
        iid = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        lab = torch.tensor(labels, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            out = model.llm(input_ids=iid, visual_states=visual_states,
                            attention_mask=torch.ones_like(iid, dtype=torch.bool),
                            visual_mask=probe["frame_mask"], labels=lab)
        scores.append(-float(out["loss"]))
    return scores


def build_default_model(feat_dim: int, d_model: int, ckpt: str | None, device: str,
                        config: str | None = None):
    import torch
    from cst_ssm.models import CSTSSMModel, CSTSSMConfig
    from cst_ssm.modules.llm_interface import LLMConfig
    if config:                                          # 训练同款 config → 架构对齐才能正确加载 --ckpt
        from cst_ssm.utils import model_config_from_dict, load_yaml
        cfg = model_config_from_dict(load_yaml(config)["model"])
    else:
        cfg = CSTSSMConfig(input_mode="feature", feat_dim=feat_dim, d_model=d_model,
                           llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=1024))
    model = CSTSSMModel(cfg).to(device).eval()
    if ckpt:
        from cst_ssm.utils import load_checkpoint
        load_checkpoint(model, ckpt, tag="infer_qa")
    return model


# ------------------------------ 批推理（可注入 model/tok 便于测试）------------------------------
def infer_manifest(rows, model, tok, data_root: str, max_len: int = 1024,
                   score_mode: str = "letter", device: str = "cpu"):
    out, fails = [], []
    for r in rows:
        try:
            options = r.get("options", [])
            if not options:                             # 无选项（开放式）→ 跳过似然打分
                continue
            feats, ts = _load_feat_ts(r["feature_ref"], data_root)
            feats = np.asarray(feats, dtype=np.float32)
            scores = score_options(model, tok, feats, ts, r.get("prompt", ""),
                                   options, max_len, score_mode, device)
            _, letter = pick_option(scores)
            out.append({"query_id": r["query_id"], "pred": letter})
        except Exception as e:
            fails.append({"query_id": r.get("query_id"), "error": str(e)})
    return out, fails


def run(args):
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    # 从首个可读特征推 feat_dim
    feat_dim = args.feat_dim
    if feat_dim is None:
        for r in rows:
            try:
                f, _ = _load_feat_ts(r["feature_ref"], args.data_root)
                feat_dim = int(np.asarray(f).shape[-1]); break
            except Exception:
                continue
    from cst_ssm.data.dataset import ByteTokenizer
    tok = ByteTokenizer()
    model = build_default_model(feat_dim or 3584, args.d_model, args.ckpt, args.device, args.config)
    preds, fails = infer_manifest(rows, model, tok, args.data_root, args.max_len,
                                  args.score_mode, args.device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if fails:
        with open(args.out + ".failed", "w", encoding="utf-8") as f:
            for x in fails:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"[infer_qa] 出预测 {len(preds)} 条 → {args.out}{'；失败 %d' % len(fails) if fails else ''}")
    if not args.ckpt:
        print("[infer_qa] ⚠ 未加载权重：CST-SSM 为未训练 stand-in，pred 无语义（仅验链路/格式）。"
              "\n[infer_qa]   真实评测请 --ckpt；打分/选择逻辑本身见 score_options/pick_option。")
    print(f"[infer_qa] 评测：python -m data_pipeline.src.qa_eval --manifest {args.manifest} --pred {args.out}")


def main():
    ap = argparse.ArgumentParser(description="四基准 QA 逐选项似然推理 → pred.jsonl")
    ap.add_argument("--manifest", required=True, help="convert_benchmarks 产出的 jsonl")
    ap.add_argument("--data-root", default="data", help="feature_ref 的根")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None, help="训练权重（文件或分片目录）；缺省用未训练 stand-in（仅跑通链路）")
    ap.add_argument("--config", default=None, help="训练同款 config（对齐架构以正确加载 --ckpt）")
    ap.add_argument("--score-mode", default="letter", choices=["letter", "text"])
    ap.add_argument("--feat-dim", type=int, default=None, help="缺省从特征推断")
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
