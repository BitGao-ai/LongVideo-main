"""四基准 QA 准确率评测（GATE-4 打分入口）。消费 convert_benchmarks 的 manifest + 模型 pred。

pred 格式（jsonl）：{"query_id","pred"}，pred 可为 字母 / 索引 / 选项原文——用与转换时相同的
normalize_answer 归一后按 answer_index 比对（鲁棒于模型输出风格）。

报告：整体准确率 + 按 task_type + 按 duration_bucket（VideoMME 的 short/medium/**long**，GATE-4 关键）。

用法：
  python -m data_pipeline.src.qa_eval --manifest videomme.jsonl --pred cstssm_videomme.jsonl
  python -m data_pipeline.src.qa_eval --manifest videomme.jsonl --demo    # 合成 perfect/random 验证逻辑
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .convert_benchmarks import normalize_answer


def _read_jsonl(path: str):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def evaluate(manifest: list, preds: dict) -> dict:
    """preds: query_id -> 原始 pred 字符串/数字。返回整体与分组准确率。"""
    correct, total, missing = 0, 0, 0
    by_task = defaultdict(lambda: [0, 0])          # task -> [correct, total]
    by_bucket = defaultdict(lambda: [0, 0])
    for r in manifest:
        gold = r.get("answer_index", -1)
        if gold < 0:                                # 金标未知（如 fullset）→ 不计入
            continue
        qid = r["query_id"]
        if qid not in preds:
            missing += 1; continue
        _, pidx = normalize_answer(preds[qid], r.get("options", []), base=1)
        ok = int(pidx == gold)
        correct += ok; total += 1
        by_task[r.get("task_type", "NA")][0] += ok
        by_task[r.get("task_type", "NA")][1] += 1
        if "duration_bucket" in r:
            by_bucket[r["duration_bucket"]][0] += ok
            by_bucket[r["duration_bucket"]][1] += 1

    acc = correct / total if total else float("nan")
    return {
        "bench": manifest[0].get("bench", "?") if manifest else "?",
        "n_eval": total, "n_missing_pred": missing,
        "accuracy": round(acc, 4),
        "by_task": {k: round(c / t, 4) for k, (c, t) in sorted(by_task.items())},
        "by_duration_bucket": {k: round(c / t, 4)
                               for k, (c, t) in _bucket_order(by_bucket)},
    }


def _bucket_order(by_bucket):
    """named 档(short/medium/long) 在前，其后数值档(LongVideoBench 的 15/60/600/3600)按数值升序。"""
    named = {"short": 0, "medium": 1, "long": 2}

    def key(kv):
        k = kv[0]
        if k in named:
            return (0, named[k])
        try:
            return (1, float(k))                       # 数值 duration_group 升序
        except (TypeError, ValueError):
            return (2, k)
    return sorted(by_bucket.items(), key=key)


def demo_preds(manifest: list, seed: int = 0):
    """合成两组 pred：perfect（金标字母）与 random，验证评测逻辑与分组。"""
    import random
    rng = random.Random(seed)
    perfect, rnd = {}, {}
    for r in manifest:
        if r.get("answer_index", -1) < 0:
            continue
        perfect[r["query_id"]] = r["answer"]                      # 金标字母
        k = max(1, len(r.get("options", [])) or 4)
        rnd[r["query_id"]] = "ABCDEFGH"[rng.randrange(k)]
    return perfect, rnd


def _print(res: dict):
    print(f"  bench={res['bench']}  n_eval={res['n_eval']}  n_missing_pred={res['n_missing_pred']}")
    print(f"  accuracy: {res['accuracy']}")
    if res["by_duration_bucket"]:
        print(f"  by_duration_bucket: {res['by_duration_bucket']}  ← GATE-4 看 long")
    if res["by_task"]:
        items = list(res["by_task"].items())
        print("  by_task: " + ", ".join(f"{k}={v}" for k, v in items[:8])
              + (" ..." if len(items) > 8 else ""))


def main():
    ap = argparse.ArgumentParser(description="四基准 QA 准确率评测")
    ap.add_argument("--manifest", required=True, help="convert_benchmarks 产出的 jsonl")
    ap.add_argument("--pred", help="预测 jsonl {query_id,pred}")
    ap.add_argument("--demo", action="store_true", help="合成 perfect/random 验证逻辑")
    a = ap.parse_args()
    manifest = _read_jsonl(a.manifest)
    if a.demo:
        perfect, rnd = demo_preds(manifest)
        print("=== [DEMO] perfect 预测（应 acc=1.0）===");  _print(evaluate(manifest, perfect))
        print("=== [DEMO] random 预测（应≈1/选项数）===");   _print(evaluate(manifest, rnd))
        return
    preds = {r["query_id"]: r["pred"] for r in _read_jsonl(a.pred)}
    _print(evaluate(manifest, preds))


if __name__ == "__main__":
    main()
