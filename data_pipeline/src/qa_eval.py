"""QA accuracy eval over benchmark manifests plus model predictions.

Usage:
    python -m data_pipeline.src.qa_eval --manifest videomme.jsonl --pred cstssm_videomme.jsonl
    python -m data_pipeline.src.qa_eval --manifest videomme.jsonl --demo
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from .convert_benchmarks import normalize_answer


def _read_jsonl(path: str):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def evaluate(manifest: list, preds: dict) -> dict:
    """preds maps query_id to raw prediction text; returns overall and grouped accuracy."""
    correct, total, missing = 0, 0, 0
    by_task = defaultdict(lambda: [0, 0])
    by_bucket = defaultdict(lambda: [0, 0])
    for r in manifest:
        gold = r.get("answer_index", -1)
        if gold < 0:
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
    """Named buckets (short/medium/long) first, then numeric buckets ascending."""
    named = {"short": 0, "medium": 1, "long": 2}

    def key(kv):
        k = kv[0]
        if k in named:
            return (0, named[k])
        try:
            return (1, float(k))
        except (TypeError, ValueError):
            return (2, k)
    return sorted(by_bucket.items(), key=key)


def demo_preds(manifest: list, seed: int = 0):
    """Synthetic perfect and random predictions for plumbing checks."""
    import random
    rng = random.Random(seed)
    perfect, rnd = {}, {}
    for r in manifest:
        if r.get("answer_index", -1) < 0:
            continue
        perfect[r["query_id"]] = r["answer"]
        k = max(1, len(r.get("options", [])) or 4)
        rnd[r["query_id"]] = "ABCDEFGH"[rng.randrange(k)]
    return perfect, rnd


def _print(res: dict):
    print(f"  bench={res['bench']}  n_eval={res['n_eval']}  n_missing_pred={res['n_missing_pred']}")
    print(f"  accuracy: {res['accuracy']}")
    if res["by_duration_bucket"]:
        print(f"  by_duration_bucket: {res['by_duration_bucket']}")
    if res["by_task"]:
        items = list(res["by_task"].items())
        print("  by_task: " + ", ".join(f"{k}={v}" for k, v in items[:8])
              + (" ..." if len(items) > 8 else ""))


def main():
    ap = argparse.ArgumentParser(description="Benchmark QA accuracy eval")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--pred")
    ap.add_argument("--demo", action="store_true")
    a = ap.parse_args()
    manifest = _read_jsonl(a.manifest)
    if a.demo:
        perfect, rnd = demo_preds(manifest)
        print("=== perfect (expect acc=1.0) ===");  _print(evaluate(manifest, perfect))
        print("=== random (expect ~1/num_options) ===");   _print(evaluate(manifest, rnd))
        return
    preds = {r["query_id"]: r["pred"] for r in _read_jsonl(a.pred)}
    _print(evaluate(manifest, preds))


if __name__ == "__main__":
    main()
