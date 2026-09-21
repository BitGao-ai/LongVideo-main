"""Benchmark QA inference: per-option likelihood scoring to pred.jsonl."""
from __future__ import annotations

import argparse
import json
import os
import string

import numpy as np

from .dist_utils import (
    get_dist_info,
    log_prefix,
    map_device_for_rank,
    mark_done,
    maybe_set_cuda_device,
    merge_jsonl_parts,
    part_path,
    resolve_shard,
    shard_list,
    wait_for_parts,
)
from .validate_dataset import _load_feat_ts

LETTERS = string.ascii_uppercase


def pick_option(scores) -> tuple:
    """Score list to (index, letter) by argmax."""
    if not len(scores):
        return -1, ""
    idx = int(np.argmax(np.asarray(scores, dtype=np.float64)))
    return idx, LETTERS[idx]


def build_example(tok, prompt: str, answer: str, max_len: int):
    """Concatenate prompt and answer with -100 labels on the prompt; truncate head first."""
    p = [tok.BOS] + list(prompt.encode("utf-8"))
    a = list(answer.encode("utf-8")) + [tok.EOS]
    if len(p) + len(a) > max_len:
        p = p[: max(1, max_len - len(a))]
    ids = p + a
    labels = [-100] * len(p) + a
    return ids, labels


def _make_batch(feats, ts, ids, labels, device="cpu"):
    import torch
    f = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(device)
    t = torch.from_numpy(np.ascontiguousarray(ts)).float().unsqueeze(0).to(device)
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
    """Per-option likelihood scores (-loss); visual states are computed once when possible."""
    import torch
    n = len(options)
    answers = [LETTERS[i] if score_mode == "letter" else str(options[i]) for i in range(n)]
    examples = [build_example(tok, prompt, a, max_len) for a in answers]

    fast = hasattr(model, "encode_visual") and hasattr(model, "llm")
    if not fast:
        scores = []
        for ids, labels in examples:
            batch = _make_batch(feats, ts, ids, labels, device)
            with torch.no_grad():
                out = model(batch)
            loss = out["loss"] if isinstance(out, dict) else out
            scores.append(-float(loss))
        return scores

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
    if config:
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


def infer_manifest(rows, model, tok, data_root: str, max_len: int = 1024,
                   score_mode: str = "letter", device: str = "cpu"):
    out, fails = [], []
    for r in rows:
        try:
            options = r.get("options", [])
            if not options:
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
    rows_all = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    rank, world, local_rank = get_dist_info()
    si, sn, shard_desc, shard_src = resolve_shard(args.shard)
    rows = shard_list(rows_all, si, sn)
    device = map_device_for_rank(args.device, local_rank)
    maybe_set_cuda_device(local_rank)
    if shard_desc is not None:
        print(f"{log_prefix()}[infer_qa] shard {shard_desc} (source={shard_src}): "
              f"this rank handles {len(rows)}/{len(rows_all)} rows; device={device}", flush=True)
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
    model = build_default_model(feat_dim or 3584, args.d_model, args.ckpt, device, args.config)
    preds, fails = infer_manifest(rows, model, tok, args.data_root, args.max_len,
                                  args.score_mode, device)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_path = part_path(args.out, si, sn) if world > 1 else args.out
    fail_path = out_path + ".failed"
    with open(out_path, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    if fails:
        with open(fail_path, "w", encoding="utf-8") as f:
            for x in fails:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{log_prefix()}[infer_qa] {len(preds)} predictions -> {out_path}"
          f"{' (%d failed)' % len(fails) if fails else ''}")
    if world > 1:
        mark_done(out_path)
        if rank == 0:
            if wait_for_parts(args.out, world):
                n = merge_jsonl_parts(args.out, world)
                print(f"[infer_qa] rank0 merged {world} parts -> {args.out} ({n} rows)")
            else:
                print(f"[infer_qa] timed out waiting for parts of {args.out}")
        return
    if not args.ckpt:
        print("[infer_qa] no --ckpt: untrained stand-in, predictions carry no signal.")
    print(f"[infer_qa] eval: python -m data_pipeline.src.qa_eval --manifest {args.manifest} --pred {args.out}")


def main():
    ap = argparse.ArgumentParser(description="Benchmark QA per-option likelihood inference")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--score-mode", default="letter", choices=["letter", "text"])
    ap.add_argument("--feat-dim", type=int, default=None)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
