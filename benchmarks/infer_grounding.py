#!/usr/bin/env python3
"""Continuous-query localization inference: inference manifest to pred.jsonl.

Usage:
    python infer_grounding.py --manifest csg_infer.jsonl --data-root . --out pred.jsonl --query-factor 8
    python infer_grounding.py --manifest csg_infer.jsonl --data-root . --ckpt ckpt_final.pt --out pred.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from common import read_jsonl, write_jsonl             # noqa: E402


def subframe_grid(ts, factor: int) -> np.ndarray:
    """Query grid denser than input frames by factor, spanning [ts0, ts_last]."""
    ts = np.asarray(ts, dtype=np.float64)
    if len(ts) < 2:
        return ts.astype(np.float64)
    return np.linspace(float(ts[0]), float(ts[-1]), (len(ts) - 1) * int(factor) + 1)


def boundaries_from_scores(t_query, scores, rel_thresh: float = 0.5):
    """Relevance curve to (start, end) via linear interpolation at threshold crossings."""
    t = np.asarray(t_query, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    Q = len(s)
    if Q == 0:
        return 0.0, 0.0
    if Q == 1:
        return float(t[0]), float(t[0])
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-12:
        return float(t[0]), float(t[-1])
    thr = lo + rel_thresh * (hi - lo)
    peak = int(np.argmax(s))
    i = peak
    while i - 1 >= 0 and s[i - 1] >= thr:
        i -= 1
    j = peak
    while j + 1 < Q and s[j + 1] >= thr:
        j += 1

    def cross(a: int, b: int) -> float:
        sa, sb = s[a], s[b]
        if abs(sb - sa) < 1e-12:
            return float(t[a])
        w = (thr - sa) / (sb - sa)
        w = min(max(w, 0.0), 1.0)
        return float(t[a] + w * (t[b] - t[a]))

    start = cross(i - 1, i) if i > 0 else float(t[i])
    end = cross(j, j + 1) if j < Q - 1 else float(t[j])
    if end < start:
        start, end = end, start
    return start, end


def hash_unit_vector(s: str, d: int, seed: int = 0) -> np.ndarray:
    """Deterministic unit vector from a query string (untrained placeholder)."""
    h = int(hashlib.sha1(f"{seed}:{s}".encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(h).standard_normal(d)
    return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)


class HashDirectionScorer:
    """Untrained placeholder scorer: continuous readouts dotted with a hash direction."""

    def __init__(self, ckpt: str | None = None, d_model: int = 96, device: str = "cpu"):
        self.ckpt = ckpt
        self.d_model = d_model
        self.device = device
        self._model = None
        self._feat_dim = None

    def _build(self, feat_dim: int):
        import torch
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.modules.llm_interface import LLMConfig
        cfg = CSTSSMConfig(input_mode="feature", feat_dim=feat_dim, d_model=self.d_model,
                           llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
        model = CSTSSMModel(cfg).to(self.device).eval()
        if self.ckpt:
            from cst_ssm.utils import load_checkpoint
            load_checkpoint(model, self.ckpt, tag="hash-scorer")
        self._model, self._feat_dim = model, feat_dim

    def score(self, feats: np.ndarray, ts: np.ndarray, t_query: np.ndarray, query: str) -> np.ndarray:
        import torch
        from cst_ssm.modules.continuous_query import ContinuousQuery
        if self._model is None or self._feat_dim != feats.shape[-1]:
            self._build(int(feats.shape[-1]))
        fe = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(self.device)
        tt = torch.from_numpy(np.ascontiguousarray(ts)).float().unsqueeze(0).to(self.device)
        tq = torch.from_numpy(np.ascontiguousarray(t_query)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self._model.vision(fe)
            cq = ContinuousQuery(self._model.temporal.branches[0])
            y = cq.query(x, tt, tq)[0].cpu().numpy()
        q = hash_unit_vector(query, y.shape[-1])
        return y @ q


class TrainedGroundingScorer:
    """Trained grounding-head scorer from --config plus --ckpt weights."""

    def __init__(self, ckpt=None, config=None, d_model=96, device="cpu"):
        self.ckpt, self.config, self.d_model, self.device = ckpt, config, d_model, device
        self._model = None
        self._tok = None
        self._feat_dim = None

    def _build(self, feat_dim: int):
        import torch
        from cst_ssm.models import CSTSSMModel, CSTSSMConfig
        from cst_ssm.modules.llm_interface import LLMConfig
        from cst_ssm.data.dataset import ByteTokenizer
        if self.config:
            from cst_ssm.utils import model_config_from_dict, load_yaml
            d = load_yaml(self.config)["model"]; d["grounding_head"] = True
            cfg = model_config_from_dict(d)
        else:
            cfg = CSTSSMConfig(input_mode="feature", feat_dim=feat_dim, d_model=self.d_model,
                               grounding_head=True,
                               llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
        model = CSTSSMModel(cfg).to(self.device).eval()
        if self.ckpt:
            from cst_ssm.utils import load_checkpoint
            load_checkpoint(model, self.ckpt, tag="grounding-scorer")
        self._model, self._tok, self._feat_dim = model, ByteTokenizer(), feat_dim

    def score(self, feats, ts, t_query, query: str) -> np.ndarray:
        import torch
        if self._model is None or self._feat_dim != feats.shape[-1]:
            self._build(int(feats.shape[-1]))
        ids = self._tok.encode(query)[: 512]
        fe = torch.from_numpy(np.ascontiguousarray(feats)).float().unsqueeze(0).to(self.device)
        tt = torch.from_numpy(np.ascontiguousarray(ts)).float().unsqueeze(0).to(self.device)
        tq = torch.from_numpy(np.ascontiguousarray(t_query)).float().unsqueeze(0).to(self.device)
        iid = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
        am = torch.ones_like(iid, dtype=torch.bool)
        with torch.no_grad():
            logits = self._model.ground_scores(fe, tt, iid, tq, am)[0].cpu().numpy()
        return logits


def build_scorer(args):
    kind = args.scorer
    if kind == "hash":
        return HashDirectionScorer(ckpt=args.ckpt, d_model=args.d_model, device=args.device)
    return TrainedGroundingScorer(ckpt=args.ckpt, config=args.config,
                                  d_model=args.d_model, device=args.device)


def load_features(feature_ref: str, data_root: str):
    """Load .npz or .npy(+.ts.npy) features with float32 timestamps."""
    path = os.path.join(data_root, feature_ref)
    if feature_ref.endswith(".npy"):
        feats = np.load(path, mmap_mode="r")
        ts = np.load(path[:-4] + ".ts.npy")
    else:
        z = np.load(path)
        feats, ts = z["features"], z["timestamps"]
    return np.asarray(feats, np.float32), np.asarray(ts, np.float32)


def run(args):
    rows = read_jsonl(args.manifest)
    scorer = build_scorer(args)
    out, n_fail = [], 0
    fails = []
    for r in rows:
        try:
            feats, ts = load_features(r["feature_ref"], args.data_root)
            tq = subframe_grid(ts, args.query_factor)
            scores = scorer.score(feats, ts, tq, r.get("prompt", r.get("query", "")))
            ps, pe = boundaries_from_scores(tq, scores, args.rel_thresh)
            out.append({"query_id": r["query_id"], "pred_start": round(ps, 4),
                        "pred_end": round(pe, 4)})
        except Exception as e:
            n_fail += 1
            fails.append({"query_id": r.get("query_id"), "error": str(e)})
    write_jsonl(args.out, out)
    if fails:
        write_jsonl(args.out + ".failed", fails)
    print(f"[infer] {len(out)} predictions -> {args.out}{' (%d failed)' % n_fail if n_fail else ''}")
    if not args.ckpt:
        print("[infer] no --ckpt: scorer is untrained, predictions carry no signal.")


def main():
    ap = argparse.ArgumentParser(description="Continuous-query grounding inference to pred.jsonl")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", default=".")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scorer", default="trained", choices=["trained", "hash"])
    ap.add_argument("--query-factor", type=int, default=8)
    ap.add_argument("--rel-thresh", type=float, default=0.5)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
