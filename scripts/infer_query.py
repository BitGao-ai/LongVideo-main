#!/usr/bin/env python3
"""Continuous-query inference demo: readouts at arbitrary sub-frame times.

    python scripts/infer_query.py --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig
from cst_ssm.modules.continuous_query import ContinuousQuery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--nquery", type=int, default=50)
    args = ap.parse_args()

    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96,
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
    model = CSTSSMModel(cfg).to(args.device).eval()

    B, L, P = 1, args.frames, 4
    feats = torch.randn(B, L, P, cfg.feat_dim, device=args.device)
    ts = torch.cumsum(torch.rand(B, L, device=args.device) * 1.0 + 0.5, 1)

    with torch.no_grad():
        x = model.vision(feats)
    branch = model.temporal.branches[0]
    cq = ContinuousQuery(branch)

    t0, t1 = float(ts[0, 0]), float(ts[0, -1])
    tq = torch.linspace(t0, t1, args.nquery, device=args.device).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        y_query = cq.query(x, ts, tq)

    delta = float(ts[0, 1] - ts[0, 0])
    print(f"input {L} frames, mean gap={(t1 - t0)/(L-1):.3f}s -> {args.nquery} sub-frame queries")
    print(f"readout shape: {tuple(y_query.shape)} (resolves below d/4={delta/4:.3f}s)")


if __name__ == "__main__":
    main()
