#!/usr/bin/env python3
"""Streaming inference demo: update rate against content redundancy.

    python scripts/infer_stream.py --device cpu --frames 200
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cst_ssm.models import CSTSSMModel, CSTSSMConfig
from cst_ssm.modules.llm_interface import LLMConfig


def make_video(L, P, d, redundancy: float, device):
    """Higher redundancy means more similar adjacent frames, hence fewer updates."""
    base = torch.randn(1, 1, P, d, device=device)
    noise = torch.randn(1, L, P, d, device=device)
    feats = redundancy * base + (1 - redundancy) * noise
    ts = torch.cumsum(torch.rand(1, L, device=device) * 0.3 + 0.1, 1)
    return feats, ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=200)
    args = ap.parse_args()

    cfg = CSTSSMConfig(input_mode="feature", feat_dim=64, d_model=96,
                       llm=LLMConfig(vocab_size=259, dim=128, n_layer=2, n_head=4, max_len=64))
    model = CSTSSMModel(cfg).to(args.device).eval()

    print(f"{'redundancy':>10} | {'upd_rate':>10} | per-branch (short/mid/long)")
    for red in (0.1, 0.5, 0.9, 0.99):
        feats, ts = make_video(args.frames, 4, cfg.feat_dim, red, args.device)
        with torch.no_grad():
            x = model.vision(feats)
            ms = model.temporal(x, ts)
        pbur = [f"{v.item():.3f}" for v in ms.per_branch_update_rate]
        print(f"{red:>10.2f} | {ms.update_rate.item():>10.3f} | {' / '.join(pbur)}")


if __name__ == "__main__":
    main()
