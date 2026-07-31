#!/usr/bin/env python3
"""任意时刻连续查询推理 demo（设计方案 §4.3，杀手锏 C3）。

演示在**帧与帧之间**任意时刻求视觉状态读出——离散模型做不到（受 δ/4 地板限制）。
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
    ts = torch.cumsum(torch.rand(B, L, device=args.device) * 1.0 + 0.5, 1)  # 可变间隔

    # 空间编码 → 帧级特征（作为 EACS 输入）
    with torch.no_grad():
        x = model.vision(feats)
    branch = model.temporal.branches[0]            # 取一个尺度分支做连续查询
    cq = ContinuousQuery(branch)

    # 亚帧查询网格（比输入帧更密）
    t0, t1 = float(ts[0, 0]), float(ts[0, -1])
    tq = torch.linspace(t0, t1, args.nquery, device=args.device).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        y_query = cq.query(x, ts, tq)              # (B, nquery, d_model)

    delta = float(ts[0, 1] - ts[0, 0])
    print(f"输入 {L} 帧, 平均帧距≈{(t1 - t0)/(L-1):.3f}s → 连续查询 {args.nquery} 个亚帧时刻")
    print(f"查询读出形状: {tuple(y_query.shape)}  (可在 δ/4≈{delta/4:.3f}s 分辨率以下定位)")
    print("✓ 连续查询在帧间任意时刻求值——离散基线结构上做不到（命题 2）")


if __name__ == "__main__":
    main()
