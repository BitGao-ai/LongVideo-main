#!/usr/bin/env python3
"""流式长视频推理 demo：报告有效更新率与内容自适应效率（设计方案 §4.2 效率）。

    python scripts/infer_stream.py --device cpu --frames 200
显存只维护当前分层状态（O(1)，与视频长度无关）；有效更新率随内容稀疏度下降。
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
    """redundancy∈[0,1]：越大越冗余（相邻帧越相似）→ 有效更新率应越低。"""
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

    print(f"{'内容冗余度':>10} | {'整体更新率':>10} | 各分支(短/中/长)更新率")
    for red in (0.1, 0.5, 0.9, 0.99):
        feats, ts = make_video(args.frames, 4, cfg.feat_dim, red, args.device)
        # 推理全程无梯度：vision 前向也包在 no_grad 内，避免长序列激活无谓驻留内存
        with torch.no_grad():
            x = model.vision(feats)
            ms = model.temporal(x, ts)
        pbur = [f"{v.item():.3f}" for v in ms.per_branch_update_rate]
        print(f"{red:>10.2f} | {ms.update_rate.item():>10.3f} | {' / '.join(pbur)}")
    print("\n注：未训练模型在随机内容上更新率≈1（无法预测）；训练后/真实冗余场景更新率显著下降。")
    print("显存：全程仅维护三分支当前状态向量 O(n)，与视频长度无关。")


if __name__ == "__main__":
    main()
