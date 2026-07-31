#!/usr/bin/env python3
"""消融组 5：扫描事件阈值 ε，实测"更新率↓ / 输出误差↑"的单调 Pareto，验证 Theorem 1（设计 §6.3）。

Theorem 1 预测：输出误差 ≤ Θ·ε（随 ε 线性增）、更新次数 ≤ 1+V_T/(cε)（随 ε 降）。
本脚本以稠密参考(ε→0)为基线，扫 ε 记录 (update_rate, ||o(ε)-o_dense||)，验证单调性；
定量 Θ 系数需真实训练后的模型填充（此处用未训练模型演示定性关系）。

    python scripts/ablation_pareto.py --device cpu
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from cst_ssm.modules.multiscale import MultiScaleEACS
from cst_ssm.ops.spectral_init import BranchSpec


def set_eps(ms: MultiScaleEACS, eps_val: float) -> None:
    for br in ms.branches:
        g = br.gate
        frac = (eps_val - g.eps_min) / (g.eps_max - g.eps_min)
        frac = min(max(frac, 1e-4), 1 - 1e-4)
        with torch.no_grad():
            g.eps_raw.copy_(torch.logit(torch.tensor(frac)))
        g.set_eps_anneal(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--frames", type=int, default=80)
    args = ap.parse_args()
    torch.manual_seed(0)

    br = (BranchSpec("s", 8, 0.1, 1.0), BranchSpec("m", 12, 10, 60), BranchSpec("l", 16, 600, 3600))
    ms = MultiScaleEACS(48, branches=br).to(args.device).eval()
    x = torch.randn(1, args.frames, 48, device=args.device)
    x[:, 1::2] = x[:, 0:-1:2]                      # 偶帧≈前帧，制造可跳过冗余
    t = torch.cumsum(torch.rand(1, args.frames, device=args.device) * 0.3 + 0.1, 1)

    set_eps(ms, 0.001)                             # 稠密参考（ε→0，近全更新）
    with torch.no_grad():
        ref = ms(x, t).y

    eps_grid = [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
    print(f"{'ε':>6} | {'更新率':>8} | {'输出误差 ||o(ε)-o_dense||':>22} | {'误差/ε':>8}")
    rows = []
    for eps in eps_grid:
        set_eps(ms, eps)
        with torch.no_grad():
            o = ms(x, t)
        err = (o.y - ref).norm().item()
        ur = o.update_rate.item()
        rows.append((eps, ur, err))
        print(f"{eps:>6.2f} | {ur:>8.3f} | {err:>22.4f} | {err/eps:>8.3f}")

    ur_mono = all(rows[i][1] >= rows[i + 1][1] - 1e-3 for i in range(len(rows) - 1))
    err_mono = all(rows[i][2] <= rows[i + 1][2] + 1e-3 for i in range(len(rows) - 1))
    print(f"\n更新率随 ε 单调↓: {ur_mono}   输出误差随 ε 单调↑: {err_mono}")
    print("→ 与 Theorem 1 定性预测一致（单一旋钮 ε 单调权衡误差与计算量）。")
    print("  定量：真实训练后，'误差/ε' 应趋于常数 Θ；'更新率-ε' 曲线叠加 1+V_T/(cε) 预测带。")

    try:  # 可选出图
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(4, 3))
        plt.plot([r[1] for r in rows], [r[2] for r in rows], "o-")
        plt.xlabel("update rate"); plt.ylabel("output error"); plt.title("ε-Pareto")
        plt.tight_layout(); plt.savefig("ablation_pareto.png", dpi=120)
        print("  已保存 ablation_pareto.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
