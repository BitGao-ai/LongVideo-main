"""组合损失（设计方案 §4.1/§4.4 训练目标）。

stage-2 微调：L = λ_task·L_task + λ_pred·L_pred + λ_upd·更新率 + λ_spec·谱正则
stage-1 预训练：L = λ_pred·未来预测MSE + λ_recon·掩码帧重构 + λ_spec·谱正则
其中"更新率正则"鼓励模型在达标前提下自发降低有效更新率（内容自适应效率）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LossWeights:
    task: float = 1.0
    pred: float = 0.5            # λ_pred：未来预测（stage-1 主项 / stage-2 辅助项）
    recon: float = 0.5          # λ_recon：stage-1 掩码帧重构
    update_rate: float = 0.1     # λ_upd：越大越稀疏
    spectral: float = 0.05       # λ_spec：谱防坍缩


def finetune_loss(out: dict, w: LossWeights):
    comp = {"task": out["loss"].detach()}
    total = w.task * out["loss"]
    if "pred_loss" in out:
        total = total + w.pred * out["pred_loss"]
        comp["pred"] = out["pred_loss"].detach()
    if "update_rate" in out:
        total = total + w.update_rate * out["update_rate"]
        comp["update_rate"] = out["update_rate"].detach()
    if "spectral_reg" in out:
        total = total + w.spectral * out["spectral_reg"]
        comp["spectral"] = out["spectral_reg"].detach()
    comp["total"] = total.detach()
    return total, comp


def pretrain_loss(out: dict, w: LossWeights):
    comp = {"pred": out["pred_loss"].detach()}
    total = w.pred * out["pred_loss"]
    if "recon_loss" in out:
        total = total + w.recon * out["recon_loss"]
        comp["recon"] = out["recon_loss"].detach()
    if "spectral_reg" in out:
        sr = out["spectral_reg"]
        sr = sr if torch.is_tensor(sr) else torch.tensor(float(sr))
        total = total + w.spectral * sr
        comp["spectral"] = sr.detach()
    comp["total"] = total.detach()
    return total, comp
