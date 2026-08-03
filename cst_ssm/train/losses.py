"""组合损失（设计方案 §4.1/§4.4 训练目标 + 创新点1 CPIB-Distill）。

stage-2 微调：L = λ_task·L_task + λ_pred·L_pred + λ_upd·更新率 + λ_spec·谱正则 + L_cpib
stage-1 预训练：L = λ_pred·未来预测MSE + λ_recon·掩码帧重构 + λ_spec·谱正则 + L_cpib
其中 L_cpib = λ1·L_NCE + λ2·R_KL + λ3·L_cf（创新点1，仅当 cpib_distill=True 时计算）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .contrastive import CPIBWeights, cpib_loss


@dataclass
class LossWeights:
    task: float = 1.0
    pred: float = 0.5            # λ_pred：未来预测（stage-1 主项 / stage-2 辅助项）
    recon: float = 0.5          # λ_recon：stage-1 掩码帧重构
    update_rate: float = 0.1     # λ_upd：越大越稀疏
    spectral: float = 0.05       # λ_spec：谱防坍缩
    diff_kv: float = 0.05        # λ_diffkv：差分 KV 重构损失（创新点3）
    cpib: CPIBWeights | None = None   # 创新点1 权重（None=不启用）


def _compute_cpib(out: dict, model, w: LossWeights, step: int = 0):
    """若模型启用 CPIB-Distill 且输出含 CPI 信号，计算附加损失。"""
    if w.cpib is None or model.cpib is None:
        return torch.tensor(0.0), {}
    if "cpi_tokens" not in out or "cpib_tokens_raw" not in out:
        return torch.tensor(0.0), {}
    scores = out["cpi_tokens"]               # (B,L,P)
    tokens = out["cpib_tokens_raw"]          # (B,L,P,d)
    # 用 tokens 均值作为帧级表征 (B,L,d)
    frame_feat_d = tokens.mean(dim=2)        # (B,L,d)
    # 未来帧目标：时序 shift，z_future[t] = frame_feat[t+1]（设计文档：预测未来 τ 窗口表征）
    z_future = torch.cat([frame_feat_d[:, 1:].detach(),
                          frame_feat_d[:, -1:].detach()], dim=1)  # (B,L,d)
    distiller_fwd = model.cpib.distiller
    critic = model.cpib_critic
    frame_mask = out.get("visual_mask")      # 可能为 None
    loss, comp = cpib_loss(
        scores, frame_feat_d, tokens, z_future,
        distiller_fwd, w.cpib, step=step,
        frame_mask=frame_mask, critic=critic)
    return loss, comp


def finetune_loss(out: dict, w: LossWeights, model=None, step: int = 0):
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
    # 创新点3 差分 KV 重构损失
    if "diff_kv_loss" in out and w.diff_kv > 0:
        total = total + w.diff_kv * out["diff_kv_loss"]
        comp["diff_kv"] = out["diff_kv_loss"].detach()
    # 创新点1 CPIB-Distill 附加损失
    if model is not None:
        cpib_l, cpib_comp = _compute_cpib(out, model, w, step)
        total = total + cpib_l
        comp.update(cpib_comp)
    comp["total"] = total.detach()
    return total, comp


def pretrain_loss(out: dict, w: LossWeights, model=None, step: int = 0):
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
    # 创新点1 CPIB-Distill 附加损失
    if model is not None:
        cpib_l, cpib_comp = _compute_cpib(out, model, w, step)
        total = total + cpib_l
        comp.update(cpib_comp)
    comp["total"] = total.detach()
    return total, comp
