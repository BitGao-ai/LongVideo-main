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
    """若模型启用 CPIB-Distill 且输出含 CPI 信号，计算附加损失。

    未启用时返回标量 0.0（python float）：与任意设备上的 loss 张量相加都安全，
    避免 `torch.tensor(0.0)` 常驻 CPU 导致 GPU 训练时 `cuda + cpu` 设备错配崩溃。
    """
    if w.cpib is None or model.cpib is None:
        return 0.0, {}
    if "cpi_tokens" not in out or "cpib_tokens_raw" not in out:
        return 0.0, {}
    scores = out["cpi_tokens"]               # (B,L,P)
    tokens = out["cpib_tokens_raw"]          # (B,L,P,d)
    # 帧级表征必须用真实 distiller 输出（分数加权保留 + 注意力池化摘要 + 门控融合），
    # 与反事实消融 feat_cf 同源同语义；用 tokens.mean 会混入与消融无关的恒定基线偏差，
    # 使 L_cf 回归被污染的量、InfoNCE 的 z_t 与设计文档"CGU 保留表征"不符。
    # 模型 forward 已把这个量透传出来（cpib_frame_feat），直接复用——重新调一次
    # distiller 意味着每步多一遍 (B·L,P,d) 的注意力池化，纯属白算。
    frame_feat_d = out.get("cpib_frame_feat")
    if frame_feat_d is None:                 # 兼容自定义模型未透传的情况
        frame_feat_d, _ = model.cpib.distiller(tokens, scores)   # (B,L,d)
    # 未来帧目标：时序 shift，z_future[t] = frame_feat[t+1]（设计文档：预测未来 τ 窗口表征）
    z_future = torch.cat([frame_feat_d[:, 1:].detach(),
                          frame_feat_d[:, -1:].detach()], dim=1)  # (B,L,d)
    distiller_fwd = model.cpib.distiller
    critic = model.cpib_critic
    # 有效帧 mask：模型 forward 透传 batch["frame_mask"]（旧键 visual_mask 兼容）
    frame_mask = out.get("frame_mask", out.get("visual_mask"))
    loss, comp = cpib_loss(
        scores, frame_feat_d, tokens, z_future,
        distiller_fwd, w.cpib, step=step,
        frame_mask=frame_mask, critic=critic)
    return loss, comp


def finetune_loss(out: dict, w: LossWeights, model=None, step: int = 0):
    """stage-2 组合损失。out 必须含 "loss"（LLM 的任务损失）——即 batch 里要有 labels。"""
    if "loss" not in out:
        raise KeyError(
            'finetune_loss 需要 out["loss"]，但 LLM 只在 labels 非空时才产出它。'
            "请确认 batch 里带 labels（Trainer.validate 走 finetune 阶段时同样需要），"
            "或改用不依赖任务损失的评测入口。")
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
