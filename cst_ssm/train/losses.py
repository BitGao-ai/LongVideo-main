"""Combined training losses for pretraining and finetuning stages."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .contrastive import CPIBWeights, cpib_loss


@dataclass
class LossWeights:
    task: float = 1.0
    pred: float = 0.5
    recon: float = 0.5
    update_rate: float = 0.1
    spectral: float = 0.05
    diff_kv: float = 0.05
    cpib: CPIBWeights | None = None


def _compute_cpib(out: dict, model, w: LossWeights, step: int = 0):
    """CPIB-Distill auxiliary loss; returns (loss, components), zero when disabled."""
    if w.cpib is None or model.cpib is None:
        return 0.0, {}
    if "cpi_tokens" not in out or "cpib_tokens_raw" not in out:
        return 0.0, {}
    scores = out["cpi_tokens"]
    tokens = out["cpib_tokens_raw"]
    frame_feat_d = out.get("cpib_frame_feat")
    if frame_feat_d is None:
        frame_feat_d, _ = model.cpib.distiller(tokens, scores)
    z_future = torch.cat([frame_feat_d[:, 1:].detach(),
                          frame_feat_d[:, -1:].detach()], dim=1)
    distiller_fwd = model.cpib.distiller
    critic = model.cpib_critic
    frame_mask = out.get("frame_mask", out.get("visual_mask"))
    loss, comp = cpib_loss(
        scores, frame_feat_d, tokens, z_future,
        distiller_fwd, w.cpib, step=step,
        frame_mask=frame_mask, critic=critic)
    return loss, comp


def finetune_loss(out: dict, w: LossWeights, model=None, step: int = 0):
    """Stage-2 combined loss; requires out["loss"] (batch must carry labels)."""
    if "loss" not in out:
        raise KeyError(
            'finetune_loss requires out["loss"]; ensure the batch carries labels.')
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
    if "diff_kv_loss" in out and w.diff_kv > 0:
        total = total + w.diff_kv * out["diff_kv_loss"]
        comp["diff_kv"] = out["diff_kv_loss"].detach()
    if model is not None:
        cpib_l, cpib_comp = _compute_cpib(out, model, w, step)
        total = total + cpib_l
        comp.update(cpib_comp)
    comp["total"] = total.detach()
    return total, comp


def pretrain_loss(out: dict, w: LossWeights, model=None, step: int = 0):
    """Stage-1 self-supervised loss."""
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
    if model is not None:
        cpib_l, cpib_comp = _compute_cpib(out, model, w, step)
        total = total + cpib_l
        comp.update(cpib_comp)
    comp["total"] = total.detach()
    return total, comp
