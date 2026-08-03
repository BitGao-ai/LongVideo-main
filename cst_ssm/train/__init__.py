"""训练层：组合损失、两阶段训练器、CPIB-Distill 损失。"""
from __future__ import annotations

from .losses import LossWeights, finetune_loss, pretrain_loss
from .trainer import Trainer, TrainConfig, move_to
from .contrastive import (
    BilinearCritic, conditional_infonce_loss, information_bottleneck_kl,
    counterfactual_consistency_loss, CPIBWeights, cpib_loss,
)

__all__ = ["LossWeights", "finetune_loss", "pretrain_loss",
           "Trainer", "TrainConfig", "move_to",
           "BilinearCritic", "conditional_infonce_loss", "information_bottleneck_kl",
           "counterfactual_consistency_loss", "CPIBWeights", "cpib_loss"]
