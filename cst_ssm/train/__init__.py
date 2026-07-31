"""训练层：组合损失、两阶段训练器。"""
from __future__ import annotations

from .losses import LossWeights, finetune_loss, pretrain_loss
from .trainer import Trainer, TrainConfig, move_to

__all__ = ["LossWeights", "finetune_loss", "pretrain_loss",
           "Trainer", "TrainConfig", "move_to"]
