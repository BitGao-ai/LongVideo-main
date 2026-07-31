"""训练器：两阶段统一循环 + 显存优化 + 分片检查点。

显存优化（需求 2）：梯度累积、bf16 autocast、EACS 分块梯度检查点、LoRA 冻结基座、
门控温度退火；保存时用 save_sharded 保证单文件 ≤4GB。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .losses import LossWeights, finetune_loss, pretrain_loss
from ..utils.memory import (set_eacs_chunk, set_gate_temperature, autocast_ctx,
                            build_optimizer, gate_temperature_schedule,
                            set_eps_anneal, eps_anneal_schedule)
from ..utils.checkpoint import save_sharded, verify_shards


@dataclass
class TrainConfig:
    stage: str = "finetune"          # "finetune" | "pretrain"
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_accum: int = 1
    max_steps: int = 1000
    grad_clip: float = 1.0
    bf16: bool = True
    device: str = "cuda"
    eacs_chunk: int = 0              # >0 启用 EACS 分块梯度检查点
    log_every: int = 10
    ckpt_every: int = 0             # >0 周期保存
    ckpt_dir: str = "checkpoints"
    t_start: float = 1.0            # 门控温度退火起点
    t_end: float = 0.05
    eps_anneal_start: float = 1.0   # ε 退火起点（<1 前期多更新学动力学）；默认1=无退火
    eps_anneal_end: float = 1.0


def move_to(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, weights: LossWeights | None = None):
        self.model = model.to(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.cfg = cfg
        self.weights = weights or LossWeights()
        set_eacs_chunk(self.model, cfg.eacs_chunk)
        self.opt = build_optimizer(self.model, cfg.lr, cfg.weight_decay)
        self.step = 0

    def train_step(self, batch: dict) -> dict:
        cfg = self.cfg
        self.model.train()
        batch = move_to(batch, cfg.device)
        # 门控温度退火
        set_gate_temperature(self.model, gate_temperature_schedule(
            self.step, cfg.max_steps, cfg.t_start, cfg.t_end))
        # ε 退火（默认 start=end=1.0 即无退火）
        set_eps_anneal(self.model, eps_anneal_schedule(
            self.step, cfg.max_steps, cfg.eps_anneal_start, cfg.eps_anneal_end))
        with autocast_ctx(cfg.bf16, device_type=("cuda" if "cuda" in cfg.device else "cpu")):
            if cfg.stage == "pretrain":
                out = self.model.pretrain_forward(batch)
                loss, comp = pretrain_loss(out, self.weights)
            elif cfg.stage == "grounding":
                out = self.model.grounding_forward(batch)
                loss = out["loss"]
                comp = {"grounding": out["grounding_loss"], "total": out["loss"].detach()}
            else:
                out = self.model(batch)
                loss, comp = finetune_loss(out, self.weights)
        (loss / cfg.grad_accum).backward()
        if (self.step + 1) % cfg.grad_accum == 0:
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                         cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)     # set_to_none 省显存
        self.step += 1
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in comp.items()}

    def fit(self, data: Iterable[dict]) -> None:
        for batch in data:
            if self.step >= self.cfg.max_steps:
                break
            logs = self.train_step(batch)
            if self.step % self.cfg.log_every == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"[step {self.step:>6}] {msg}")
            if self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                self.save(f"step{self.step}")

    def save(self, tag: str) -> str:
        out_dir = os.path.join(self.cfg.ckpt_dir, tag)
        save_sharded(self.model.state_dict(), out_dir)
        verify_shards(out_dir)
        return out_dir
