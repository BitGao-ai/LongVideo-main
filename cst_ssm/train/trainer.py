"""训练器：两阶段统一循环 + 显存优化 + 分片检查点 + DDP 分布式。

显存优化（需求 2）：梯度累积、bf16 autocast、EACS 分块梯度检查点、LoRA 冻结基座、
门控温度退火；保存时用 save_sharded 保证单文件 ≤4GB。
分布式：支持 torchrun 启动的 DDP 多卡训练（设计文档要求 8×A100）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.distributed as dist

from .losses import LossWeights, finetune_loss, pretrain_loss
from ..utils.memory import (set_eacs_chunk, set_gate_temperature, autocast_ctx,
                            build_optimizer, gate_temperature_schedule,
                            set_eps_anneal, eps_anneal_schedule)
from ..utils.checkpoint import save_sharded, verify_shards


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


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
    # 验证（P4 修复）
    val_every: int = 0              # >0 周期性验证（step 间隔）
    val_steps: int = 50             # 每次验证最多跑多少 batch
    # DDP 分布式（P3）
    ddp: bool = False               # True 时启用 DistributedDataParallel
    ddp_backend: str = "nccl"       # nccl(gpu) | gloo(cpu)


def move_to(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, weights: LossWeights | None = None):
        self.cfg = cfg
        self.weights = weights or LossWeights()
        self.step = 0

        # DDP 初始化（P3）
        self._ddp_model = None
        if cfg.ddp:
            if not is_distributed():
                dist.init_process_group(backend=cfg.ddp_backend)
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = f"cuda:{local_rank}" if "cuda" in cfg.device else cfg.device
            self.model = model.to(device)
            self._ddp_model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank] if "cuda" in device else None,
                find_unused_parameters=True)  # CPIB/grounding 可选模块可能未参与每步
            cfg.device = device
        else:
            device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"
            self.model = model.to(device)
            cfg.device = device

        set_eacs_chunk(self.model, cfg.eacs_chunk)
        self.opt = build_optimizer(self.model, cfg.lr, cfg.weight_decay)

    @property
    def _forward_model(self):
        """DDP 包装后的模型（用于前向），或原始模型。"""
        return self._ddp_model if self._ddp_model is not None else self.model

    def train_step(self, batch: dict) -> dict:
        cfg = self.cfg
        fwd_model = self._forward_model
        fwd_model.train()
        batch = move_to(batch, cfg.device)
        # 门控温度退火
        set_gate_temperature(self.model, gate_temperature_schedule(
            self.step, cfg.max_steps, cfg.t_start, cfg.t_end))
        # ε 退火（默认 start=end=1.0 即无退火）
        set_eps_anneal(self.model, eps_anneal_schedule(
            self.step, cfg.max_steps, cfg.eps_anneal_start, cfg.eps_anneal_end))
        with autocast_ctx(cfg.bf16, device_type=("cuda" if "cuda" in cfg.device else "cpu")):
            if cfg.stage == "pretrain":
                out = fwd_model.pretrain_forward(batch)
                loss, comp = pretrain_loss(out, self.weights, model=self.model, step=self.step)
            elif cfg.stage == "grounding":
                out = fwd_model.grounding_forward(batch)
                loss = out["loss"]
                comp = {"grounding": out["grounding_loss"], "total": out["loss"].detach()}
            else:
                out = fwd_model(batch)
                loss, comp = finetune_loss(out, self.weights, model=self.model, step=self.step)
        (loss / cfg.grad_accum).backward()
        if (self.step + 1) % cfg.grad_accum == 0:
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                         cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)     # set_to_none 省显存
        self.step += 1
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in comp.items()}

    def fit(self, data: Iterable[dict], val_loader: Iterable[dict] | None = None) -> None:
        for batch in data:
            if self.step >= self.cfg.max_steps:
                break
            logs = self.train_step(batch)
            if self.step % self.cfg.log_every == 0 and is_main_process():
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"[step {self.step:>6}] {msg}")
            if self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                self.save(f"step{self.step}")
            # 周期性验证
            if (self.cfg.val_every and val_loader is not None
                    and self.step > 0 and self.step % self.cfg.val_every == 0):
                val_metrics = self.validate(val_loader)
                if is_main_process():
                    vmsg = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                    print(f"[val  {self.step:>6}] {vmsg}")

    @torch.no_grad()
    def validate(self, val_loader: Iterable[dict]) -> dict:
        """在验证集上跑 val_steps 个 batch，返回平均 loss 组件。"""
        fwd_model = self._forward_model
        fwd_model.eval()
        accum: dict[str, list] = {}
        n = 0
        for batch in val_loader:
            if n >= self.cfg.val_steps:
                break
            batch = move_to(batch, self.cfg.device)
            with autocast_ctx(self.cfg.bf16, device_type=("cuda" if "cuda" in self.cfg.device else "cpu")):
                if self.cfg.stage == "pretrain":
                    out = fwd_model.pretrain_forward(batch)
                    _, comp = pretrain_loss(out, self.weights, model=self.model, step=self.step)
                else:
                    out = fwd_model(batch)
                    _, comp = finetune_loss(out, self.weights, model=self.model, step=self.step)
            for k, v in comp.items():
                val = v.item() if torch.is_tensor(v) else float(v)
                accum.setdefault(k, []).append(val)
            n += 1
        fwd_model.train()
        return {k: sum(vs) / len(vs) for k, vs in accum.items()}

    def save(self, tag: str) -> str:
        """DDP 时只在 rank 0 保存。"""
        if not is_main_process():
            return ""
        out_dir = os.path.join(self.cfg.ckpt_dir, tag)
        save_sharded(self.model.state_dict(), out_dir)
        verify_shards(out_dir)
        return out_dir
