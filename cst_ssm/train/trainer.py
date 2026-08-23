"""训练器：两阶段统一循环 + 显存优化 + 分片检查点 + DDP 分布式。

显存优化（需求 2）：梯度累积、bf16 autocast、EACS 分块梯度检查点、LoRA 冻结基座、
门控温度退火；保存时用 save_sharded 保证单文件 ≤4GB。
分布式：支持 torchrun 启动的 DDP 多卡训练（设计文档要求 8×A100）。
"""
from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.distributed as dist

from .losses import LossWeights, finetune_loss, pretrain_loss
from ..modules.eacs import EACSLayer
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
    # EACS 分块梯度检查点。None（默认）= **沿用模型自身的 CSTSSMConfig.eacs_chunk**，
    # 不做任何干预；显式给 int 才覆盖（0=强制关闭，>0=强制开启并指定块大小）。
    # 早期这里默认 0 且无条件覆盖，会把 CSTSSMConfig/YAML 里配好的 chunk 静默清零，
    # 导致省显存开关看似打开、实际从未生效（实测 985MB → 60MB 的差别）。
    eacs_chunk: int | None = None
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

        # DDP 初始化（P3）‰
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

        # 分块梯度检查点：仅在显式指定时覆盖模型自身配置；无论哪条路径都把最终生效值
        # 打出来，避免"以为开了其实没开"（省显存开关必须可见）。
        if cfg.eacs_chunk is not None:
            set_eacs_chunk(self.model, cfg.eacs_chunk)
        if is_main_process():
            chunks = sorted({m.chunk_size for m in self.model.modules()
                             if isinstance(m, EACSLayer)})
            if chunks:
                src = "TrainConfig" if cfg.eacs_chunk is not None else "模型配置"
                state = "关闭" if chunks == [0] else f"块大小={chunks[0] if len(chunks) == 1 else chunks}"
                print(f"[train] EACS 分块梯度检查点: {state}（来自 {src}）")
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
        # DDP 梯度累积：非同步步跳过 all-reduce（no_sync），只在真正 step 时同步一次，
        # 否则每个微批都通信，多卡梯度累积效率大打折扣。
        grad_sync_ctx = nullcontext()
        if (self._ddp_model is not None
                and (self.step + 1) % cfg.grad_accum != 0):
            grad_sync_ctx = self._ddp_model.no_sync()
        with autocast_ctx(cfg.bf16, device_type=("cuda" if "cuda" in cfg.device else "cpu")), grad_sync_ctx:
            # 统一走 forward(batch, stage=...)：DDP 只有经 DDP.forward 才会挂上梯度同步 hook，
            # 直接调 pretrain_forward/grounding_forward 既会 AttributeError，绕过后又会静默丢同步。
            out = fwd_model(batch, stage=cfg.stage)
            if cfg.stage == "pretrain":
                loss, comp = pretrain_loss(out, self.weights, model=self.model, step=self.step)
            elif cfg.stage == "grounding":
                loss = out["loss"]
                comp = {"grounding": out["grounding_loss"], "total": out["loss"].detach()}
            else:
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
        """在验证集上跑 val_steps 个 batch，返回平均 loss 组件。

        进出的 train/eval 模式必须守恒：此前结束时无条件 .train()，评测脚本只要在推理
        循环里插一次 validate，之后的推理就会跑在 train 模式下——EACS 走软门控、
        RunningStandardizer 继续更新统计、门控温度语义改变，指标静默偏移且不报错。
        """
        fwd_model = self._forward_model
        was_training = fwd_model.training
        fwd_model.eval()
        accum: dict[str, list] = {}
        n = 0
        try:
            for batch in val_loader:
                if n >= self.cfg.val_steps:
                    break
                batch = move_to(batch, self.cfg.device)
                with autocast_ctx(self.cfg.bf16, device_type=("cuda" if "cuda" in self.cfg.device else "cpu")):
                    out = fwd_model(batch, stage=self.cfg.stage)
                    if self.cfg.stage == "pretrain":
                        _, comp = pretrain_loss(out, self.weights, model=self.model, step=self.step)
                    elif self.cfg.stage == "grounding":
                        # 与 train_step 对齐：grounding 阶段必须走 grounding_forward，
                        # 否则会拿 QA 前向算 BCE 组件（输出无 grounding_loss 且语义错误）
                        comp = {"grounding": out["grounding_loss"], "total": out["loss"].detach()}
                    else:
                        _, comp = finetune_loss(out, self.weights, model=self.model, step=self.step)
                for k, v in comp.items():
                    val = v.item() if torch.is_tensor(v) else float(v)
                    accum.setdefault(k, []).append(val)
                n += 1
        finally:
            fwd_model.train(was_training)
        return {k: sum(vs) / len(vs) for k, vs in accum.items()}

    def save(self, tag: str) -> str:
        """DDP 时只在 rank 0 保存。"""
        if not is_main_process():
            return ""
        out_dir = os.path.join(self.cfg.ckpt_dir, tag)
        save_sharded(self.model.state_dict(), out_dir)
        verify_shards(out_dir)
        return out_dir
