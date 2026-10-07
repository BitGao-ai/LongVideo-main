"""Unified training loop with memory optimizations and DDP support."""
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
from ..modules.multiscale import MultiScaleEACS
from ..utils.memory import (set_eacs_chunk, set_gate_temperature, autocast_ctx,
                            build_optimizer, gate_temperature_schedule,
                            set_eps_anneal, eps_anneal_schedule)
from ..utils.checkpoint import (save_sharded, verify_shards, trainable_state_dict,
                                frozen_param_names, base_weights_recoverable)
from ..utils.distributed import (maybe_init_distributed, resolve_device,
                                 env_local_rank, process_group_ready,
                                 dist_info, is_main_process, barrier)


def is_distributed() -> bool:
    """True when this process has joined a process group (legacy alias)."""
    return process_group_ready()


def get_rank() -> int:
    return dist_info()[1]


def get_world_size() -> int:
    return dist_info()[0]


@dataclass
class TrainConfig:
    stage: str = "finetune"          # "finetune" | "pretrain" | "grounding"
    lr: float = 1e-4
    weight_decay: float = 0.05
    grad_accum: int = 1
    max_steps: int = 1000            # counts micro-batches, not optimizer steps
    grad_clip: float = 1.0
    bf16: bool = True
    device: str = "cuda"
    eacs_chunk: int | None = None    # None = keep model config; 0 = disable; >0 = override
    log_every: int = 10
    ckpt_every: int = 0
    ckpt_dir: str = "checkpoints"
    save_trainable_only: bool | None = None  # None = auto by weight recoverability
    t_start: float = 1.0
    t_end: float = 0.05
    eps_anneal_start: float = 1.0
    eps_anneal_end: float = 1.0
    val_every: int = 0
    val_steps: int = 50
    ddp: bool = False
    ddp_backend: str = "nccl"        # nccl (GPU) | gloo (CPU)
    seed: int = 0
    # No static_graph option by design: it conflicts with non-reentrant
    # gradient checkpointing under DDP with find_unused_parameters=True.


def move_to(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, weights: LossWeights | None = None):
        self.cfg = cfg
        if cfg.grad_accum < 1:
            raise ValueError("grad_accum must be positive")
        self._accumulated = 0
        import inspect
        self._cpib_forward = "cpib_weights" in inspect.signature(model.forward).parameters
        self.weights = weights or LossWeights()
        adaptive = (getattr(getattr(model, "cfg", None), "cpib_distill", False)
                    and getattr(model.cfg, "cpib_mode", "legacy") == "adaptive")
        if adaptive and self.weights.cpib is None:
            from .contrastive import CPIBWeights
            self.weights.cpib = CPIBWeights()
        self.step = 0

        self._ddp_model = None
        if cfg.ddp:
            maybe_init_distributed(True, backend=cfg.ddp_backend, device=cfg.device)
            local_rank = env_local_rank()
            device = resolve_device(cfg.device)
            if device.startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"
            self.model = model.to(device)
            self._ddp_model = nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank] if device.startswith("cuda") else None,
                find_unused_parameters=True,
                broadcast_buffers=False)
            if is_main_process():
                print("[dist] DDP: find_unused_parameters=True, broadcast_buffers=False")
            cfg.device = device
        else:
            device = cfg.device if (torch.cuda.is_available() or cfg.device == "cpu") else "cpu"
            self.model = model.to(device)
            cfg.device = device

        if cfg.eacs_chunk is not None:
            set_eacs_chunk(self.model, cfg.eacs_chunk)
        if is_main_process():
            if adaptive:
                print("[train] adaptive CPIB uses causal per-frame recurrence; "
                      "legacy EACS chunk/merged/fused flags do not accelerate this path. "
                      "Use stream_visual + explicit state.detach() for truncated BPTT.")
            if not adaptive:
                chunks = sorted({m.chunk_size for m in self.model.modules()
                                 if isinstance(m, EACSLayer)})
                if chunks:
                    src = "TrainConfig" if cfg.eacs_chunk is not None else "model config"
                    state = "off" if chunks == [0] else f"chunk={chunks[0] if len(chunks) == 1 else chunks}"
                    print(f"[train] EACS chunked checkpoint: {state} (from {src})")
                for ms in self.model.modules():
                    if isinstance(ms, MultiScaleEACS):
                        ok, why = ms.merge_status()
                        print(f"[train] EACS branch merge: {'on' if ok else 'off'} - {why}")
                        break
            world, _ = dist_info()
            eff_batch = getattr(cfg, "_eff_batch", None)
            if world > 1 and eff_batch:
                print(f"[train] global batch = {eff_batch} "
                      f"(per-rank batch x world={world} x grad_accum={cfg.grad_accum})")
        self.opt = build_optimizer(self.model, cfg.lr, cfg.weight_decay)

    @property
    def _forward_model(self):
        return self._ddp_model if self._ddp_model is not None else self.model

    def _aux_options(self):
        return ({"cpib_weights": self.weights.cpib, "step": self.step}
                if self._cpib_forward else {})

    def train_step(self, batch: dict, force_step: bool = False) -> dict:
        cfg = self.cfg
        fwd_model = self._forward_model
        fwd_model.train()
        batch = move_to(batch, cfg.device)
        set_gate_temperature(self.model, gate_temperature_schedule(
            self.step, cfg.max_steps, cfg.t_start, cfg.t_end))
        set_eps_anneal(self.model, eps_anneal_schedule(
            self.step, cfg.max_steps, cfg.eps_anneal_start, cfg.eps_anneal_end))
        self._accumulated += 1
        take_step = (self._accumulated == cfg.grad_accum or force_step
                     or self.step + 1 >= cfg.max_steps)
        grad_sync_ctx = nullcontext()
        if self._ddp_model is not None and not take_step:
            grad_sync_ctx = self._ddp_model.no_sync()
        with autocast_ctx(cfg.bf16, device_type=("cuda" if "cuda" in cfg.device else "cpu")), grad_sync_ctx:
            out = fwd_model(batch, stage=cfg.stage, **self._aux_options())
            if cfg.stage == "pretrain":
                loss, comp = pretrain_loss(out, self.weights, model=self.model, step=self.step)
            elif cfg.stage == "grounding":
                loss = out["loss"]
                comp = {"grounding": out["grounding_loss"], "total": out["loss"].detach()}
            else:
                loss, comp = finetune_loss(out, self.weights, model=self.model, step=self.step)
            (loss / cfg.grad_accum).backward()
        if take_step:
            if self._accumulated != cfg.grad_accum:
                scale = cfg.grad_accum / self._accumulated
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],
                                         cfg.grad_clip)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self._accumulated = 0
        self.step += 1
        return {k: (v.item() if torch.is_tensor(v) else v) for k, v in comp.items()}

    def fit(self, data: Iterable[dict], val_loader: Iterable[dict] | None = None) -> None:
        if self.step >= self.cfg.max_steps:
            return
        iterator, end = iter(data), object()
        batch = next(iterator, end)
        while batch is not end and self.step < self.cfg.max_steps:
            following = (next(iterator, end) if self.step + 1 < self.cfg.max_steps else end)
            logs = self.train_step(batch, force_step=following is end)
            batch = following
            if self.step % self.cfg.log_every == 0 and is_main_process():
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"[step {self.step:>6}] {msg}")
            if self.cfg.ckpt_every and self.step % self.cfg.ckpt_every == 0:
                self.save(f"step{self.step}")
            if (self.cfg.val_every and val_loader is not None
                    and self.step > 0 and self.step % self.cfg.val_every == 0):
                val_metrics = self.validate(val_loader)
                if is_main_process():
                    vmsg = " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
                    print(f"[val  {self.step:>6}] {vmsg}")

    @torch.no_grad()
    def validate(self, val_loader: Iterable[dict]) -> dict:
        """Average loss components over val_steps batches; restores train mode."""
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
                    out = fwd_model(batch, stage=self.cfg.stage, need_logits=False,
                                    **self._aux_options())
                    if self.cfg.stage == "pretrain":
                        _, comp = pretrain_loss(out, self.weights, model=self.model, step=self.step)
                    elif self.cfg.stage == "grounding":
                        comp = {"grounding": out["grounding_loss"], "total": out["loss"].detach()}
                    else:
                        _, comp = finetune_loss(out, self.weights, model=self.model, step=self.step)
                for k, v in comp.items():
                    val = v.item() if torch.is_tensor(v) else float(v)
                    accum.setdefault(k, []).append(val)
                n += 1
        finally:
            fwd_model.train(was_training)
        return self._reduce_metrics(accum, self.cfg.device)

    @staticmethod
    def _reduce_metrics(accum: dict[str, list], device: str = "cpu") -> dict:
        """Average per-rank (sum, count) pairs via all-reduce; handles uneven shards."""
        keys = sorted(accum)
        if process_group_ready():
            rank_keys = [None] * dist.get_world_size()
            dist.all_gather_object(rank_keys, keys)
            keys = sorted({k for ks in rank_keys for k in ks})
        if not keys:
            return {}
        local = [[sum(accum.get(k, [])), float(len(accum.get(k, [])))] for k in keys]
        if process_group_ready():
            dev = device if dist.get_backend() == "nccl" else "cpu"
            t = torch.tensor(local, dtype=torch.float64, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            local = t.cpu().tolist()
        return {k: (s / n if n else float("nan")) for k, (s, n) in zip(keys, local)}

    def _resolve_trainable_only(self) -> bool:
        if self.cfg.save_trainable_only is not None:
            return bool(self.cfg.save_trainable_only)
        return bool(frozen_param_names(self.model)) and base_weights_recoverable(self.model)

    def save(self, tag: str) -> str:
        """Rank 0 writes shards; other ranks wait at a barrier."""
        out_dir = os.path.join(self.cfg.ckpt_dir, tag)
        if is_main_process():
            trainable_only = self._resolve_trainable_only()
            state = (trainable_state_dict(self.model) if trainable_only
                     else self.model.state_dict())
            if trainable_only:
                n_all = len(self.model.state_dict())
                print(f"[train] checkpoint stores trainable weights only: "
                      f"{len(state)}/{n_all} tensors")
            from dataclasses import asdict, is_dataclass
            from ..utils.restoration import restoration_metadata
            metadata = {"trainable_only": trainable_only,
                        "restoration": restoration_metadata(self.model)}
            model_cfg = getattr(self.model, "cfg", None)
            if is_dataclass(model_cfg):
                metadata["model_config"] = asdict(model_cfg)
            save_sharded(state, out_dir, extra_metadata=metadata)
            verify_shards(out_dir)
        barrier()
        return out_dir if is_main_process() else ""
