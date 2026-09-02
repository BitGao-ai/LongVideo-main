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
    """本进程是否已加入进程组（保留旧名，供外部脚本沿用）。"""
    return process_group_ready()


def get_rank() -> int:
    return dist_info()[1]


def get_world_size() -> int:
    return dist_info()[0]


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
    # 检查点是否**只存可训练权重**。None（默认）= 自动：冻结权重能从外部来源重新取回时
    # 才省（判定见 checkpoint.base_weights_recoverable）。
    #   · Qwen3-VL + LoRA → True。基座权重下次 from_pretrained 原样拿回，存它纯属浪费：
    #     4B bf16 ≈ 8.8GB/次，ckpt_every=500 跑 5000 步 ≈ 88GB 磁盘，且 rank0 每次写盘
    #     期间其余 rank 都堵在 barrier 上。可训练部分只有约 150M 参数（≈600MB）。
    #   · stand-in + LoRA → False。那时被冻的是**随机初始化**的 stand-in 基座，
    #     没有任何外部来源能复现，丢了检查点就废了。
    # 显式 True/False 覆盖自动判定（True 时请自行确认冻结权重可复现）。
    save_trainable_only: bool | None = None
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
    # 注：这里**没有** static_graph 开关，是实测后有意不做的。
    # 动机是合理的：finetune 阶段 recon_head / mask_token 确实不参与前向，所以
    # find_unused_parameters=True 关不掉，而它每步都要遍历整张图找未用参数，8 卡上是纯开销；
    # static_graph=True 本可同时解决这两件事（它支持"固定不变的未用参数集"）。
    # 但它与**非重入梯度检查点**不兼容——4 进程 gloo 实测直接抛
    #   RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED (reducer.cpp:1660)
    # 而 eacs_chunk 与 llm.grad_checkpoint 是本项目最主要的两个省显存开关、生产上恒开，
    # 所以这个组合在真实配置下永远不可用。加一个"检测到检查点就拒绝"的开关只是死重量 +
    # 新的踩坑面，故不加。留这段注释是为了别人不必再验一遍。


def move_to(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Trainer:
    def __init__(self, model: nn.Module, cfg: TrainConfig, weights: LossWeights | None = None):
        self.cfg = cfg
        self.weights = weights or LossWeights()
        self.step = 0

        # DDP 初始化（P3）。maybe_init_distributed 是幂等的：训练脚本通常已经在建
        # DataLoader 之前调过一次（分片必须先知道 rank），这里只是兜住"直接 new Trainer"
        # 的调用方。它内部保证 torch.cuda.set_device 先于 init_process_group。
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
                # find_unused_parameters=True 是必需项而非保守选择：finetune 阶段
                # recon_head / mask_token 不参与前向，关掉会在反向报"某些参数没收到梯度"。
                # 唯一的替代品 static_graph 与非重入梯度检查点不兼容，见 TrainConfig 的注释。
                find_unused_parameters=True,
                # 本模型只有 EventGate 的 RunningStandardizer 这类**运行时统计**缓冲，
                # 每次前向都从 rank0 广播一遍既费同步、又让另外 7/8 的数据不参与统计。
                # 关掉后各 rank 各自累计自己分片的统计量，语义更正确（也更省）。
                broadcast_buffers=False)
            if is_main_process():
                print("[dist] DDP: find_unused_parameters=True, broadcast_buffers=False, "
                      "static_graph=False（与梯度检查点不兼容，见 TrainConfig 注释）")
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
            # 并轴扫描的实际状态必须可见：它要求各分支 n_state 相同，而默认的
            # DEFAULT_BRANCHES 是 16/32/64，所以默认配置下这条吞吐优化恒不生效。
            # 不打出来的话，"已经做过并轴优化"会变成一个看不出不成立的假设。
            for ms in self.model.modules():
                if isinstance(ms, MultiScaleEACS):
                    ok, why = ms.merge_status()
                    print(f"[train] EACS 分支并轴扫描: {'启用' if ok else '未启用'} —— {why}")
                    break
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
                    # need_logits=False：本方法只读 out["loss"] 等标量组件，从不碰
                    # out["logits"]。不声明的话 no_grad 会让 LLM 走一次性路径物化整块
                    # (B,T,V)——Qwen3-VL 的 V=151936、B=2/T=8192 约 20 GB，于是训练步
                    # 正常而验证步 OOM。声明后验证也走分块 CE，峰值与 T 无关。
                    # eval_benchmark 的 MCQ 打分不受影响（它按默认 True 调用）。
                    out = fwd_model(batch, stage=self.cfg.stage, need_logits=False)
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
        return self._reduce_metrics(accum, self.cfg.device)

    @staticmethod
    def _reduce_metrics(accum: dict[str, list], device: str = "cpu") -> dict:
        """把各 rank 的验证统计量跨进程求和后再平均。

        分片修好之后这一步就成了必需品：每个 rank 只跑到验证集的 1/world_size，
        直接打印 rank0 的数等于"用 1/8 验证集报指标"，而且各 rank 数值互不相同。
        这里聚合的是 (Σloss, N) 两个量而不是各 rank 的均值——各 rank batch 数可能
        差一个（DistributedSampler 补齐后一般相同，但不该依赖），按均值再平均会给
        样本少的 rank 过高权重。

        device 必须传对：**NCCL 只接受 CUDA 张量**，拿 CPU 张量调 all_reduce 会直接
        报错——而且只在真上 8 卡时才暴露，CPU/gloo 冒烟测试一路绿灯。

        键集合各 rank 必须一致（由 stage + 模型配置决定，天然满足）；这里按 sorted
        固定顺序打包，避免 dict 顺序差异导致 all_reduce 错位。
        """
        keys = sorted(accum)
        if not keys:
            return {}
        local = [[sum(accum[k]), float(len(accum[k]))] for k in keys]
        if process_group_ready():
            dev = device if dist.get_backend() == "nccl" else "cpu"
            t = torch.tensor(local, dtype=torch.float64, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            local = t.cpu().tolist()
        return {k: (s / n if n else float("nan")) for k, (s, n) in zip(keys, local)}

    def _resolve_trainable_only(self) -> bool:
        """本次保存是否只存可训练权重（见 TrainConfig.save_trainable_only）。"""
        if self.cfg.save_trainable_only is not None:
            return bool(self.cfg.save_trainable_only)
        # 自动：有冻结参数 **且** 那些冻结权重能从外部来源重新取回时才省
        return bool(frozen_param_names(self.model)) and base_weights_recoverable(self.model)

    def save(self, tag: str) -> str:
        """DDP 时只在 rank 0 落盘，其余 rank 在栅栏处等它写完。"""
        out_dir = os.path.join(self.cfg.ckpt_dir, tag)
        if is_main_process():
            trainable_only = self._resolve_trainable_only()
            state = (trainable_state_dict(self.model) if trainable_only
                     else self.model.state_dict())
            if trainable_only:
                n_all = len(self.model.state_dict())
                print(f"[train] 检查点只存可训练权重: {len(state)}/{n_all} 个张量"
                      f"（冻结的底座权重下次由 from_pretrained 取回；"
                      f"热启时 load_checkpoint 会把它们报成 missing，属预期）")
            # trainable_only 写进 index.metadata，让检查点自解释：加载侧看到大量 missing
            # 时能分清"这是省下来的底座"还是"架构真的不匹配"。
            save_sharded(state, out_dir, extra_metadata={"trainable_only": trainable_only})
            verify_shards(out_dir)
        # 不加栅栏也不会崩（其余 rank 会在下一次 allreduce 上干等同样久），但那样
        # "检查点已落盘"就没有一个确定的时刻，rank 漂移也更难排查。
        barrier()
        return out_dir if is_main_process() else ""
