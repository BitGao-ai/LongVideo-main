"""CST-SSM 端到端顶层模型：四段式架构（设计方案 §1.1）。

    帧/特征 ──▶ 空间编码 ──▶ 连续时序建模(EACS) ──▶ 跨模态投影 ──▶ LLM(交叉注意力)
                (spatial)      (MultiScaleEACS)        (projector)     (visual_states)

全程无外挂采样/分块，训练与推理逻辑一致。forward 返回 logits/loss 及供正则的辅助量
（有效更新率、谱正则、逐分支更新率、门控），stage-1 自监督入口 pretrain_forward 单列。
高内聚低耦合：各段仅通过张量接口耦合；空间骨干/LLM 可按协议替换。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..ops.spectral_init import BranchSpec, DEFAULT_BRANCHES
from ..modules.spatial_encoder import WindowedSpatialEncoder, FeatureAdapter
from ..modules.multiscale import MultiScaleEACS
from ..modules.projector import CrossModalProjector
from ..modules.cgu import CPIBDistill
from ..modules.llm_interface import VisualConditionedLM, LLMConfig


@dataclass
class CSTSSMConfig:
    # 输入模式："feature"（读预抽取特征，内存友好，推荐）| "pixel"（从像素帧端到端）
    input_mode: str = "feature"
    # 空间编码（像素模式）
    patch: int = 16
    window: int = 7
    vis_depth: int = 4
    vis_heads: int = 6
    # 特征模式：预抽取特征维度
    feat_dim: int = 768
    # 时序建模
    d_model: int = 384
    branches: tuple[BranchSpec, ...] = DEFAULT_BRANCHES
    gate_init_eps: float = 0.1
    eacs_chunk: int = 32               # >0 启用分块梯度检查点；0=关闭
    # ↑ 训练显存的**主开关**，也是唯一的数量级杠杆。实测（d_model=384、三分支、B=1）：
    #     chunk=0  → 5.32 MB/帧/样本      chunk=32 → 0.52 MB/帧/样本（约 1/10）
    #     代价：单步 1669ms → 2064ms（+24%，反向重算一遍块内前向）
    #   L=512、batch=8 时就是 21.8 GB vs 2.1 GB 的差别——长视频训练几乎总要开。
    #   默认取 32 而不是 0：本模型的目标场景就是长视频，"默认能跑通"比"默认最快"重要；
    #   数值上与 chunk=0 **逐位相同**（loss/update_rate/全部参数梯度实测 Δ=0，含
    #   robust_guard 开启时），所以打开它不改变任何结果，只换时间省显存。
    #   显式设 0 可关闭（tests/_baseline_snapshot.py 就这么做，以隔离检查点变量）。
    #   见 tests/profile_memory.py 复现这两组数字。
    eacs_fused_train: bool = False     # True：EACS 训练走可微融合 cell（常数图显存）
    eacs_robust_guard: bool = False    # True：启用 §4.4 残差暴涨鲁棒兜底
    # 消融开关（§6.3）：默认即完整 CST-SSM
    eacs_disc_mode: str = "continuous"     # continuous | fixed | learned
    eacs_gate_kind: str = "event"          # event | random | always
    eacs_use_spectral_init: bool = True    # 谱初始化 vs 随机初始化
    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)
    # stage-1 预测头
    pretrain_horizon: int = 1          # 预测未来第 s 帧特征（s∈[1,horizon]）
    mask_ratio: float = 0.15           # stage-1 掩码帧重构比例（设计 §4.4）
    # 可选定位打分头（CSG/VFR 亚帧定位；默认关，开启后随 state_dict 保存/加载）
    grounding_head: bool = False
    # 创新点1：CPIB-Distill（CGU + Token 蒸馏 + CPI 信号）
    cpib_distill: bool = False           # 开启后在空间编码与时序建模之间插入 CGU 蒸馏
    cpib_hidden: int = 256               # CGU MLP 隐藏维
    cpib_ema_alpha: float = 0.9          # 因果 EMA 衰减系数
    cpib_context_mode: str = "ema"       # "ema"(默认) | "ssm"(用 EACS 提交状态作为因果上下文)
    cpi_modulation: float = 0.5          # CPI 对 EventGate 阈值的调制强度（P1.5，0=不启用）
    # 创新点3：误差有界差分 KV 缓存（推理侧效率）
    diff_kv: bool = False                # 开启后附带低秩残差预测头（训练时学习重构）
    diff_kv_rank: int = 8                # 低秩残差秩 r
    diff_kv_interval: int = 8            # 基准帧间隔 G
    diff_kv_threshold: float = 0.1       # 残差能量阈值 ε
    diff_kv_cpi_sparse: bool = True      # 是否启用 CPI 稀疏注意力
    diff_kv_keep_ratio: float = 0.3      # 高 CPI token 保留比例


def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """按 (B,L) 有效位求 MSE。mask=None 时退化为普通 mse_loss。

    变长 batch 里 collate 把 padding 帧的特征填 0、时间戳复制最后一个有效值；不掩码就等于
    把这些位置当成真实的预测目标（实测 8 帧含 4 帧 padding 会让 pred_loss 偏高约 29%），
    且偏差随批内帧数差异放大——长视频数据集里这是常态。
    """
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.to(pred.dtype).unsqueeze(-1)                  # (B,L,1)
    denom = (m.sum() * pred.shape[-1]).clamp_min(1.0)
    return ((pred - target) ** 2 * m).sum() / denom


class CSTSSMModel(nn.Module):
    def __init__(self, cfg: CSTSSMConfig,
                 vision: nn.Module | None = None,
                 llm: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # 段 1：空间编码 → (B,L,d_model)。可注入外部视觉骨干（如 Qwen3-VL 视觉塔）
        if vision is not None:
            self.vision = vision
        elif cfg.input_mode == "pixel":
            self.vision = WindowedSpatialEncoder(
                dim=d, patch=cfg.patch, window=cfg.window,
                depth=cfg.vis_depth, n_heads=cfg.vis_heads)
        else:
            self.vision = FeatureAdapter(d_in=cfg.feat_dim, d_out=d)

        # 创新点1：CPIB-Distill（CGU + Token 蒸馏），插入段1与段2之间
        self.cpib: CPIBDistill | None = None
        self.cpib_critic = None                             # InfoNCE 双线性 critic
        if cfg.cpib_distill:
            # SSM 上下文模式需要状态维度：取第一分支的 H*N*2（实部+虚部）
            ssm_dim = 0
            if cfg.cpib_context_mode == "ssm":
                first_branch = cfg.branches[0]
                ssm_dim = d * first_branch.n_state * 2  # H=d_model, 复数=2x
            self.cpib = CPIBDistill(d=d, hidden=cfg.cpib_hidden, ema_alpha=cfg.cpib_ema_alpha,
                                    context_mode=cfg.cpib_context_mode, ssm_state_dim=ssm_dim)
            from ..train.contrastive import BilinearCritic
            self.cpib_critic = BilinearCritic(d)

        # 段 2：连续时序建模（EACS 多尺度）
        gate_kw = dict(init_eps=cfg.gate_init_eps, gate_kind=cfg.eacs_gate_kind)
        if cfg.cpib_distill and cfg.cpi_modulation > 0:
            gate_kw["cpi_modulation"] = cfg.cpi_modulation   # P1.5：CPI 贯通 EventGate
        self.temporal = MultiScaleEACS(
            d, branches=cfg.branches,
            gate_kwargs=gate_kw,
            chunk_size=cfg.eacs_chunk, fused_train=cfg.eacs_fused_train,
            robust_guard=cfg.eacs_robust_guard,
            disc_mode=cfg.eacs_disc_mode, use_spectral_init=cfg.eacs_use_spectral_init)

        # 段 3：跨模态投影 → LLM 空间
        self.projector = CrossModalProjector(d, cfg.llm.dim)

        # 段 4：LLM。可注入外部 LLM（如 Qwen3-VL），否则用自包含 stand-in
        self.llm = llm if llm is not None else VisualConditionedLM(cfg.llm)

        # stage-1 自监督：未来帧特征预测 + 掩码帧重构
        self.pred_head = nn.Linear(d, d)
        self.recon_head = nn.Linear(d, d)                 # 掩码帧重构头（设计 §4.4）
        self.mask_token = nn.Parameter(torch.zeros(d))    # 掩码占位向量

        # 可选：定位打分头（连续查询 + query 条件相关度；供 CSG/VFR 亚帧定位）
        self.grounding_head = None
        if cfg.grounding_head:
            from ..modules.grounding_head import GroundingHead
            self.grounding_head = GroundingHead(d_model=d, d_txt=cfg.llm.dim)

        # 创新点3：差分 KV 缓存（推理侧效率，训练时学习低秩重构）+ CPI 稀疏注意力
        self.diff_kv = None
        self.cpi_sparse_attn = None
        if cfg.diff_kv:
            from ..modules.diff_kv import DifferentialKVCache, DiffKVConfig, CPISparseAttention
            # 差分 KV / 稀疏注意力都作用于投影后的 visual_states（LLM 维），非 d_model
            self.diff_kv = DifferentialKVCache(
                DiffKVConfig(rank=cfg.diff_kv_rank, base_interval=cfg.diff_kv_interval,
                             error_threshold=cfg.diff_kv_threshold,
                             cpi_sparse=cfg.diff_kv_cpi_sparse,
                             cpi_keep_ratio=cfg.diff_kv_keep_ratio),
                d_model=cfg.llm.dim)
            # CPI 稀疏注意力（因果）：对投影后的视觉状态（LLM 维，非 d_model）做稀疏压缩后再喂 LLM。
            # 受 diff_kv_cpi_sparse 控制——此前这个开关传进 DiffKVConfig 后从没被读过，
            # 稀疏注意力只要 diff_kv=True 就无条件构造、关不掉。
            if cfg.diff_kv_cpi_sparse:
                sparse_dim = cfg.llm.dim
                sparse_heads = next(h for h in (8, 4, 2, 1) if sparse_dim % h == 0)
                self.cpi_sparse_attn = CPISparseAttention(
                    d=sparse_dim, n_heads=sparse_heads,
                    keep_ratio=cfg.diff_kv_keep_ratio, causal=True)

    # ---------- 段 1：视觉前端（所有阶段共用的唯一入口）----------
    def _frontend(self, vis_in: Tensor, timestamps: Tensor):
        """视觉输入 → 喂给 EACS 的帧级表征 + CPI 信号。返回 (feat, cpi_frame, cpi_tokens, tokens)。

        **所有会跑 EACS 的路径都必须经过这里**：QA/微调（encode_temporal）、预训练
        （pretrain_forward）、定位读出（continuous_readout）。

        此前定位路径直接用 self.vision(vis_in)（FeatureAdapter 的注意力池化输出），而
        QA 路径在 cpib_distill=True 时喂的是 self.cpib(get_tokens(...))（CGU 蒸馏输出），
        两者不是同一个表征空间（实测 max|Δ|=0.95）。后果是 stage-2 在 CPIB 表征上训好的
        EACS 权重，到了定位训练/推理却跑在 attn-pool 表征上——迁移收益被静默吃掉，而 BCE
        照样下降，从 loss 曲线上完全看不出来。CSG / VFR 是本文的杀手锏基准，这条必须统一。
        """
        cpi_frame, cpi_tokens, tokens = None, None, None
        ssm_context = None
        if self.cpib is not None and hasattr(self.vision, "get_tokens"):
            # CPIB-Distill 路径：token 蒸馏 → 帧级特征 + CPI 信号
            tokens = self.vision.get_tokens(vis_in)       # (B,L,P,d)
            # SSM 上下文模式：先用 EMA 上下文跑一次蒸馏得初始表征，再用它跑 EACS
            # 取提交状态作为 CGU 的因果上下文（两遍，无冗余）。
            # 注：run_with_commits 内部自带 norm，外部不得再 norm（避免双重 LayerNorm）；
            #     扫描在无梯度下进行，投影层 project_ssm_states 在其外调用 → 可训练。
            if (self.cpib.context_mode == "ssm" and self.cpib.ssm_ctx is not None):
                # 这一遍"蒸馏 + 扫描"只为拿到提交状态当因果上下文，结果不进任何损失：
                # commits 在 no_grad 下产出，通往 ssm_context 的梯度是从
                # project_ssm_states 这个线性层才开始的。此前 feat_init 是在**带梯度**下
                # 算的，于是 CGU+TokenDistiller 的整张计算图被建起来、存好激活，然后
                # 原样丢掉——一份纯浪费的显存与建图开销。整段包进 no_grad 后前向数值不变。
                with torch.no_grad():
                    feat_init, _, _ = self.cpib(tokens, ssm_context=None)
                    _, commits = self.temporal.branches[0].run_with_commits(
                        feat_init, timestamps)
                ssm_context = self.cpib.project_ssm_states(commits["h"])  # (B,L,d)
            feat, cpi_frame, cpi_tokens = self.cpib(tokens, ssm_context=ssm_context)
        else:
            feat = self.vision(vis_in)                    # (B,L,d_model)
        return feat, cpi_frame, cpi_tokens, tokens

    # ---------- 段 1+2：视觉→时序状态 ----------
    def encode_temporal(self, batch: dict):
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        feat, cpi_frame, cpi_tokens, tokens = self._frontend(vis_in, batch["timestamps"])
        ms = self.temporal(feat, batch["timestamps"], batch.get("frame_mask"),
                           cpi=cpi_frame)  # MultiScaleOutput，CPI 贯通 EventGate
        return feat, ms, cpi_frame, cpi_tokens, tokens

    # ---------- 统一前向入口（stage 分派）----------
    def forward(self, batch: dict, stage: str = "finetune") -> dict:
        """所有训练/推理阶段的**唯一**入口。

        必须走 forward 而不是直接调 pretrain_forward/grounding_forward：
        DistributedDataParallel 不代理自定义方法名（会 AttributeError），更要命的是即使
        用 .module 绕过属性问题，绕开 DDP.forward 就不会挂上反向的 all-reduce hook——
        各卡梯度不同步，多卡训练静默退化成 N 个独立单卡训练，且不会有任何报错。
        """
        if stage == "pretrain":
            return self.pretrain_forward(batch)
        if stage == "grounding":
            return self.grounding_forward(batch)
        return self.finetune_forward(batch)

    # ---------- 段 1+2+3：视觉 → 时序 → 投影（喂给 LLM 的 visual_states）----------
    def encode_visual(self, batch: dict):
        """返回 (visual_states, aux)。aux 含 feat/ms/cpi_* 供损失与诊断使用。

        单独拆出来是为了"同一段视频、多个候选文本"的场景（多选题逐选项打分、
        同一视频多个 query）：视觉侧与文本无关，只需跑一次 O(L) 扫描，
        之后对 LLM 循环即可，而不是每个选项都重跑一遍整条视觉+时序链路。
        """
        feat, ms, cpi_frame, cpi_tokens, tokens = self.encode_temporal(batch)
        visual_states = self.projector(ms.y)           # (B,L,d_llm)
        # 创新点3：CPI 稀疏注意力（高 CPI 帧走因果注意力，低 CPI 用因果摘要替代）
        if self.cpi_sparse_attn is not None and cpi_frame is not None:
            visual_states = self.cpi_sparse_attn(visual_states, cpi=cpi_frame)
        return visual_states, dict(feat=feat, ms=ms, cpi_frame=cpi_frame,
                                   cpi_tokens=cpi_tokens, tokens=tokens)

    # ---------- 端到端任务前向（stage-2 / 推理）----------
    def finetune_forward(self, batch: dict) -> dict:
        visual_states, aux = self.encode_visual(batch)
        feat, ms = aux["feat"], aux["ms"]
        cpi_frame, cpi_tokens, tokens = aux["cpi_frame"], aux["cpi_tokens"], aux["tokens"]
        diff_loss = None
        if self.diff_kv is not None and self.training:
            # 训练：无状态的序列重构损失，让低秩头学会重构 K/V 残差。
            # 本架构 LLM 注意力前无独立 K/V，用 visual_states 作 K/V 代理（双头共享同一信号，
            # 但编码/基独立，学到的是残差低秩重构能力本身）
            diff_loss = self.diff_kv.sequence_reconstruction_loss(
                visual_states, visual_states)
        # 推理侧不在 forward 里跑流式缓存：此前那个逐帧 update 循环的返回值不被接收、
        # 缓存也不参与后面的 self.llm(...)，是纯粹的空转，且每帧一次 .item() 会在 GPU 上
        # 强制同步 L 次（实测 L=128 多花 25% 推理时间）。需要压缩率时显式调
        # measure_kv_compression()。

        out = self.llm(
            input_ids=batch["input_ids"],
            visual_states=visual_states,
            attention_mask=batch.get("attention_mask"),
            visual_mask=batch.get("frame_mask"),
            labels=batch.get("labels"),
        )
        out.update(
            update_rate=ms.update_rate,
            per_branch_update_rate=ms.per_branch_update_rate,
            gates=ms.gates,
            spectral_reg=self.temporal.spectral_reg(),
            residual=ms.residual,
            frame_mask=batch.get("frame_mask"),   # 透传有效帧 mask（CPIB 损失用）
        )
        # 创新点3 辅助损失（训练时）
        if diff_loss is not None:
            out["diff_kv_loss"] = diff_loss
        # CPI 信号输出（供创新点1 损失 + P1.5 EventGate 复用）
        if cpi_frame is not None:
            out["cpi_frame"] = cpi_frame
            out["cpi_tokens"] = cpi_tokens
            out["cpib_tokens_raw"] = tokens       # 原始 token（反事实损失用）
            # 蒸馏后的帧级表征：这里的 feat 就是 TokenDistiller 的输出。透传出去让
            # losses._compute_cpib 直接复用，避免它为了拿同一个量再跑一遍 distiller
            # （(B·L,P,d) 的注意力池化，每步白算一次）。
            out["cpib_frame_feat"] = feat
        # 辅助未来预测损失（stage-2 总损失的 L_pred 项，设计 §4.4）
        # 目标是"帧 t 的状态预测帧 t+s 的特征"，因此有效位 = t 与 t+s 都是有效帧
        s = self.cfg.pretrain_horizon
        if feat.shape[1] > s:
            pred = self.pred_head(ms.y)
            fm = batch.get("frame_mask")
            pm = (fm[:, :-s] & fm[:, s:]) if fm is not None else None
            out["pred_loss"] = _masked_mse(pred[:, :-s], feat[:, s:].detach(), pm)
        return out

    # ---------- 创新点3：差分 KV 缓存的效率度量（显式调用，不在 forward 热路径）----------
    @torch.no_grad()
    def measure_kv_compression(self, batch: dict) -> dict:
        """流式跑一遍差分 KV 缓存，返回压缩统计 + 重构误差（效率评测/消融用）。

        返回 {n_frames, n_base, n_residual, compression_ratio, recon_mse}。
        recon_mse 是历史中间帧从 (基准 + r 维系数) 重建后与真值的均方误差——
        压缩率要和它一起看才有意义（只报压缩率可以靠疯狂丢信息刷好看）。
        """
        assert self.diff_kv is not None, "未启用 diff_kv（CSTSSMConfig.diff_kv=True）"
        visual_states, aux = self.encode_visual(batch)
        feat = aux["feat"]

        self.diff_kv.reset()
        residual_truth = []           # 走了残差路径（非基准）的帧的真值，按顺序
        for t in range(visual_states.shape[1]):
            v = visual_states[:, t]
            n_base_before = self.diff_kv.n_base_frames
            self.diff_kv.update(v, v, feat[:, t] if feat.dim() == 3 else feat, t)
            if self.diff_kv.n_base_frames == n_base_before:
                residual_truth.append(v)

        stats = self.diff_kv.stats
        K_rec, _ = self.diff_kv.reconstruct_residual_frames()
        stats["recon_mse"] = (float(F.mse_loss(K_rec, torch.stack(residual_truth)).item())
                              if K_rec is not None else 0.0)
        return stats

    # ---------- stage-1 自监督：未来帧预测 + 掩码帧重构 ----------
    def pretrain_forward(self, batch: dict) -> dict:
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        # 与 QA / 定位共用同一个前端（见 _frontend）。CPIB 路径下输入与预测/重构目标都取
        # **蒸馏后的帧级表征**，同一表征空间；旧写法目标用 self.vision(vis_in)（注意力池化
        # 空间）、输入用 CGU 蒸馏空间，任务变成"跨空间翻译"，而且 pred_head 在 stage-1 学的
        # 目标与 stage-2 forward 里 pred_loss 的目标（就是 CPIB 输出）根本不是同一个东西。
        feat_target, cpi_frame, cpi_tokens, cpib_tokens_raw = self._frontend(
            vis_in, batch["timestamps"])
        # 掩码之前的蒸馏输出，供 CPIB 损失复用（非 CPIB 路径下无此概念）
        cpib_frame_feat = feat_target if cpib_tokens_raw is not None else None
        feat_in = feat_target
        B, L, d = feat_target.shape
        # 掩码帧：随机把一部分帧输入替换为 mask_token，让模型从时序上下文重构
        # 只在有效帧（frame_mask=True）上掩码，避免 padding 帧成为重构目标
        valid = batch.get("frame_mask", torch.ones(B, L, dtype=torch.bool, device=feat_in.device))
        if self.training and self.cfg.mask_ratio > 0 and L > 1:
            mask = (torch.rand(B, L, device=feat_in.device) < self.cfg.mask_ratio) & valid
            feat_in = torch.where(mask.unsqueeze(-1), self.mask_token.to(feat_in.dtype), feat_in)
        else:
            mask = torch.zeros(B, L, dtype=torch.bool, device=feat_in.device)
        # cpi 必须与 stage-2 同口径地送进 EventGate：此前 stage-1 不传，两阶段门控动力学
        # 不一致——stage-1 学到的 ε / log_dt_scale / λ 是在「无 CPI 调制」下收敛的，
        # stage-2 一开始就换了门控规则；消融组 5（CPI 贯通 vs 独立信号）也只覆盖了一半流程。
        # 注意 cpi 取自**未掩码**的前端输出：CPI 是输入帧的内容显著性，掩码是重构任务的
        # 人为扰动，让 mask_token 去改变门控会把两件事混在一起。
        ms = self.temporal(feat_in, batch["timestamps"], batch.get("frame_mask"),
                           cpi=cpi_frame)

        s = self.cfg.pretrain_horizon
        pred = self.pred_head(ms.y)                     # 未来预测
        fm = batch.get("frame_mask")
        pm = (fm[:, :-s] & fm[:, s:]) if (fm is not None and L > s) else None
        pred_loss = (_masked_mse(pred[:, :-s], feat_target[:, s:].detach(), pm)
                     if L > s else pred.sum() * 0.0)
        recon = self.recon_head(ms.y)                  # 掩码帧重构（mask 已 & valid）
        recon_loss = (F.mse_loss(recon[mask], feat_target.detach()[mask])
                      if mask.any() else recon.sum() * 0.0)
        out = {
            "loss": pred_loss + recon_loss,
            "pred_loss": pred_loss,
            "recon_loss": recon_loss,
            "update_rate": ms.update_rate,
            "spectral_reg": self.temporal.spectral_reg(),
            "frame_mask": batch.get("frame_mask"),   # 透传有效帧 mask（CPIB 损失用）
        }
        if cpi_frame is not None:
            out["cpi_frame"] = cpi_frame
            out["cpi_tokens"] = cpi_tokens
            out["cpib_tokens_raw"] = cpib_tokens_raw
            out["cpib_frame_feat"] = cpib_frame_feat
        return out

    # ---------- 定位（连续查询 + query 条件打分头）----------
    def _embed_text(self, input_ids: Tensor) -> Tensor:
        """取 LLM 的 token 嵌入 (B,T,d_txt)，兼容自带 stand-in 与注入的 HF LLM。"""
        llm = self.llm
        if hasattr(llm, "embed"):                        # VisualConditionedLM
            return llm.embed(input_ids)
        if hasattr(llm, "get_input_embeddings"):         # HF 因果 LM
            return llm.get_input_embeddings()(input_ids)
        if hasattr(llm, "hf"):                           # Qwen3VLLanguageModel 包装
            return llm.hf.get_input_embeddings()(input_ids)
        raise AttributeError("无法获取 LLM 文本嵌入；请为注入的 LLM 提供 embed/get_input_embeddings。")

    def continuous_readout(self, features: Tensor, timestamps: Tensor, t_query: Tensor) -> Tensor:
        """在任意时刻 t_query 求连续读出，按 fusion 权重融合各尺度分支 → (B,Q,d_model)。

        前端与聚合口径都与主前向对齐：

        ① 表征（N3）：走 self._frontend，cpib_distill=True 时喂给 EACS 的是 CGU 蒸馏输出，
           与 encode_temporal / pretrain_forward 同一空间；此前这里直接用 self.vision(features)
           的注意力池化输出，同一套 EACS 权重被喂了两种表征（实测 max|Δ|=0.95）。
           前端顺带产出 cpi，与主前向同口径地送进各分支 EventGate（P1.5 统一主线）。

        ② 聚合（N5）：复用 MultiScaleTemporal.fusion 的输入依赖门控权重
           softmax(fusion(x))，按 t_query 的落段帧取权重，而非算术平均。算术平均下
           fusion 在定位任务上完全拿不到梯度——A1「定位训练要回传到时序模型」只完成了一半。

        不加 out_norm 与残差（主前向有）：那两者作用在序列输出 y = out_norm(x + fused) 上，
        而读出是**查询时刻的插值**，t_query 处没有对应的 x 可做残差；强行取落段帧的 x 会把
        帧栅格重新引回来，正是命题 2 要突破的东西。定位头 grounding_head 自带归一化，
        尺度差异由它吸收。

        显存：逐分支顺序求读出并就地累加，不 stack 出 (n_branch,B,Q,d)。
        **推理（no_grad）下**峰值≈最大的那一个分支：每个分支的提交轨迹在其读出算完后即被释放。
        **训练下**三份提交轨迹本来都要留到反向，故逐分支包了梯度检查点（见下）。
        实测反向留存量（saved_tensors_hooks 计量，d_model=384/三分支/B=1）：
            逐分支 checkpoint 关 → 8.18 MB/帧      开 → 0.030 MB/帧（约 1/272）
        注意 EACSLayer.chunk_size 对这条路几乎无效（8.176 vs 8.170）——它只压缩扫描的
        每步中间量，而这里的大头是 run_with_commits 的**返回值**（(B,L,H,N) 复数轨迹），
        那是它的输出、不是内部激活。定位训练要省显存靠的是这里的逐分支检查点。
        """
        from ..modules.continuous_query import continuous_query, segment_index
        import torch.utils.checkpoint as _ckpt

        x, cpi_frame, _, _ = self._frontend(features, timestamps)   # (B,L,d_model)
        w_frame = torch.softmax(self.temporal.fusion(x), dim=-1)    # (B,L,n_branch)
        # 落段帧的融合权重。恒等定段快路与 readout_from_commits 同口径：定位训练传的
        # t_query 就是 timestamps 本身，此时定段索引恒等于 arange(L)，gather 是整份拷贝，
        # searchsorted 也是白算的一次 kernel（N8 修的就是这条路上的无谓开销）。
        # 判定用对象同一性而非 torch.equal——后者要把结果取回 host，在 GPU 上是强制同步。
        if t_query is timestamps:
            w = w_frame                                             # (B,Q=L,n_branch)
        else:
            idx = segment_index(timestamps, t_query)                # (B,Q)
            w = torch.gather(w_frame, 1,
                             idx.unsqueeze(-1).expand(-1, -1, w_frame.shape[-1]))

        def one_branch(br, xx, ts, tq, cpi, wi, inner_chunk):
            y = continuous_query(br, xx, ts, tq, cpi=cpi, use_chunk=inner_chunk)
            return y * wi.unsqueeze(-1)

        # 训练时逐分支做梯度检查点：每个分支的提交轨迹 (B,L,H,N) 复数是 run_with_commits
        # 的**返回值**，正常情况下三份都要留到反向（实测 5.34 MB/帧，是单分支 1.80 的 3 倍）。
        # 包上 checkpoint 后只存该分支的输入，反向时逐个重算再释放，峰值回到约一个分支。
        # 安全性：这条路上没有 dropout / RNG，运行统计也只在 EACSLayer.forward 里更新一次
        # （_step 走的是只读的 apply_stats），所以重算不会双更新统计——这正是 B2 修过的坑。
        #
        # 同时**关掉分支内部的分块检查点**：两层检查点会叠乘成三遍扫描（外层重算一次、
        # 内层再重算一次），实测定位训练每步 9×L 个 cell 步、下限 3×L。而内层分块在这条
        # 路上几乎不省显存（大头是提交轨迹这个返回值，分块压不到——实测 8.176 vs 8.170
        # MB/帧），是纯付出。不包外层检查点时（推理 / 单分支）保持自动决定。
        n_br = len(self.temporal.branches)
        use_ckpt = torch.is_grad_enabled() and n_br > 1
        acc = None
        for i, br in enumerate(self.temporal.branches):
            if use_ckpt:
                y = _ckpt.checkpoint(one_branch, br, x, timestamps, t_query,
                                     cpi_frame, w[..., i], False, use_reentrant=False)
            else:
                y = one_branch(br, x, timestamps, t_query, cpi_frame, w[..., i], None)
            acc = y if acc is None else acc + y
        return acc

    def ground_scores(self, features: Tensor, timestamps: Tensor, input_ids: Tensor,
                      t_query: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """query 条件的时间相关度 logits (B,Q)。需 cfg.grounding_head=True。"""
        assert self.grounding_head is not None, "未启用 grounding_head（CSTSSMConfig.grounding_head=True）"
        qvec = self.grounding_head.encode_query(self._embed_text(input_ids), attention_mask)
        readouts = self.continuous_readout(features, timestamps, t_query)
        return self.grounding_head(qvec, readouts)

    def grounding_forward(self, batch: dict) -> dict:
        """训练用：在帧时刻打分，对 [gt_start,gt_end] 内/外做 BCE。返回 {loss, grounding_loss}。"""
        from ..modules.grounding_head import span_targets, grounding_loss

        def _to_vec(v):
            if torch.is_tensor(v):
                return v.to(batch["timestamps"].device).float()
            return torch.tensor([float(x) for x in v], device=batch["timestamps"].device)

        ts = batch["timestamps"]                         # (B,L)：查询点=帧时刻（真实 Δt）
        logits = self.ground_scores(batch["features"], ts, batch["input_ids"], ts,
                                    batch.get("attention_mask"))
        tgt = span_targets(ts, _to_vec(batch["gt_start"]), _to_vec(batch["gt_end"]))
        loss = grounding_loss(logits, tgt, batch.get("frame_mask"))
        return {"loss": loss, "grounding_loss": loss.detach()}


def build_model(cfg: CSTSSMConfig | None = None) -> CSTSSMModel:
    return CSTSSMModel(cfg or CSTSSMConfig())


def assert_config_applied(model: CSTSSMModel, cfg: CSTSSMConfig) -> None:
    """校验模型的**实际模块状态**与 cfg 一致；不一致直接 raise。

    存在的理由：所有可选模块（CPIB、diff_kv、EventGate 的门控类型与 CPI 调制、EACS 的离散化
    模式与谱初始化）都是在 `CSTSSMModel.__init__` 里按 cfg 一次性建好的。事后写
    `setattr(model.cfg, "eacs_gate_kind", "always")` 只会改那个 dataclass、**不动任何模块**，
    于是消融各臂拿到的其实是同一个模型，而且不会有任何报错——消融表看起来正常、结论全是噪声。
    任何"先建模型、再改开关"的写法都会被这里挡下。

    只查能从模块反查的字段；feat_dim / d_model / llm 这类由底座决定的维度不在此列。
    """
    bad: list[str] = []

    def chk(name, want, got):
        if want != got:
            bad.append(f"{name}: cfg={want!r} 但模型实际={got!r}")

    branches = list(model.temporal.branches)
    chk("eacs_gate_kind", [cfg.eacs_gate_kind] * len(branches),
        [getattr(b.gate, "gate_kind", None) for b in branches])
    chk("eacs_disc_mode", [cfg.eacs_disc_mode] * len(branches),
        [b.disc_mode for b in branches])
    chk("eacs_use_spectral_init", [cfg.eacs_use_spectral_init] * len(branches),
        [b.use_spectral_init for b in branches])
    chk("eacs_chunk", [cfg.eacs_chunk] * len(branches), [b.chunk_size for b in branches])
    # CPI 调制只在 cpib_distill 开启时才接进门控（见 __init__ 的 gate_kw）
    want_mod = cfg.cpi_modulation if cfg.cpib_distill else 0.0
    chk("cpi_modulation", [want_mod] * len(branches),
        [getattr(b.gate, "cpi_modulation", 0.0) for b in branches])
    chk("cpib_distill", cfg.cpib_distill, model.cpib is not None)
    chk("diff_kv", cfg.diff_kv, model.diff_kv is not None)
    chk("grounding_head", cfg.grounding_head, model.grounding_head is not None)
    if model.diff_kv is not None:
        chk("diff_kv_interval", cfg.diff_kv_interval, model.diff_kv.cfg.base_interval)
        chk("diff_kv_rank", cfg.diff_kv_rank, model.diff_kv.cfg.rank)
        chk("diff_kv_cpi_sparse", cfg.diff_kv_cpi_sparse, model.cpi_sparse_attn is not None)

    if bad:
        raise RuntimeError(
            "模型的实际模块状态与 CSTSSMConfig 不一致——多半是在**构造之后**才改 cfg "
            "（那样不会重建任何模块）。请把开关放进构造前的 cfg。不一致项：\n  "
            + "\n  ".join(bad))


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
