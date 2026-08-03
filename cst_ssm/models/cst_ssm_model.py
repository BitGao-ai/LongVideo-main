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
    eacs_chunk: int = 0                # >0 启用分块梯度检查点
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
            self.diff_kv = DifferentialKVCache(
                DiffKVConfig(rank=cfg.diff_kv_rank, base_interval=cfg.diff_kv_interval,
                             error_threshold=cfg.diff_kv_threshold,
                             cpi_sparse=cfg.diff_kv_cpi_sparse,
                             cpi_keep_ratio=cfg.diff_kv_keep_ratio),
                d_model=d)
            # CPI 稀疏注意力（因果）：对视觉状态做稀疏压缩后再喂 LLM
            self.cpi_sparse_attn = CPISparseAttention(
                d=d, n_heads=8, keep_ratio=cfg.diff_kv_keep_ratio, causal=True)

    # ---------- 段 1+2：视觉→时序状态 ----------
    def encode_temporal(self, batch: dict):
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        cpi_frame, cpi_tokens, tokens = None, None, None
        ssm_context = None
        if self.cpib is not None and hasattr(self.vision, "get_tokens"):
            # CPIB-Distill 路径：token 蒸馏 → 帧级特征 + CPI 信号
            tokens = self.vision.get_tokens(vis_in)       # (B,L,P,d)
            # SSM 上下文模式：先跑一次 EACS 获取提交状态，再反馈给 CGU
            if (self.cpib.context_mode == "ssm" and self.cpib.ssm_ctx is not None):
                # 第一遍：用 EMA 上下文获取初始 CPI，跑 EACS 得状态
                feat_ema, cpi_ema, _ = self.cpib(tokens, ssm_context=None)
                ms_probe = self.temporal(feat_ema, batch["timestamps"], batch.get("frame_mask"),
                                         cpi=cpi_ema)
                # 从第一分支提取提交状态 h_pi (B,L,H,N)
                with torch.no_grad():
                    _, commits = self.temporal.branches[0].run_with_commits(
                        self.temporal.branches[0].norm(feat_ema),
                        batch["timestamps"])
                ssm_context = self.cpib.project_ssm_states(commits["h"])  # (B,L,d)
            feat, cpi_frame, cpi_tokens = self.cpib(tokens, ssm_context=ssm_context)
        else:
            feat = self.vision(vis_in)                    # (B,L,d_model)
        ms = self.temporal(feat, batch["timestamps"], batch.get("frame_mask"),
                           cpi=cpi_frame)  # MultiScaleOutput，CPI 贯通 EventGate
        return feat, ms, cpi_frame, cpi_tokens, tokens

    # ---------- 端到端前向（stage-2 / 推理）----------
    def forward(self, batch: dict) -> dict:
        feat, ms, cpi_frame, cpi_tokens, tokens = self.encode_temporal(batch)
        visual_states = self.projector(ms.y)           # (B,L,d_llm)

        # 创新点3：CPI 稀疏注意力 + 差分 KV 压缩（训练和推理均启用）
        if self.cpi_sparse_attn is not None and cpi_frame is not None:
            # CPI 稀疏注意力：高 CPI 帧走因果全注意力，低 CPI 用摘要替代
            visual_states = self.cpi_sparse_attn(visual_states, cpi=cpi_frame)
        if self.diff_kv is not None:
            # 差分 KV 缓存：训练时计算重构损失，推理时压缩 KV
            if self.training:
                # 训练：用 visual_states 作为 K/V 代理，学习低秩重构
                diff_loss = self.diff_kv.reconstruction_loss(
                    visual_states.mean(dim=1),  # (B, d) 帧级 K 代理
                    visual_states.mean(dim=1),  # (B, d) 帧级 V 代理
                    feat.mean(dim=1) if feat.dim() == 3 else feat)  # (B, d)
            else:
                # 推理：逐帧更新差分缓存（显存 O(n)，与时长无关）
                self.diff_kv.reset()
                for t in range(visual_states.shape[1]):
                    self.diff_kv.update(
                        visual_states[:, t], visual_states[:, t],
                        feat[:, t] if feat.dim() == 3 else feat, t)

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
        )
        # 创新点3 辅助损失（训练时）
        if self.diff_kv is not None and self.training:
            out["diff_kv_loss"] = diff_loss
        # CPI 信号输出（供创新点1 损失 + P1.5 EventGate 复用）
        if cpi_frame is not None:
            out["cpi_frame"] = cpi_frame
            out["cpi_tokens"] = cpi_tokens
            out["cpib_tokens_raw"] = tokens       # 原始 token（反事实损失用）
        # 辅助未来预测损失（stage-2 总损失的 L_pred 项，设计 §4.4）
        s = self.cfg.pretrain_horizon
        if feat.shape[1] > s:
            pred = self.pred_head(ms.y)
            out["pred_loss"] = F.mse_loss(pred[:, :-s], feat[:, s:].detach())
        return out

    # ---------- stage-1 自监督：未来帧预测 + 掩码帧重构 ----------
    def pretrain_forward(self, batch: dict) -> dict:
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        # 目标特征（重构/预测的 ground-truth）：始终用完整视觉特征
        feat_target = self.vision(vis_in)               # (B,L,d)
        B, L, d = feat_target.shape
        # 输入特征：若启用 CPIB，走 token 蒸馏路径
        cpi_frame, cpi_tokens, cpib_tokens_raw = None, None, None
        if self.cpib is not None and hasattr(self.vision, "get_tokens"):
            tokens = self.vision.get_tokens(vis_in)
            feat_in, cpi_frame, cpi_tokens = self.cpib(tokens)
            cpib_tokens_raw = tokens
        else:
            feat_in = feat_target
        # 掩码帧：随机把一部分帧输入替换为 mask_token，让模型从时序上下文重构
        # 只在有效帧（frame_mask=True）上掩码，避免 padding 帧成为重构目标
        valid = batch.get("frame_mask", torch.ones(B, L, dtype=torch.bool, device=feat_in.device))
        if self.training and self.cfg.mask_ratio > 0 and L > 1:
            mask = (torch.rand(B, L, device=feat_in.device) < self.cfg.mask_ratio) & valid
            feat_in = torch.where(mask.unsqueeze(-1), self.mask_token.to(feat_in.dtype), feat_in)
        else:
            mask = torch.zeros(B, L, dtype=torch.bool, device=feat_in.device)
        ms = self.temporal(feat_in, batch["timestamps"], batch.get("frame_mask"))

        s = self.cfg.pretrain_horizon
        pred = self.pred_head(ms.y)                     # 未来预测
        pred_loss = (F.mse_loss(pred[:, :-s], feat_target[:, s:].detach())
                     if L > s else pred.sum() * 0.0)
        recon = self.recon_head(ms.y)                  # 掩码帧重构
        recon_loss = (F.mse_loss(recon[mask], feat_target.detach()[mask])
                      if mask.any() else recon.sum() * 0.0)
        out = {
            "loss": pred_loss + recon_loss,
            "pred_loss": pred_loss,
            "recon_loss": recon_loss,
            "update_rate": ms.update_rate,
            "spectral_reg": self.temporal.spectral_reg(),
        }
        if cpi_frame is not None:
            out["cpi_frame"] = cpi_frame
            out["cpi_tokens"] = cpi_tokens
            out["cpib_tokens_raw"] = cpib_tokens_raw
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
        """在任意时刻 t_query 求连续读出，各尺度分支均值融合 → (B,Q,d_model)。"""
        from ..modules.continuous_query import ContinuousQuery
        x = self.vision(features)                        # (B,L,d_model)
        outs = [ContinuousQuery(br).query(x, timestamps, t_query) for br in self.temporal.branches]
        return torch.stack(outs, 0).mean(0)              # (B,Q,d_model)

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


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
