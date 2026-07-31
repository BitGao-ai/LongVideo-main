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

        # 段 2：连续时序建模（EACS 多尺度）
        self.temporal = MultiScaleEACS(
            d, branches=cfg.branches,
            gate_kwargs=dict(init_eps=cfg.gate_init_eps, gate_kind=cfg.eacs_gate_kind),
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

    # ---------- 段 1+2：视觉→时序状态 ----------
    def encode_temporal(self, batch: dict):
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        feat = self.vision(vis_in)                     # (B,L,d_model)
        ms = self.temporal(feat, batch["timestamps"], batch.get("frame_mask"))  # MultiScaleOutput
        return feat, ms

    # ---------- 端到端前向（stage-2 / 推理）----------
    def forward(self, batch: dict) -> dict:
        feat, ms = self.encode_temporal(batch)
        visual_states = self.projector(ms.y)           # (B,L,d_llm)
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
        # 辅助未来预测损失（stage-2 总损失的 L_pred 项，设计 §4.4）
        s = self.cfg.pretrain_horizon
        if feat.shape[1] > s:
            pred = self.pred_head(ms.y)
            out["pred_loss"] = F.mse_loss(pred[:, :-s], feat[:, s:].detach())
        return out

    # ---------- stage-1 自监督：未来帧预测 + 掩码帧重构 ----------
    def pretrain_forward(self, batch: dict) -> dict:
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        feat = self.vision(vis_in)                     # (B,L,d) 完整特征=重构/预测目标
        B, L, d = feat.shape
        # 掩码帧：随机把一部分帧输入替换为 mask_token，让模型从时序上下文重构
        if self.training and self.cfg.mask_ratio > 0 and L > 1:
            mask = torch.rand(B, L, device=feat.device) < self.cfg.mask_ratio    # (B,L)
            feat_in = torch.where(mask.unsqueeze(-1), self.mask_token.to(feat.dtype), feat)
        else:
            mask = torch.zeros(B, L, dtype=torch.bool, device=feat.device)
            feat_in = feat
        ms = self.temporal(feat_in, batch["timestamps"], batch.get("frame_mask"))

        s = self.cfg.pretrain_horizon
        pred = self.pred_head(ms.y)                     # 未来预测
        pred_loss = (F.mse_loss(pred[:, :-s], feat[:, s:].detach())
                     if L > s else pred.sum() * 0.0)
        recon = self.recon_head(ms.y)                  # 掩码帧重构
        recon_loss = (F.mse_loss(recon[mask], feat.detach()[mask])
                      if mask.any() else recon.sum() * 0.0)
        return {
            "loss": pred_loss + recon_loss,            # 便捷合计；权重在 losses.pretrain_loss
            "pred_loss": pred_loss,
            "recon_loss": recon_loss,
            "update_rate": ms.update_rate,
            "spectral_reg": self.temporal.spectral_reg(),
        }

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
