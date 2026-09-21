"""Top-level CST-SSM model: spatial encoding, temporal modeling, projection, LLM."""
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
    input_mode: str = "feature"
    patch: int = 16
    window: int = 7
    vis_depth: int = 4
    vis_heads: int = 6
    feat_dim: int = 768
    d_model: int = 384
    branches: tuple[BranchSpec, ...] = DEFAULT_BRANCHES
    gate_init_eps: float = 0.1
    eacs_chunk: int = 32
    eacs_fused_train: bool = False
    eacs_robust_guard: bool = False
    eacs_disc_mode: str = "continuous"
    eacs_gate_kind: str = "event"
    eacs_use_spectral_init: bool = True
    llm: LLMConfig = field(default_factory=LLMConfig)
    pretrain_horizon: int = 1
    mask_ratio: float = 0.15
    grounding_head: bool = False
    cpib_distill: bool = False
    cpib_hidden: int = 256
    cpib_ema_alpha: float = 0.9
    cpib_context_mode: str = "ema"
    cpi_modulation: float = 0.5
    diff_kv: bool = False
    diff_kv_rank: int = 8
    diff_kv_interval: int = 8
    diff_kv_threshold: float = 0.1
    diff_kv_cpi_sparse: bool = True
    diff_kv_keep_ratio: float = 0.3


def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """MSE over valid (B,L) positions; plain MSE when mask is None."""
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.to(pred.dtype).unsqueeze(-1)
    denom = (m.sum() * pred.shape[-1]).clamp_min(1.0)
    return ((pred - target) ** 2 * m).sum() / denom


class CSTSSMModel(nn.Module):
    def __init__(self, cfg: CSTSSMConfig,
                 vision: nn.Module | None = None,
                 llm: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        if vision is not None:
            self.vision = vision
        elif cfg.input_mode == "pixel":
            self.vision = WindowedSpatialEncoder(
                dim=d, patch=cfg.patch, window=cfg.window,
                depth=cfg.vis_depth, n_heads=cfg.vis_heads)
        else:
            self.vision = FeatureAdapter(d_in=cfg.feat_dim, d_out=d)

        self.cpib: CPIBDistill | None = None
        self.cpib_critic = None
        if cfg.cpib_distill:
            ssm_dim = 0
            if cfg.cpib_context_mode == "ssm":
                first_branch = cfg.branches[0]
                ssm_dim = d * first_branch.n_state * 2
            self.cpib = CPIBDistill(d=d, hidden=cfg.cpib_hidden, ema_alpha=cfg.cpib_ema_alpha,
                                    context_mode=cfg.cpib_context_mode, ssm_state_dim=ssm_dim)
            from ..train.contrastive import BilinearCritic
            self.cpib_critic = BilinearCritic(d)

        gate_kw = dict(init_eps=cfg.gate_init_eps, gate_kind=cfg.eacs_gate_kind)
        if cfg.cpib_distill and cfg.cpi_modulation > 0:
            gate_kw["cpi_modulation"] = cfg.cpi_modulation
        self.temporal = MultiScaleEACS(
            d, branches=cfg.branches,
            gate_kwargs=gate_kw,
            chunk_size=cfg.eacs_chunk, fused_train=cfg.eacs_fused_train,
            robust_guard=cfg.eacs_robust_guard,
            disc_mode=cfg.eacs_disc_mode, use_spectral_init=cfg.eacs_use_spectral_init)

        self.projector = CrossModalProjector(d, cfg.llm.dim)

        self.llm = llm if llm is not None else VisualConditionedLM(cfg.llm)
        self._llm_need_logits: bool | None = None

        self.pred_head = nn.Linear(d, d)
        self.recon_head = nn.Linear(d, d)
        self.mask_token = nn.Parameter(torch.zeros(d))

        self.grounding_head = None
        if cfg.grounding_head:
            from ..modules.grounding_head import GroundingHead
            self.grounding_head = GroundingHead(d_model=d, d_txt=cfg.llm.dim)

        self.diff_kv = None
        self.cpi_sparse_attn = None
        if cfg.diff_kv:
            from ..modules.diff_kv import DifferentialKVCache, DiffKVConfig, CPISparseAttention
            self.diff_kv = DifferentialKVCache(
                DiffKVConfig(rank=cfg.diff_kv_rank, base_interval=cfg.diff_kv_interval,
                             error_threshold=cfg.diff_kv_threshold,
                             cpi_sparse=cfg.diff_kv_cpi_sparse,
                             cpi_keep_ratio=cfg.diff_kv_keep_ratio),
                d_model=cfg.llm.dim)
            if cfg.diff_kv_cpi_sparse:
                sparse_dim = cfg.llm.dim
                sparse_heads = next(h for h in (8, 4, 2, 1) if sparse_dim % h == 0)
                self.cpi_sparse_attn = CPISparseAttention(
                    d=sparse_dim, n_heads=sparse_heads,
                    keep_ratio=cfg.diff_kv_keep_ratio, causal=True)

    def _frontend(self, vis_in: Tensor, timestamps: Tensor):
        """Visual input to EACS frame features plus CPI signals.

        All EACS paths (QA, pretraining, localization) share this entry point so
        stage-2 weights transfer to localization on the same representation.
        """
        cpi_frame, cpi_tokens, tokens = None, None, None
        ssm_context = None
        if self.cpib is not None and hasattr(self.vision, "get_tokens"):
            tokens = self.vision.get_tokens(vis_in)
            if (self.cpib.context_mode == "ssm" and self.cpib.ssm_ctx is not None):
                with torch.no_grad():
                    feat_init, _, _ = self.cpib(tokens, ssm_context=None)
                    _, commits = self.temporal.branches[0].run_with_commits(
                        feat_init, timestamps)
                ssm_context = self.cpib.project_ssm_states(commits["h"])
            feat, cpi_frame, cpi_tokens = self.cpib(tokens, ssm_context=ssm_context)
        else:
            feat = self.vision(vis_in)
        return feat, cpi_frame, cpi_tokens, tokens

    def encode_temporal(self, batch: dict):
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        feat, cpi_frame, cpi_tokens, tokens = self._frontend(vis_in, batch["timestamps"])
        ms = self.temporal(feat, batch["timestamps"], batch.get("frame_mask"),
                           cpi=cpi_frame)
        return feat, ms, cpi_frame, cpi_tokens, tokens

    def forward(self, batch: dict, stage: str = "finetune",
                need_logits: bool = True) -> dict:
        """Single entry for all stages; DDP requires routing through this method."""
        if stage == "pretrain":
            return self.pretrain_forward(batch)
        if stage == "grounding":
            return self.grounding_forward(batch)
        return self.finetune_forward(batch, need_logits=need_logits)

    def encode_visual(self, batch: dict):
        """Visual states for the LLM plus auxiliaries for losses; runs once per video."""
        feat, ms, cpi_frame, cpi_tokens, tokens = self.encode_temporal(batch)
        visual_states = self.projector(ms.y)
        if self.cpi_sparse_attn is not None and cpi_frame is not None:
            visual_states = self.cpi_sparse_attn(visual_states, cpi=cpi_frame)
        return visual_states, dict(feat=feat, ms=ms, cpi_frame=cpi_frame,
                                   cpi_tokens=cpi_tokens, tokens=tokens)

    def _llm_takes_need_logits(self) -> bool:
        """Probe once whether the injected LLM accepts the need_logits keyword."""
        if self._llm_need_logits is None:
            import inspect
            try:
                params = inspect.signature(self.llm.forward).parameters
                self._llm_need_logits = (
                    "need_logits" in params
                    or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()))
            except (TypeError, ValueError):
                self._llm_need_logits = False
        return self._llm_need_logits

    def finetune_forward(self, batch: dict, need_logits: bool = True) -> dict:
        visual_states, aux = self.encode_visual(batch)
        feat, ms = aux["feat"], aux["ms"]
        cpi_frame, cpi_tokens, tokens = aux["cpi_frame"], aux["cpi_tokens"], aux["tokens"]
        diff_loss = None
        if self.diff_kv is not None and self.training:
            diff_loss = self.diff_kv.sequence_reconstruction_loss(
                visual_states, visual_states)

        out = self.llm(
            input_ids=batch["input_ids"],
            visual_states=visual_states,
            attention_mask=batch.get("attention_mask"),
            visual_mask=batch.get("frame_mask"),
            labels=batch.get("labels"),
            **({"need_logits": need_logits} if self._llm_takes_need_logits() else {}),
        )
        out.update(
            update_rate=ms.update_rate,
            per_branch_update_rate=ms.per_branch_update_rate,
            gates=ms.gates,
            spectral_reg=self.temporal.spectral_reg(),
            residual=ms.residual,
            frame_mask=batch.get("frame_mask"),
        )
        if diff_loss is not None:
            out["diff_kv_loss"] = diff_loss
        if cpi_frame is not None:
            out["cpi_frame"] = cpi_frame
            out["cpi_tokens"] = cpi_tokens
            out["cpib_tokens_raw"] = tokens
            out["cpib_frame_feat"] = feat
        s = self.cfg.pretrain_horizon
        if feat.shape[1] > s:
            pred = self.pred_head(ms.y)
            fm = batch.get("frame_mask")
            pm = (fm[:, :-s] & fm[:, s:]) if fm is not None else None
            out["pred_loss"] = _masked_mse(pred[:, :-s], feat[:, s:].detach(), pm)
        return out

    @torch.no_grad()
    def measure_kv_compression(self, batch: dict) -> dict:
        """Streaming differential-KV pass; returns compression stats and recon error."""
        assert self.diff_kv is not None, "diff_kv is not enabled (CSTSSMConfig.diff_kv=True)"
        visual_states, aux = self.encode_visual(batch)
        feat = aux["feat"]

        self.diff_kv.reset()
        residual_truth = []
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

    def pretrain_forward(self, batch: dict) -> dict:
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        feat_target, cpi_frame, cpi_tokens, cpib_tokens_raw = self._frontend(
            vis_in, batch["timestamps"])
        cpib_frame_feat = feat_target if cpib_tokens_raw is not None else None
        feat_in = feat_target
        B, L, d = feat_target.shape
        valid = batch.get("frame_mask", torch.ones(B, L, dtype=torch.bool, device=feat_in.device))
        if self.training and self.cfg.mask_ratio > 0 and L > 1:
            mask = (torch.rand(B, L, device=feat_in.device) < self.cfg.mask_ratio) & valid
            feat_in = torch.where(mask.unsqueeze(-1), self.mask_token.to(feat_in.dtype), feat_in)
        else:
            mask = torch.zeros(B, L, dtype=torch.bool, device=feat_in.device)
        ms = self.temporal(feat_in, batch["timestamps"], batch.get("frame_mask"),
                           cpi=cpi_frame)

        s = self.cfg.pretrain_horizon
        pred = self.pred_head(ms.y)
        fm = batch.get("frame_mask")
        pm = (fm[:, :-s] & fm[:, s:]) if (fm is not None and L > s) else None
        pred_loss = (_masked_mse(pred[:, :-s], feat_target[:, s:].detach(), pm)
                     if L > s else pred.sum() * 0.0)
        recon = self.recon_head(ms.y)
        recon_loss = (F.mse_loss(recon[mask], feat_target.detach()[mask])
                      if mask.any() else recon.sum() * 0.0)
        out = {
            "loss": pred_loss + recon_loss,
            "pred_loss": pred_loss,
            "recon_loss": recon_loss,
            "update_rate": ms.update_rate,
            "spectral_reg": self.temporal.spectral_reg(),
            "frame_mask": batch.get("frame_mask"),
        }
        if cpi_frame is not None:
            out["cpi_frame"] = cpi_frame
            out["cpi_tokens"] = cpi_tokens
            out["cpib_tokens_raw"] = cpib_tokens_raw
            out["cpib_frame_feat"] = cpib_frame_feat
        return out

    def _embed_text(self, input_ids: Tensor) -> Tensor:
        """Token embeddings (B,T,d_txt) from the stand-in or an injected HF LLM."""
        llm = self.llm
        if hasattr(llm, "embed"):
            return llm.embed(input_ids)
        if hasattr(llm, "get_input_embeddings"):
            return llm.get_input_embeddings()(input_ids)
        if hasattr(llm, "hf"):
            return llm.hf.get_input_embeddings()(input_ids)
        raise AttributeError("Cannot access LLM text embeddings.")

    def continuous_readout(self, features: Tensor, timestamps: Tensor, t_query: Tensor) -> Tensor:
        """Continuous readouts at t_query fused across branches; returns (B,Q,d_model)."""
        from ..modules.continuous_query import continuous_query, segment_index
        import torch.utils.checkpoint as _ckpt

        x, cpi_frame, _, _ = self._frontend(features, timestamps)
        w_frame = torch.softmax(self.temporal.fusion(x), dim=-1)
        if t_query is timestamps:
            w = w_frame
        else:
            idx = segment_index(timestamps, t_query)
            w = torch.gather(w_frame, 1,
                             idx.unsqueeze(-1).expand(-1, -1, w_frame.shape[-1]))

        def one_branch(br, xx, ts, tq, cpi, wi, inner_chunk):
            y = continuous_query(br, xx, ts, tq, cpi=cpi, use_chunk=inner_chunk)
            return y * wi.unsqueeze(-1)

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
        """Query-conditioned relevance logits (B,Q); requires grounding_head."""
        assert self.grounding_head is not None, "grounding_head is not enabled"
        qvec = self.grounding_head.encode_query(self._embed_text(input_ids), attention_mask)
        readouts = self.continuous_readout(features, timestamps, t_query)
        return self.grounding_head(qvec, readouts)

    def grounding_forward(self, batch: dict) -> dict:
        """Score frame times against [gt_start, gt_end]; returns {loss, grounding_loss}."""
        from ..modules.grounding_head import span_targets, grounding_loss

        def _to_vec(v):
            if torch.is_tensor(v):
                return v.to(batch["timestamps"].device).float()
            return torch.tensor([float(x) for x in v], device=batch["timestamps"].device)

        ts = batch["timestamps"]
        logits = self.ground_scores(batch["features"], ts, batch["input_ids"], ts,
                                    batch.get("attention_mask"))
        tgt = span_targets(ts, _to_vec(batch["gt_start"]), _to_vec(batch["gt_end"]))
        loss = grounding_loss(logits, tgt, batch.get("frame_mask"))
        return {"loss": loss, "grounding_loss": loss.detach()}


def build_model(cfg: CSTSSMConfig | None = None) -> CSTSSMModel:
    return CSTSSMModel(cfg or CSTSSMConfig())


def assert_config_applied(model: CSTSSMModel, cfg: CSTSSMConfig) -> None:
    """Verify live module state matches cfg; raise on post-construction edits."""
    bad: list[str] = []

    def chk(name, want, got):
        if want != got:
            bad.append(f"{name}: cfg={want!r} but model has {got!r}")

    branches = list(model.temporal.branches)
    chk("eacs_gate_kind", [cfg.eacs_gate_kind] * len(branches),
        [getattr(b.gate, "gate_kind", None) for b in branches])
    chk("eacs_disc_mode", [cfg.eacs_disc_mode] * len(branches),
        [b.disc_mode for b in branches])
    chk("eacs_use_spectral_init", [cfg.eacs_use_spectral_init] * len(branches),
        [b.use_spectral_init for b in branches])
    chk("eacs_chunk", [cfg.eacs_chunk] * len(branches), [b.chunk_size for b in branches])
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
            "Live module state does not match CSTSSMConfig; "
            "set switches before construction:\n  " + "\n  ".join(bad))


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
