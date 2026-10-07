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
    grounding_mode: str = "query"  # query | offset | direct (discrete-readout baselines)
    cpib_distill: bool = False
    cpib_hidden: int = 256
    cpib_ema_alpha: float = 0.9
    cpib_context_mode: str = "ema"
    # Legacy checkpoint defaults stay unchanged; new recipes explicitly select adaptive.
    cpib_mode: str = "legacy"       # legacy | adaptive
    cpib_min_tokens: int = 1
    cpib_max_tokens: int = 16
    cpib_threshold: float = 0.5
    cpib_refresh_frames: int = 64   # bounded periodic safeguard for static details; 0 disables
    cpib_counterfactual: bool = True
    cpib_nce_max_pairs: int = 256  # bound contrastive matrix size independently of video length
    cpi_modulation: float = 0.5
    diff_kv: bool = False
    diff_kv_rank: int = 8
    diff_kv_interval: int = 8
    diff_kv_threshold: float = 0.1
    diff_kv_cpi_sparse: bool = True
    diff_kv_keep_ratio: float = 0.3
    # Observer branches: their innovation drives the self-supervised predictive-coding
    # loss and (in "commit" mode) which frames become LLM visual tokens. Long-memory
    # branches integrate rather than track the observation, so they are excluded.
    innov_branches: tuple[int, ...] = (0,)
    # "all": one visual token per frame; "commit": only frames where an observer
    # branch committed an update, so LLM tokens scale with events, not duration.
    visual_token_mode: str = "all"


def _masked_mse(pred: Tensor, target: Tensor, mask: Tensor | None) -> Tensor:
    """MSE over valid (B,L) positions; plain MSE when mask is None."""
    if mask is None:
        return F.mse_loss(pred, target)
    m = mask.to(pred.dtype).unsqueeze(-1)
    denom = (m.sum() * pred.shape[-1]).clamp_min(1.0)
    return ((pred - target) ** 2 * m).sum() / denom


def innovation_loss(innov: Tensor, frame_mask: Tensor | None,
                    branches: tuple[int, ...]) -> Tensor:
    """Mean squared innovation over observer branches and valid frames after the first.

    innov (B,S,L). Frame 0 is excluded: its prediction starts from the zero state.
    """
    n = innov[:, list(branches)]
    valid = (torch.ones_like(n[:, 0], dtype=torch.bool) if frame_mask is None
             else frame_mask.bool().clone())
    valid = valid & (valid.long().cumsum(1) > 1)
    w = valid.unsqueeze(1).to(n.dtype).expand_as(n)
    return (n.pow(2) * w).sum() / w.sum().clamp_min(1.0)


def select_committed(visual_states: Tensor, gates: Tensor, frame_mask: Tensor | None,
                     branches: tuple[int, ...]):
    """Keep only frames where an observer branch committed an update.

    visual_states (B,L,d), gates (B,S,L). The first valid frame is always kept so
    every sample has at least one token. Kept states are multiplied by a
    straight-through factor (forward 1, backward d gate) so the task loss reaches
    the gate. Returns (states (B,K,d), mask (B,K), keep (B,L)).
    """
    B, L, d = visual_states.shape
    g = gates[:, list(branches)]
    g_any = 1 - torch.prod(1 - g, dim=1)
    valid = (torch.ones(B, L, dtype=torch.bool, device=visual_states.device)
             if frame_mask is None else frame_mask.bool())
    keep = (g_any.detach() > 0.5) & valid
    rows = torch.arange(B, device=keep.device)
    first = torch.argmax(valid.to(torch.int8), dim=1)
    keep[rows, first] = keep[rows, first] | valid[rows, first]
    counts = keep.sum(dim=1)
    K = max(1, int(counts.max()))
    pos = torch.arange(L, device=keep.device).expand(B, L)
    idx = torch.where(keep, pos, pos + L).argsort(dim=1)[:, :K]
    st = (1 + (g_any - g_any.detach())).to(visual_states.dtype)
    sel = (visual_states * st.unsqueeze(-1)).gather(1, idx.unsqueeze(-1).expand(B, K, d))
    vmask = torch.arange(K, device=keep.device).unsqueeze(0) < counts.unsqueeze(1)
    return sel * vmask.unsqueeze(-1).to(sel.dtype), vmask, keep


def drop_video_placeholders(input_ids: Tensor, attention_mask: Tensor | None,
                            labels: Tensor | None, video_token_id: int,
                            keep: Tensor, frame_mask: Tensor | None, pad_id: int = 0):
    """Remove placeholder tokens of dropped frames and right-pad the compacted rows.

    Placeholder j of a sample corresponds to its j-th valid frame (the order the
    placeholder LLM scatters visual states in).
    """
    B, T = input_ids.shape
    valid = (torch.ones_like(keep) if frame_mask is None else frame_mask.bool())
    tok_keep = torch.ones(B, T, dtype=torch.bool, device=input_ids.device)
    for b in range(B):
        ph = (input_ids[b] == video_token_id).nonzero(as_tuple=True)[0]
        kv = keep[b][valid[b]]
        if ph.numel() != kv.numel():
            raise ValueError(f"sample {b}: {ph.numel()} video placeholders vs "
                             f"{kv.numel()} valid frames")
        tok_keep[b, ph[~kv]] = False
    if attention_mask is not None:
        tok_keep_att = tok_keep & attention_mask.bool()
    else:
        tok_keep_att = tok_keep
    n = tok_keep_att.sum(dim=1)
    T2 = int(n.max())
    ids = input_ids.new_full((B, T2), pad_id)
    att = torch.zeros(B, T2, dtype=(attention_mask.dtype if attention_mask is not None
                                    else torch.long), device=input_ids.device)
    lab = None if labels is None else labels.new_full((B, T2), -100)
    for b in range(B):
        k = int(n[b])
        ids[b, :k] = input_ids[b][tok_keep_att[b]]
        att[b, :k] = 1
        if lab is not None:
            lab[b, :k] = labels[b][tok_keep_att[b]]
    return ids, att, lab


class CSTSSMModel(nn.Module):
    def __init__(self, cfg: CSTSSMConfig,
                 vision: nn.Module | None = None,
                 llm: nn.Module | None = None):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        if cfg.cpib_distill and cfg.cpib_mode == "adaptive":
            self.vision = nn.Identity()  # the adaptive encoder below owns this path
        elif vision is not None:
            self.vision = vision
        elif cfg.input_mode == "pixel":
            self.vision = WindowedSpatialEncoder(
                dim=d, patch=cfg.patch, window=cfg.window,
                depth=cfg.vis_depth, n_heads=cfg.vis_heads)
        else:
            self.vision = FeatureAdapter(d_in=cfg.feat_dim, d_out=d)

        self.cpib: CPIBDistill | None = None
        self.cpib_critic = None
        if cfg.cpib_mode not in ("legacy", "adaptive"):
            raise ValueError("cpib_mode must be legacy or adaptive")
        if cfg.cpib_distill and cfg.cpib_mode == "adaptive" and cfg.diff_kv:
            raise ValueError("DiffKV is a standalone feature codec prototype, not an adaptive Qwen KV cache")
        if cfg.cpib_distill and cfg.cpib_mode == "legacy":
            ssm_dim = 0
            if cfg.cpib_context_mode == "ssm":
                first_branch = cfg.branches[0]
                ssm_dim = d * first_branch.n_state * 2
            self.cpib = CPIBDistill(d=d, hidden=cfg.cpib_hidden, ema_alpha=cfg.cpib_ema_alpha,
                                    context_mode=cfg.cpib_context_mode, ssm_state_dim=ssm_dim)
            from ..train.contrastive import ConditionalCritic
            # History-conditioned contrastive surrogate; cross-history negatives
            # do not establish a conditional mutual-information lower bound.
            self.cpib_critic = ConditionalCritic(d)

        gate_kw = dict(init_eps=cfg.gate_init_eps, gate_kind=cfg.eacs_gate_kind)
        if cfg.cpib_distill and cfg.cpi_modulation > 0:
            gate_kw["cpi_modulation"] = cfg.cpi_modulation
        self.temporal = MultiScaleEACS(
            d, branches=cfg.branches,
            gate_kwargs=gate_kw,
            chunk_size=cfg.eacs_chunk, fused_train=cfg.eacs_fused_train,
            robust_guard=cfg.eacs_robust_guard,
            disc_mode=cfg.eacs_disc_mode, use_spectral_init=cfg.eacs_use_spectral_init)

        if cfg.cpib_distill and cfg.cpib_mode == "adaptive":
            if cfg.cpib_context_mode != "ssm":
                raise ValueError("adaptive CPIB requires strictly historical SSM context")
            if vision is not None:
                raise ValueError("adaptive CPIB owns its visual encoder; external vision needs an explicit adapter")
            from ..modules.adaptive_tokens import AdaptiveTokenEncoder
            from ..modules.cgu import SSMStateContext
            from ..train.contrastive import ConditionalCritic
            self.cpib = AdaptiveTokenEncoder(
                d=d, hidden=cfg.cpib_hidden, min_tokens=cfg.cpib_min_tokens,
                max_tokens=cfg.cpib_max_tokens, threshold=cfg.cpib_threshold,
                vision_depth=cfg.vis_depth, heads=cfg.vis_heads, window=cfg.window,
                patch=cfg.patch, input_mode=cfg.input_mode, feat_dim=cfg.feat_dim)
            self.cpib_context = SSMStateContext(d * cfg.branches[0].n_state * 2, d)
            self.cpib_critic = ConditionalCritic(d)
            # The adaptive encoder replaces, rather than bypasses, the old frontend.
            self.vision = nn.Identity()

        self.projector = CrossModalProjector(d, cfg.llm.dim)

        self.llm = llm if llm is not None else VisualConditionedLM(cfg.llm)
        self._llm_need_logits: bool | None = None

        self.pred_head = nn.Linear(d, d)
        self.recon_head = nn.Linear(d, d)
        self.mask_token = nn.Parameter(torch.zeros(d))

        self.grounding_head = None
        self.boundary_head = None
        if cfg.grounding_mode not in ("query", "offset", "direct"):
            raise ValueError("grounding_mode must be query, offset or direct")
        if cfg.grounding_head:
            from ..modules.grounding_head import GroundingHead
            self.grounding_head = GroundingHead(d_model=d, d_txt=cfg.llm.dim)
            if cfg.grounding_mode != "query":
                from ..modules.boundary_baselines import BoundaryRegressionHead
                self.boundary_head = BoundaryRegressionHead(d, cfg.grounding_mode)

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
        if self.cfg.cpib_distill and self.cfg.cpib_mode == "adaptive":
            from .adaptive_path import encode_adaptive
            key = "features" if self.cfg.input_mode == "feature" else "frames"
            feat, _, aux = encode_adaptive(self, {key: vis_in, "timestamps": timestamps},
                                           collect_loss=False)
            return feat, torch.stack([s.cpi for s in aux["spatial"]], 1), None, None
        cpi_frame, cpi_tokens, tokens = None, None, None
        ssm_context = None
        if self.cpib is not None and hasattr(self.vision, "get_tokens"):
            tokens = self.vision.get_tokens(vis_in)
            if (self.cpib.context_mode == "ssm" and self.cpib.ssm_ctx is not None):
                with torch.no_grad():
                    feat_init, _, _ = self.cpib(tokens, ssm_context=None)
                    _, commits = self.temporal.branches[0].run_with_commits(
                        feat_init, timestamps)
                    # Exclusive history: the current frame's commit must never
                    # be used to score that same frame.
                    h = commits["h"]
                    history = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), 1)
                ssm_context = self.cpib.project_ssm_states(history)
            feat, cpi_frame, cpi_tokens = self.cpib(tokens, ssm_context=ssm_context)
        else:
            feat = self.vision(vis_in)
        return feat, cpi_frame, cpi_tokens, tokens

    def encode_temporal(self, batch: dict):
        if self.cfg.cpib_distill and self.cfg.cpib_mode == "adaptive":
            from .adaptive_path import encode_adaptive
            feat, ms, aux = encode_adaptive(self, batch, collect_loss=False)
            return feat, ms, torch.stack([s.cpi for s in aux["spatial"]], 1), None, None
        vis_in = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        feat, cpi_frame, cpi_tokens, tokens = self._frontend(vis_in, batch["timestamps"])
        ms = self.temporal(feat, batch["timestamps"], batch.get("frame_mask"),
                           cpi=cpi_frame)
        return feat, ms, cpi_frame, cpi_tokens, tokens

    def forward(self, batch: dict, stage: str = "finetune",
                need_logits: bool = True, cpib_weights=None, step: int = 0) -> dict:
        """Single entry for all stages; DDP requires routing auxiliary losses here."""
        if stage == "pretrain":
            out = self.pretrain_forward(batch, cpib_weights=cpib_weights)
        elif stage == "grounding":
            return self.grounding_forward(batch)
        else:
            out = self.finetune_forward(batch, need_logits=need_logits,
                                        cpib_weights=cpib_weights)
        if cpib_weights is not None and self.cfg.cpib_mode == "legacy":
            from ..train.losses import _compute_cpib, LossWeights
            loss, components = _compute_cpib(out, self, LossWeights(cpib=cpib_weights), step)
            out["legacy_cpib_loss"] = loss
            out["legacy_cpib_components"] = components
        return out

    def encode_visual(self, batch: dict, collect_loss: bool = True, cpib_weights=None):
        """Visual states for the LLM plus auxiliaries for losses; runs once per video."""
        if self.cfg.cpib_distill and self.cfg.cpib_mode == "adaptive":
            from .adaptive_path import encode_adaptive, pack_visual
            feat, ms, aux = encode_adaptive(self, batch, collect_loss=collect_loss,
                                            rho=getattr(cpib_weights, "rho", 0.3))
            visual, mask, keep, counts, extra = pack_visual(self, ms, aux)
            aux.update(feat=feat, ms=ms, cpi_frame=torch.stack([s.cpi for s in aux["spatial"]], 1),
                       cpi_tokens=None, tokens=None, visual_mask=mask, keep=keep,
                       frame_token_counts=counts, safeguard_frames=extra)
            return visual, aux
        feat, ms, cpi_frame, cpi_tokens, tokens = self.encode_temporal(batch)
        visual_states = self.projector(ms.y)
        if self.cpi_sparse_attn is not None and cpi_frame is not None:
            visual_states = self.cpi_sparse_attn(visual_states, cpi=cpi_frame,
                                                  mask=batch.get("frame_mask"))
        frame_mask = batch.get("frame_mask")
        visual_mask, keep = frame_mask, None
        if self.cfg.visual_token_mode == "commit":
            visual_states, visual_mask, keep = select_committed(
                visual_states, ms.gates, frame_mask, self.cfg.innov_branches)
        elif self.cfg.visual_token_mode != "all":
            raise ValueError(f"unknown visual_token_mode {self.cfg.visual_token_mode!r}")
        return visual_states, dict(feat=feat, ms=ms, cpi_frame=cpi_frame,
                                   cpi_tokens=cpi_tokens, tokens=tokens,
                                   visual_mask=visual_mask, keep=keep)

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

    def prepare_language_inputs(self, batch: dict, aux: dict,
                                max_new_tokens: int = 0) -> dict:
        """Align text, masks and labels with the exact visual tokens being injected."""
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch.get("labels")
        keep = aux["keep"]
        vid = getattr(self.llm, "video_token_id", None)
        if keep is not None and vid is not None:
            if "frame_token_counts" in aux:
                from .adaptive_path import expand_video_placeholders
                input_ids, attention_mask, labels = expand_video_placeholders(
                    input_ids, attention_mask, labels, vid, aux["frame_token_counts"], aux["valid"])
            else:
                input_ids, attention_mask, labels = drop_video_placeholders(
                    input_ids, attention_mask, labels, vid, keep, batch.get("frame_mask"))
        if input_ids.shape[1] + max_new_tokens > self.cfg.llm.max_len:
            raise ValueError(f"expanded prompt has {input_ids.shape[1]} tokens plus "
                             f"{max_new_tokens} generation tokens, exceeding "
                             f"llm.max_len={self.cfg.llm.max_len}")
        return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def language_forward(self, batch: dict, visual_states: Tensor, aux: dict,
                         need_logits: bool = True) -> dict:
        inputs = self.prepare_language_inputs(batch, aux)
        out = self.llm(**inputs, visual_states=visual_states, visual_mask=aux["visual_mask"],
                       **({"need_logits": need_logits} if self._llm_takes_need_logits() else {}))
        out["effective_labels"] = inputs["labels"]
        return out

    def finetune_forward(self, batch: dict, need_logits: bool = True,
                         cpib_weights=None) -> dict:
        visual_states, aux = self.encode_visual(batch, cpib_weights=cpib_weights)
        feat, ms = aux["feat"], aux["ms"]
        cpi_frame, cpi_tokens, tokens = aux["cpi_frame"], aux["cpi_tokens"], aux["tokens"]
        diff_loss = None
        if self.diff_kv is not None and self.training:
            diff_loss = self.diff_kv.sequence_reconstruction_loss(
                visual_states, visual_states, mask=aux["visual_mask"])
        keep = aux["keep"]
        out = self.language_forward(batch, visual_states, aux, need_logits)
        out.update(
            update_rate=ms.update_rate,
            per_branch_update_rate=ms.per_branch_update_rate,
            gates=ms.gates,
            spectral_reg=self.temporal.spectral_reg(),
            residual=ms.residual,
            frame_mask=batch.get("frame_mask"),
        )
        if ms.innovation is not None:
            out["innov_loss"] = innovation_loss(ms.innovation, batch.get("frame_mask"),
                                                self.cfg.innov_branches)
        if keep is not None:
            fm = batch.get("frame_mask")
            n_valid = (fm.sum() if fm is not None else torch.tensor(keep.numel()))
            out["visual_token_ratio"] = (aux["visual_mask"].sum().float()
                                         / n_valid.clamp_min(1).float()).detach()
        if diff_loss is not None:
            out["feature_codec_loss"] = diff_loss
        if "cpib_losses" in aux:
            out["cpib_losses"] = aux["cpib_losses"]
            out["pred_loss"] = aux["cpib_losses"]["prediction"]
            out["visual_token_count"] = aux["visual_mask"].sum().detach()
            n_frames = aux["valid"].sum().clamp_min(1)
            out["frame_commit_ratio"] = (aux["keep"].sum() / n_frames).detach()
            original_spatial = sum(s.scores.shape[1] * aux["valid"][:, t].sum()
                                   for t, s in enumerate(aux["spatial"]))
            out["visual_token_ratio"] = (out["visual_token_count"]
                                         / original_spatial.clamp_min(1)).detach()
            out["safeguard_frames"] = aux["safeguard_frames"].detach()
            out["frame_token_counts"] = aux["frame_token_counts"]
            return out
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
        """Legacy-named visual-feature codec diagnostic, NOT language-model KV savings."""
        import warnings
        warnings.warn("measure_kv_compression measures projected features, not Qwen KV cache", stacklevel=2)
        assert self.diff_kv is not None, "diff_kv is not enabled (CSTSSMConfig.diff_kv=True)"
        assert self.cfg.visual_token_mode == "all", "measure_kv_compression needs per-frame tokens"
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
        stats["representation"] = "projected_visual_features_not_llm_kv"
        K_rec, _ = self.diff_kv.reconstruct_residual_frames()
        stats["recon_mse"] = (float(F.mse_loss(K_rec, torch.stack(residual_truth)).item())
                              if K_rec is not None else 0.0)
        return stats

    def pretrain_forward(self, batch: dict, cpib_weights=None) -> dict:
        if self.cfg.cpib_distill and self.cfg.cpib_mode == "adaptive":
            from .adaptive_path import encode_adaptive
            _, ms, aux = encode_adaptive(self, batch, mask_inputs=True,
                                         rho=getattr(cpib_weights, "rho", 0.3))
            pred = aux["cpib_losses"]["prediction"]
            return dict(loss=pred + aux["recon_loss"], pred_loss=pred,
                        recon_loss=aux["recon_loss"], cpib_losses=aux["cpib_losses"],
                        update_rate=ms.update_rate, spectral_reg=self.temporal.spectral_reg(),
                        innov_loss=innovation_loss(ms.innovation, aux["valid"], self.cfg.innov_branches),
                        frame_mask=aux["valid"])
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
        if ms.innovation is not None:
            out["innov_loss"] = innovation_loss(ms.innovation, batch.get("frame_mask"),
                                                self.cfg.innov_branches)
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

    @torch.no_grad()
    def generate(self, batch: dict, **decode_kwargs) -> dict:
        """Pack adaptive visual tokens and decode through an injected Qwen wrapper."""
        if self.training:
            raise ValueError("generate requires eval()")
        if not hasattr(self.llm, "greedy_decode"):
            raise ValueError("this language-model adapter does not support cached generation")
        visual, aux = self.encode_visual(batch, collect_loss=False)
        inputs = self.prepare_language_inputs(batch, aux, decode_kwargs.get("max_new_tokens", 32))
        ids, att = inputs["input_ids"], inputs["attention_mask"]
        out = self.llm.greedy_decode(ids, visual, attention_mask=att,
                                     visual_mask=aux["visual_mask"], **decode_kwargs)
        out["visual_token_count"] = aux["visual_mask"].sum(1) if aux["visual_mask"] is not None else ids.new_full((ids.shape[0],), visual.shape[1])
        return out

    def stream_visual(self, batch: dict, state=None):
        """Process one chunk; caller owns recurrent state and consumes emitted tokens.

        This does not retain a language-model KV cache or historical query index.
        Call state.detach() explicitly for truncated-BPTT boundaries.
        """
        if not (self.cfg.cpib_distill and self.cfg.cpib_mode == "adaptive"):
            raise ValueError("stream_visual requires adaptive CPIB")
        from .adaptive_path import encode_adaptive, pack_visual
        _, ms, aux = encode_adaptive(self, batch, state=state, collect_loss=False)
        visual, mask, keep, counts, extra = pack_visual(self, ms, aux)
        return dict(visual_states=visual, visual_mask=mask, keep=keep,
                    frame_token_counts=counts, safeguard_frames=extra,
                    state=aux["state"], temporal=ms)

    def continuous_readout(self, features: Tensor, timestamps: Tensor, t_query: Tensor,
                           frame_mask: Tensor | None = None) -> Tensor:
        """Continuous readouts at t_query fused across branches; returns (B,Q,d_model)."""
        if self.cfg.cpib_distill and self.cfg.cpib_mode == "adaptive":
            from .adaptive_path import encode_adaptive
            from .adaptive_query import query_history
            key = "features" if self.cfg.input_mode == "feature" else "frames"
            batch = {key: features, "timestamps": timestamps}
            if frame_mask is not None:
                batch["frame_mask"] = frame_mask
            _, _, aux = encode_adaptive(self, batch, collect_loss=False, record_history=True)
            return query_history(self, aux["history"], t_query)
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
                      t_query: Tensor, attention_mask: Tensor | None = None,
                      frame_mask: Tensor | None = None) -> Tensor:
        """Query-conditioned relevance logits (B,Q); requires grounding_head."""
        assert self.grounding_head is not None, "grounding_head is not enabled"
        if self.boundary_head is not None:
            raise ValueError("regression grounding uses predict_boundaries(), not a grid score head")
        qvec = self.grounding_head.encode_query(self._embed_text(input_ids), attention_mask)
        readouts = self.continuous_readout(features, timestamps, t_query, frame_mask)
        return self.grounding_head(qvec, readouts)

    def predict_boundaries(self, batch: dict) -> Tensor:
        """Fair baseline: discrete frame readouts with a continuous regression head."""
        if self.boundary_head is None:
            raise ValueError("predict_boundaries requires grounding_mode=offset or direct")
        _, ms, _, _, _ = self.encode_temporal(batch)
        query = self.grounding_head.encode_query(self._embed_text(batch["input_ids"]),
                                                  batch.get("attention_mask"))
        return self.boundary_head(ms.y, query, batch["timestamps"], batch.get("frame_mask"))

    def grounding_forward(self, batch: dict) -> dict:
        """Score frame times against [gt_start, gt_end]; returns {loss, grounding_loss}."""
        from ..modules.grounding_head import span_targets, grounding_loss

        def _to_vec(v):
            if torch.is_tensor(v):
                return v.to(batch["timestamps"].device).float()
            return torch.tensor([float(x) for x in v], device=batch["timestamps"].device)

        ts = batch["timestamps"]
        if self.boundary_head is not None:
            from ..modules.boundary_baselines import boundary_regression_loss
            spans = self.predict_boundaries(batch)
            target = torch.stack((_to_vec(batch["gt_start"]), _to_vec(batch["gt_end"])), -1)
            loss = boundary_regression_loss(spans, target, ts, batch.get("frame_mask"))
            return {"loss": loss, "grounding_loss": loss.detach(), "predicted_spans": spans}
        visual_input = batch["features"] if self.cfg.input_mode == "feature" else batch["frames"]
        logits = self.ground_scores(visual_input, ts, batch["input_ids"], ts,
                                    batch.get("attention_mask"), batch.get("frame_mask"))
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
