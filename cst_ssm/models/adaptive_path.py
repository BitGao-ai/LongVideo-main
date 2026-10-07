"""Single-pass spatial CPIB -> historical SSM -> event token submission."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..modules.streaming import initial_states, temporal_step, sequence_output


@dataclass
class VideoStreamState:
    branches: list
    seen: Tensor
    last_observed: Tensor

    def detach(self):
        return VideoStreamState([s.detach() for s in self.branches],
                                self.seen.detach(), self.last_observed.detach())


def encode_adaptive(model, batch, state=None, mask_inputs=False, collect_loss=True,
                    record_history=False, rho=0.3):
    cfg = model.cfg
    x = batch['features'] if cfg.input_mode == 'feature' else batch['frames']
    B, L = x.shape[:2]
    if L == 0:
        raise ValueError('a video batch needs at least one frame position')
    valid = batch.get('frame_mask', torch.ones(B, L, dtype=torch.bool, device=x.device)).bool()
    timestamps = batch['timestamps'].float()
    if timestamps.shape != valid.shape or not torch.isfinite(timestamps[valid]).all():
        raise ValueError('valid timestamps must be finite with shape (B,L)')
    if state is None:
        state = VideoStreamState(initial_states(model.temporal, B, x.device),
                                 torch.zeros(B, dtype=torch.long, device=x.device),
                                 timestamps.new_full((B,), -torch.inf))
    ys, gs, rs, ns, ws, spatial, contexts, refresh, masked = [], [], [], [], [], [], [], [], []
    base_preds, cf_preds, cf_scores = [], [], []
    history = [[] for _ in model.temporal.branches] if record_history else None
    for t in range(L):
        v = valid[:, t]
        safe_previous = state.last_observed.float().nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
        ts = torch.where(v, timestamps[:, t].float(), safe_previous)
        if not torch.isfinite(ts).all():
            raise ValueError('timestamps must be representable in float32 SSM arithmetic')
        if bool((v & (ts < state.last_observed)).any()):
            raise ValueError('valid timestamps must be nondecreasing across chunks')
        context = model.cpib_context(state.branches[0].h[:, None])[:, 0]
        contexts.append(context)
        frame = torch.where(v.reshape(B, *([1] * (x.ndim - 2))), x[:, t], 0)
        tokens = model.cpib.forward_frame(frame, context, v, rho=rho,
                                           collect_loss=collect_loss)
        spatial.append(tokens)
        use_mask = ((torch.rand(B, device=x.device) < cfg.mask_ratio) & v
                    if mask_inputs and model.training else torch.zeros_like(v))
        masked.append(use_mask)
        feat = torch.where(use_mask[:, None], model.mask_token.to(tokens.frame), tokens.frame)
        if collect_loss and model.training and cfg.cpib_counterfactual:
            # Paired input interventions use the same historical state and posterior
            # means. There is no extra stochastic noise confounding the loss delta.
            with torch.no_grad():
                normal = model.cpib.forward_frame(frame, context, v, deterministic=True)
                P = tokens.scores.shape[1]
                index = torch.randint(P, (B, 1), device=x.device)
                ablate = torch.zeros(B, P, dtype=torch.bool, device=x.device).scatter(1, index, True)
                counter = model.cpib.forward_frame(frame, context, v, ablate_mask=ablate,
                                                   deterministic=True)
                by = temporal_step(model.temporal, normal.frame, ts, state.branches, v, normal.cpi)[0]
                cy = temporal_step(model.temporal, counter.frame, ts, state.branches, v, counter.cpi)[0]
                base_preds.append(model.pred_head(by))
                cf_preds.append(model.pred_head(cy))
            cf_scores.append(tokens.scores.gather(1, index).squeeze(1))
        y, g, r, n, w, branches = temporal_step(
            model.temporal, feat, ts, state.branches, v, tokens.cpi)
        period = cfg.cpib_refresh_frames
        refresh.append(v & ((state.seen == 0) | ((state.seen % max(period, 1) == 0) & (period > 0))))
        state = VideoStreamState(branches, state.seen + v,
                                 torch.where(v, ts, state.last_observed))
        if history is not None:
            for i, (s, gate) in enumerate(zip(branches, g.unbind(1))):
                # Store only steps with at least one commit; per-row validity is
                # retained. This index is O(events), never claimed to be O(1).
                fire = v & (gate.detach() > 0.5)
                if bool(fire.any()):
                    history[i].append((s, fire, w[:, i], ts))
        ys.append(y); gs.append(g); rs.append(r); ns.append(n); ws.append(w)
    ms = sequence_output(ys, gs, rs, ns, ws, valid)
    feat = torch.stack([s.frame for s in spatial], 1)
    target = torch.stack([s.raw_target for s in spatial], 1)
    if not collect_loss:
        return feat, ms, dict(spatial=spatial, state=state, target=target,
                              refresh=torch.stack(refresh, 1), history=history, valid=valid)
    mask = torch.stack(masked, 1)
    losses = {}
    denom = valid.sum().clamp_min(1)
    losses['mask_kl'] = (torch.stack([s.rate_kl for s in spatial], 1) * valid).sum() / denom
    # Sum every transmitted latent dimension and metadata bit before averaging
    # frames. This is a rate, not a dimension-averaged surrogate for I(Z;X).
    rate = torch.stack([s.bottleneck_kl + s.selection_bits for s in spatial], 1)
    losses['representation_rate'] = (rate * valid).sum() / denom
    horizon = cfg.pretrain_horizon
    if horizon < 1:
        raise ValueError('pretrain_horizon must be positive')
    pred = model.pred_head(ms.y)
    pairs = []
    for b in range(B):
        index = valid[b].nonzero(as_tuple=True)[0]
        if index.numel() > horizon:
            pairs.extend((b, int(i), int(j)) for i, j in zip(index[:-horizon], index[horizon:]))
    losses['prediction'] = pred.sum() * 0
    losses['contrastive_surrogate'] = feat.sum() * 0
    losses['counterfactual'] = feat.sum() * 0
    if pairs:
        indices = torch.tensor(pairs, device=x.device)
        bb, src, dst = indices.unbind(1)
        truth = target[bb, dst]
        losses['prediction'] = F.mse_loss(pred[bb, src], truth)
        if cfg.cpib_nce_max_pairs < 2:
            raise ValueError('cpib_nce_max_pairs must be at least two')
        if len(pairs) > 1:
            # A bounded, deterministically spaced set avoids an O((B*L)^2)
            # contrastive matrix. This remains a surrogate, not a CMI bound.
            take = torch.linspace(0, len(pairs)-1, min(len(pairs), cfg.cpib_nce_max_pairs),
                                  device=x.device).long()
            ctx = torch.stack(contexts, 1)
            logits = model.cpib_critic(feat[bb[take], src[take]], truth[take],
                                       ctx[bb[take], src[take]]) / 0.1
            losses['contrastive_surrogate'] = F.cross_entropy(
                logits, torch.arange(take.numel(), device=x.device))
        if cf_preds:
            bp = torch.stack(base_preds, 1)[bb, src]
            cp = torch.stack(cf_preds, 1)[bb, src]
            base_error = (bp - truth).square().mean(-1)
            cf_error = (cp - truth).square().mean(-1)
            # Predictive damage, not a real-world causal effect. Padding is not
            # an observation: horizon counts subsequent *valid* observations.
            damage = ((cf_error-base_error) / truth.square().mean(-1).clamp_min(1e-4)).clamp(0, 1)
            score = torch.stack(cf_scores, 1)[bb, src]
            losses['counterfactual'] = (score - damage.detach()).square().mean()
    recon = masked_mse(model.recon_head(ms.y), target, mask)
    return feat, ms, dict(spatial=spatial, state=state, cpib_losses=losses,
                          target=target, recon_loss=recon, refresh=torch.stack(refresh, 1),
                          history=history, valid=valid)


def masked_mse(pred, target, valid):
    weight = valid.to(pred.dtype).unsqueeze(-1)
    return ((pred - target).square() * weight).sum() / (weight.sum() * pred.shape[-1]).clamp_min(1)


def pack_visual(model, ms, aux):
    """Keep spatial detail tokens, conditioned on time, instead of repooling them."""
    spatial, valid = aux['spatial'], aux['valid']
    gates = ms.gates[:, list(model.cfg.innov_branches)]
    union = 1 - (1 - gates).prod(1)
    if model.cfg.visual_token_mode == 'commit':
        event = (union.detach() > 0.5) & valid
        keep = event | aux['refresh']
    elif model.cfg.visual_token_mode == 'all':
        event = keep = valid
    else:
        raise ValueError('visual_token_mode must be all or commit')
    rows, count_rows = [], []
    for b in range(valid.shape[0]):
        pieces, counts = [], []
        for t, s in enumerate(spatial):
            n = int(s.mask[b].sum()) if bool(keep[b, t]) else 0
            counts.append(n)
            if n:
                # All content originates from bottleneck payloads; temporal state
                # is a deterministic function of earlier payloads/count signals.
                payload = s.tokens[b, s.mask[b]] + ms.y[b, t]
                payload = payload * (1 + (union[b, t] - union[b, t].detach()))
                pieces.append(model.projector(payload))
        rows.append(torch.cat(pieces) if pieces else ms.y.new_zeros((0, model.cfg.llm.dim)))
        count_rows.append(counts)
    max_tokens = max(1, max(row.shape[0] for row in rows))
    lengths = torch.tensor([row.shape[0] for row in rows], device=ms.y.device)
    tokens = torch.stack([F.pad(row, (0, 0, 0, max_tokens - row.shape[0])) for row in rows])
    mask = torch.arange(max_tokens, device=ms.y.device)[None] < lengths[:, None]
    counts = torch.tensor(count_rows, dtype=torch.long, device=ms.y.device)
    return tokens, mask, keep, counts, ((keep & ~event).sum())


def expand_video_placeholders(input_ids, attention_mask, labels, video_token_id, counts,
                              frame_mask, pad_id=0):
    """Replace one placeholder per input frame with that frame's actual token count."""
    rows, labs = [], []
    for b in range(input_ids.shape[0]):
        active = (torch.ones_like(input_ids[b], dtype=torch.bool) if attention_mask is None
                  else attention_mask[b].bool())
        ids = input_ids[b, active]
        old_labels = None if labels is None else labels[b, active]
        per_frame = counts[b, frame_mask[b].bool()]
        slots = (ids == video_token_id).nonzero(as_tuple=True)[0]
        if slots.numel() != per_frame.numel():
            raise ValueError(f'sample {b}: {slots.numel()} placeholders vs {per_frame.numel()} valid frames')
        repeat = torch.ones_like(ids)
        repeat[slots] = per_frame
        rows.append(ids.repeat_interleave(repeat))
        if old_labels is not None:
            old_labels = old_labels.clone()
            old_labels[slots] = -100
            labs.append(old_labels.repeat_interleave(repeat))
    size = max(row.numel() for row in rows)
    ids = torch.stack([F.pad(row, (0, size - row.numel()), value=pad_id) for row in rows])
    mask = torch.stack([torch.arange(size, device=ids.device) < row.numel() for row in rows]).long()
    labels = (torch.stack([F.pad(row, (0, size - row.numel()), value=-100) for row in labs])
              if labs else None)
    return ids, mask, labels
