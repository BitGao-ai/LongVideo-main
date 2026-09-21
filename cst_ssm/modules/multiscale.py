"""Multiscale EACS: short/mid/long branches with input-dependent fusion."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torch import Tensor

from ..ops.spectral_init import BranchSpec, DEFAULT_BRANCHES
from .eacs import EACSLayer, EACSOutput, StepParams, scan_range, _write_chunk
from .event_gate import EventGate


@dataclass
class MultiScaleOutput:
    y: Tensor                        # (B, L, d) fused temporal features
    gates: Tensor                    # (B, n_branch, L)
    update_rate: Tensor              # scalar overall update rate
    per_branch_update_rate: Tensor   # (n_branch,)
    fusion_weights: Tensor           # (B, L, n_branch)
    residual: Tensor                 # (B, n_branch, L)


class MultiScaleEACS(nn.Module):
    def __init__(self, d_model: int,
                 branches: tuple[BranchSpec, ...] | list[BranchSpec] = DEFAULT_BRANCHES,
                 gate_kwargs: dict | None = None,
                 dt_init: float = 1.0, chunk_size: int = 0, fused_train: bool = False,
                 robust_guard: bool = False,
                 disc_mode: str = "continuous", use_spectral_init: bool = True):
        super().__init__()
        gate_kwargs = gate_kwargs or {}
        self.branch_specs = list(branches)
        self.branches = nn.ModuleList([
            EACSLayer(d_model, spec, gate=EventGate(**gate_kwargs),
                      dt_init=dt_init, chunk_size=chunk_size, add_residual=False,
                      fused_train=fused_train, robust_guard=robust_guard,
                      disc_mode=disc_mode, use_spectral_init=use_spectral_init)
            for spec in branches
        ])
        self.n_branch = len(self.branches)
        self.fusion = nn.Linear(d_model, self.n_branch)
        self.out_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "_dt_max", torch.tensor([b.dt_max for b in self.branches]).view(-1, 1),
            persistent=False)

    def merge_status(self) -> tuple[bool, str]:
        """Whether merged branch scanning applies under the current config, and why not."""
        brs = self.branches
        if len(brs) < 2:
            return False, "single branch"
        b0 = brs[0]
        ns = sorted({b.N for b in brs})
        if len(ns) > 1:
            return False, f"branches have different state dims N ({ns}); unify n_state to enable"
        if len({b.H for b in brs}) > 1:
            return False, "branches have different channel counts H"
        if b0.disc_mode == "learned":
            return False, "disc_mode=learned (per-branch proj_dt)"
        if b0.gate.gate_kind == "random":
            return False, "gate_kind=random (merge would reorder RNG use)"
        if any(b.robust_guard for b in brs):
            return False, "robust_guard enabled"
        if any(b.fused_train for b in brs):
            return False, "eacs_fused_train enabled (fused path is faster)"
        return True, "available"

    def _merge_ok(self, x: Tensor) -> bool:
        """Check branches can share one sequential loop with identical semantics."""
        brs = self.branches
        if len(brs) < 2:
            return False
        b0 = brs[0]
        if not b0.uses_sequential_scan(x):
            return False
        if b0.disc_mode == "learned" or b0.gate.gate_kind == "random":
            return False
        for b in brs:
            if b.H != b0.H or b.N != b0.N:
                return False
            if b.disc_mode != b0.disc_mode or b.robust_guard:
                return False
            if b.dt_init != b0.dt_init or b.obs_norm.eps != b0.obs_norm.eps:
                return False
            if b.chunk_size != b0.chunk_size or b.training != b0.training:
                return False
            g = b.gate
            if (g.gate_kind != b0.gate.gate_kind or g.use_ste != b0.gate.use_ste
                    or g.norm_eta != b0.gate.norm_eta
                    or float(g.cpi_modulation) != float(b0.gate.cpi_modulation)):
                return False
        return True

    def _merged_branches(self, x: Tensor, timestamps: Tensor,
                         cpi: Tensor | None, frame_mask: Tensor | None
                         ) -> list[EACSOutput]:
        """Run all branches in one loop; bitwise identical to per-branch calls."""
        brs = list(self.branches)
        b0 = brs[0]
        B, L, _ = x.shape
        S = len(brs)

        us, Bcs, Ccs = [], [], []
        for br in brs:
            u, Bc, Cc = br._project(br.norm(x))
            if br.training:
                with torch.no_grad():
                    br._update_obs_stats(u, frame_mask)
            us.append(u); Bcs.append(Bc); Ccs.append(Cc)
        u = torch.stack(us, 2)
        Bc = torch.stack(Bcs, 2)
        Cc = torch.stack(Ccs, 2)
        t = timestamps.float().unsqueeze(-1).expand(B, L, S)
        cpi_s = None if cpi is None else cpi.unsqueeze(-1).expand(B, L, S)

        lam_s = torch.stack([br.lam() for br in brs]).to(torch.complex64)
        p = StepParams(
            lam=lam_s, inv_lam=torch.reciprocal(lam_s),
            log_dt_scale=torch.stack([br.log_dt_scale for br in brs]),
            D=torch.stack([br.D for br in brs]),
            obs_mean=torch.stack([br.obs_norm.running_mean for br in brs]),
            obs_var=torch.stack([br.obs_norm.running_var for br in brs]),
            obs_eps=b0.obs_norm.eps, gate=b0.gate, disc_mode=b0.disc_mode,
            dt_init=b0.dt_init, dt_max=self._dt_max.to(x.device),
            eps=torch.stack([br.gate.eps for br in brs]),
            temperature=torch.stack([br.gate.temperature for br in brs]))

        h_pi = torch.zeros(B, S, b0.H, b0.N, dtype=torch.complex64, device=x.device)
        state = (h_pi, t[:, 0] - b0.dt_init, u[:, 0], Bc[:, 0], Cc[:, 0])

        cs = b0.chunk_size
        if b0.training and cs and cs > 0 and L > cs:
            ys = gs = rs = None
            for k0 in range(0, L, cs):
                k1 = min(k0 + cs, L)
                y_c, g_c, r_c, state = ckpt.checkpoint(
                    scan_range, p, state, u, Bc, Cc, t, k0, k1, cpi_s,
                    None, use_reentrant=False)
                ys = _write_chunk(ys, y_c, L, k0, k1)
                gs = _write_chunk(gs, g_c, L, k0, k1)
                rs = _write_chunk(rs, r_c, L, k0, k1)
                del y_c, g_c, r_c
        else:
            ys, gs, rs, _ = scan_range(p, state, u, Bc, Cc, t, 0, L, cpi=cpi_s)

        outs = []
        for i, br in enumerate(brs):
            delta = br.proj_out(ys[..., i, :].contiguous().to(x.dtype))
            g_i = gs[..., i].contiguous()
            outs.append(EACSOutput(
                y=(x + delta if br.add_residual else delta),
                gates=g_i, update_rate=g_i.mean(), residual=rs[..., i].contiguous()))
        return outs


    def forward(self, x: Tensor, timestamps: Tensor,
                frame_mask: Tensor | None = None,
                cpi: Tensor | None = None) -> MultiScaleOutput:
        """Fuse branch outputs with input-dependent softmax weights.

        Args:
            cpi: optional (B,L) frame-level CPI signal forwarded to each gate.
            frame_mask: restricts observation stats and update-rate accounting to valid frames.
        """
        if self._merge_ok(x):
            outs: list[EACSOutput] = self._merged_branches(x, timestamps, cpi, frame_mask)
        else:
            outs = [br(x, timestamps, cpi=cpi, frame_mask=frame_mask)
                    for br in self.branches]
        deltas = torch.stack([o.y for o in outs], dim=-1)
        w = torch.softmax(self.fusion(x), dim=-1)
        fused = torch.einsum("bldn,bln->bld", deltas, w)
        y = self.out_norm(x + fused)
        gates = torch.stack([o.gates for o in outs], dim=1)
        res = torch.stack([o.residual for o in outs], dim=1)
        if frame_mask is not None:
            m = frame_mask.to(gates.dtype).unsqueeze(1)
            denom = m.sum().clamp_min(1.0)
            update_rate = (gates * m).sum() / (denom * self.n_branch)
            pbur = (gates * m).sum(dim=(0, 2)) / denom
        else:
            update_rate = gates.mean()
            pbur = torch.stack([o.update_rate for o in outs])
        return MultiScaleOutput(y=y, gates=gates, update_rate=update_rate,
                                per_branch_update_rate=pbur, fusion_weights=w, residual=res)

    def spectral_reg(self) -> Tensor:
        return sum(br.spectral_reg() for br in self.branches) / self.n_branch
