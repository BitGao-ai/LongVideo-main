"""多尺度 EACS：短/中/长三分支 + 输入依赖门控融合（设计方案 §3.2）。

三个分支按谱初始化覆盖不同记忆时长（秒级/分钟级/小时级），各自独立事件门控；
输出用输入依赖的 Softmax 门控权重动态加权融合，实现"不同内容差异化激活不同尺度"。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torch import Tensor

from ..ops.spectral_init import BranchSpec, DEFAULT_BRANCHES
from .eacs import EACSLayer, EACSOutput, StepParams, scan_range
from .event_gate import EventGate


@dataclass
class MultiScaleOutput:
    y: Tensor                        # (B, L, d) 融合后时序表示（已含残差）
    gates: Tensor                    # (B, n_branch, L)
    update_rate: Tensor              # 标量，整体有效更新率
    per_branch_update_rate: Tensor   # (n_branch,) 各分支更新率
    fusion_weights: Tensor           # (B, L, n_branch)
    residual: Tensor                 # (B, n_branch, L) 各分支预测残差


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
        self.fusion = nn.Linear(d_model, self.n_branch)   # 输入依赖门控权重
        self.out_norm = nn.LayerNorm(d_model)
        # 各分支的 Δt 上界（10·τ_max，随分支谱段而异）。并轴扫描要按分支取上界，
        # 每次 forward 现建一个 (S,1) 张量就是一次 host→device 拷贝，故存成 buffer；
        # persistent=False 使它**不进 state_dict**——否则旧检查点会平白多出缺失键，
        # 而 load_checkpoint 是按缺失比例 raise 的。
        self.register_buffer(
            "_dt_max", torch.tensor([b.dt_max for b in self.branches]).view(-1, 1),
            persistent=False)

    # ---- 分支并轴扫描（吞吐主开关）----
    def _merge_ok(self, x: Tensor) -> bool:
        """三个分支能不能并进一个 Python 逐帧循环？

        逐分支扫描时，每步只是 (B,H,N) 这么大的一串小算子，**耗时由调度而非浮点运算
        决定**——实测单步 fwd+bwd 里 90% 花在扫描上，而把三个分支并轴后同样的算术只需
        1/3 的循环次数、每个 kernel 大 3 倍。

        只有在"本来就要逐帧扫"且各分支的门控/离散化语义完全一致时才并轴；任何一条不满足
        就退回逐分支（结果完全相同，只是慢）。特别地：
          - 已有融合快路（CUDA 推理内核 / 可微融合训练）比并轴快得多，让位；
          - gate_kind="random" 要消耗 RNG，并轴会改变随机数的取用顺序 → 不并；
          - robust_guard 的 spike 计数是逐分支状态机，语义不同 → 不并；
          - disc_mode="learned" 各分支有自己的 proj_dt 线性层 → 不并。
        ε 与温度都按分支堆叠后传进门控，因此**不需要**断言它们跨分支相等（那会是一次
        GPU→CPU 同步）。
        """
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
        """把各分支并进一个逐帧循环，返回与逐分支调用**逐位相同**的 EACSOutput 列表。

        并轴只发生在扫描内部：LayerNorm / 选择性投影 / proj_out 都是沿 L 并行的大算子，
        逐分支跑本来就没有调度问题，保持原样（也保证 state_dict 与梯度路径不变）。
        """
        brs = list(self.branches)
        b0 = brs[0]
        B, L, _ = x.shape
        S = len(brs)

        us, Bcs, Ccs = [], [], []
        for br in brs:
            u, Bc, Cc = br._project(br.norm(x))
            if br.training:
                with torch.no_grad():      # 统一更新观测统计一次（scan 内只读）
                    br._update_obs_stats(u, frame_mask)
            us.append(u); Bcs.append(Bc); Ccs.append(Cc)
        u = torch.stack(us, 2)                    # (B,L,S,H)
        Bc = torch.stack(Bcs, 2)                  # (B,L,S,N)
        Cc = torch.stack(Ccs, 2)                  # (B,L,S,N)
        # 时间与 CPI 对分支是共享的，expand 出 S 轴只是视图，不复制
        t = timestamps.float().unsqueeze(-1).expand(B, L, S)
        cpi_s = None if cpi is None else cpi.unsqueeze(-1).expand(B, L, S)

        lam_s = torch.stack([br.lam() for br in brs]).to(torch.complex64)     # (S,H,N)
        p = StepParams(
            lam=lam_s, inv_lam=torch.reciprocal(lam_s),
            log_dt_scale=torch.stack([br.log_dt_scale for br in brs]),       # (S,H)
            D=torch.stack([br.D for br in brs]),                             # (S,H)
            obs_mean=torch.stack([br.obs_norm.running_mean for br in brs]),  # (S,H)
            obs_var=torch.stack([br.obs_norm.running_var for br in brs]),
            obs_eps=b0.obs_norm.eps, gate=b0.gate, disc_mode=b0.disc_mode,
            dt_init=b0.dt_init, dt_max=self._dt_max.to(x.device),            # (S,1)
            # ε 与温度逐分支堆叠：ε 的梯度仍各自回到自己的 eps_raw，退火乘子也各归各的
            eps=torch.stack([br.gate.eps for br in brs]),                    # (S,)
            temperature=torch.stack([br.gate.temperature for br in brs]))    # (S,)

        h_pi = torch.zeros(B, S, b0.H, b0.N, dtype=torch.complex64, device=x.device)
        state = (h_pi, t[:, 0] - b0.dt_init, u[:, 0], Bc[:, 0], Cc[:, 0])

        cs = b0.chunk_size
        if b0.training and cs and cs > 0 and L > cs:
            ys_all, gs_all, rs_all = [], [], []
            for k0 in range(0, L, cs):
                ys, gs, rs, state = ckpt.checkpoint(
                    scan_range, p, state, u, Bc, Cc, t, k0, min(k0 + cs, L), cpi_s,
                    None, use_reentrant=False)
                ys_all.append(ys); gs_all.append(gs); rs_all.append(rs)
            ys = torch.cat(ys_all, 1); gs = torch.cat(gs_all, 1); rs = torch.cat(rs_all, 1)
        else:
            ys, gs, rs, _ = scan_range(p, state, u, Bc, Cc, t, 0, L, cpi=cpi_s)

        outs = []
        for i, br in enumerate(brs):
            # .contiguous()：切片出来的 (B,L,H) 步幅与逐分支路径不同，先落回同一内存布局
            # 再喂 proj_out，才能保证与逐分支走同一条 gemm、逐位相同。
            delta = br.proj_out(ys[..., i, :].contiguous().to(x.dtype))
            g_i = gs[..., i].contiguous()
            outs.append(EACSOutput(
                y=(x + delta if br.add_residual else delta),
                gates=g_i, update_rate=g_i.mean(), residual=rs[..., i].contiguous()))
        return outs


    def forward(self, x: Tensor, timestamps: Tensor,
                frame_mask: Tensor | None = None,
                cpi: Tensor | None = None) -> MultiScaleOutput:
        """cpi: 可选 (B,L) 帧级 CPI 信号，贯通各分支 EventGate（P1.5 统一主线）。

        frame_mask 有两个用途：① 传给各分支，使观测运行统计只吃有效帧（padding 会让
        门控残差的尺度漂移，进而污染更新率指标）；② 只在有效帧上统计更新率。
        """
        if self._merge_ok(x):
            outs: list[EACSOutput] = self._merged_branches(x, timestamps, cpi, frame_mask)
        else:
            outs = [br(x, timestamps, cpi=cpi, frame_mask=frame_mask)
                    for br in self.branches]
        deltas = torch.stack([o.y for o in outs], dim=-1)          # (B,L,d,n_branch)
        w = torch.softmax(self.fusion(x), dim=-1)                  # (B,L,n_branch)
        fused = torch.einsum("bldn,bln->bld", deltas, w)
        y = self.out_norm(x + fused)
        gates = torch.stack([o.gates for o in outs], dim=1)        # (B,n_branch,L)
        res = torch.stack([o.residual for o in outs], dim=1)
        if frame_mask is not None:
            # 只在有效帧上统计更新率（避免 padding 稀释更新率正则与效率指标）
            m = frame_mask.to(gates.dtype).unsqueeze(1)            # (B,1,L)
            denom = m.sum().clamp_min(1.0)
            update_rate = (gates * m).sum() / (denom * self.n_branch)
            pbur = (gates * m).sum(dim=(0, 2)) / denom             # (n_branch,)
        else:
            update_rate = gates.mean()
            pbur = torch.stack([o.update_rate for o in outs])
        return MultiScaleOutput(y=y, gates=gates, update_rate=update_rate,
                                per_branch_update_rate=pbur, fusion_weights=w, residual=res)

    def spectral_reg(self) -> Tensor:
        return sum(br.spectral_reg() for br in self.branches) / self.n_branch
