"""EACS：事件自适应连续扫描（设计方案 §4.1，本文唯一主线贡献 C1）。

把两件事统一进一个可微算子：
  (1) 真实物理 Δt 参数化的连续状态演化（连续范式）；
  (2) 预测编码式事件驱动稀疏更新（事件范式）+ 保持上次输入的 ZOH 预测。

前向采用**segment 形式**（忠实于设计方案与 Theorem 1）：预测与更新都从"上次提交"(committed)
状态 x_π 出发、以 gap (t_k−t_π) 演化：
    预测(跳过分支)：x̂_k = Ā(t_k−t_π) x_π + B̄(t_k−t_π; B_π) u_π      ← 保持上次输入 u_π
    更新分支    ：x_k = Ā(t_k−t_π) x_π + B̄(t_k−t_π; B_k) u_k       ← 注入当前输入
    门控        ：r_k=||u_k−ŷ_k||/(||u_k||+η) ; 跳过时 x_k=x̂_k、提交不变
静态输入下 x̂_k 收敛到稳态、残差恒 0 → 无虚假刷新（Theorem 1(B) 干净成立的前提）。

复杂度：单分支 O(L) 序列扫描（门控破坏 associativity）。生产可替换为 Mamba-2 可变Δt CUDA 内核。
数值：SSM 核心在 float32 复数下计算（S4D 惯例），输出回投影到输入 dtype。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from torch import Tensor

from ..ops.discretization import make_lambda, zoh_discretize, effective_dt
from ..ops.spectral_init import BranchSpec, init_branch_lambda, target_real_band
from .event_gate import EventGate, RunningStandardizer


@dataclass
class EACSOutput:
    y: Tensor              # (B, L, d_model) 时序建模输出（已含残差）
    gates: Tensor          # (B, L) 每步硬/软门控（1=更新, 0=跳过）
    update_rate: Tensor    # 标量，有效更新率 mean(gate)
    residual: Tensor       # (B, L) 归一化预测残差 r_k


def _readout(h: Tensor, C: Tensor) -> Tensor:
    """复数状态读出为实特征：y = Re(Σ_n C_n · h_n) → (B, H)。h:(B,H,N) C:(B,N)。"""
    return torch.einsum("bn,bhn->bh", C, h).real


class EACSLayer(nn.Module):
    """单个尺度分支的 EACS 层（pre-LN 残差块）。

    参数化：λ=-exp(a)+i·b（谱初始化，Re(λ)<0 恒稳定）；选择性 B_k,C_k 由输入动态生成（Mamba 风格）；
    逐通道时标缩放 log_dt_scale 保持"物理时间"语义。
    """

    def __init__(self, d_model: int, spec: BranchSpec,
                 gate: EventGate | None = None,
                 dt_init: float = 1.0, chunk_size: int = 0,
                 add_residual: bool = True, fused_train: bool = False,
                 robust_guard: bool = False, robust_k: int = 3,
                 robust_spike_ratio: float = 2.0, robust_eps_factor: float = 0.3,
                 disc_mode: str = "continuous", use_spectral_init: bool = True):
        super().__init__()
        self.d_model = d_model
        self.spec = spec
        self.H = d_model
        self.N = spec.n_state
        self.dt_init = dt_init
        self.chunk_size = chunk_size  # >0 且训练时启用分块梯度检查点省显存
        self.add_residual = add_residual  # 独立使用时 True；并入多尺度融合时 False
        self.fused_train = fused_train    # True：训练走可微融合 cell（常数图显存 + 可选CUDA前向）
        # 鲁棒兜底（设计 §4.4）：连续 robust_k 帧残差阶跃暴涨 → 该步临时降 ε 进高密度更新
        self.robust_guard = robust_guard
        self.robust_k = robust_k
        self.robust_spike_ratio = robust_spike_ratio
        self.robust_eps_factor = robust_eps_factor
        self.disc_mode = disc_mode        # "continuous"(默认) | "fixed" | "learned"（消融）

        # λ 参数：谱初始化（默认）或随机初始化（消融组3对照）
        if use_spectral_init:
            a_lnr, a_imag = init_branch_lambda(spec, self.H)
        else:
            a_lnr = torch.rand(self.H, self.N) * 4.0 - 2.0   # 随机 a=log(1/τ)
            a_imag = torch.randn(self.H, self.N)
        self.a_log_neg_real = nn.Parameter(a_lnr)   # (H, N)
        self.a_imag = nn.Parameter(a_imag)          # (H, N)
        self.log_dt_scale = nn.Parameter(torch.zeros(self.H))  # 逐通道时标，初始不缩放
        # learned 步长（Mamba 风格输入依赖 Δ，消融组2）：由 u_k 投影
        self.proj_dt = nn.Linear(self.H, self.H) if disc_mode == "learned" else None

        # 选择性投影
        self.norm = nn.LayerNorm(d_model)
        self.proj_u = nn.Linear(d_model, self.H)
        self.proj_B = nn.Linear(d_model, 2 * self.N)   # 复数 B（re, im）
        self.proj_C = nn.Linear(d_model, 2 * self.N)   # 复数 C
        self.D = nn.Parameter(torch.zeros(self.H))     # 直连跳接
        self.proj_out = nn.Linear(self.H, d_model)

        self.obs_norm = RunningStandardizer(self.H)
        self.gate = gate if gate is not None else EventGate()

        # 谱正则目标频段（供损失使用）
        lo, hi = target_real_band(spec)
        self.register_buffer("band_lo", torch.tensor(lo))
        self.register_buffer("band_hi", torch.tensor(hi))

    # ---- 参数装配 ----
    def lam(self) -> Tensor:
        return make_lambda(self.a_log_neg_real, self.a_imag)  # (H, N) complex

    def _fast_paths_ok(self) -> bool:
        """消融/鲁棒模式下强制走序列 cell（融合/CUDA 快路仅支持 continuous + event + 无兜底）。"""
        return (self.disc_mode == "continuous"
                and getattr(self.gate, "gate_kind", "event") == "event"
                and not self.robust_guard)

    def _project(self, x: Tensor):
        u = self.proj_u(x)                                   # (B,L,H)
        b = self.proj_B(x); Bc = torch.complex(b[..., :self.N], b[..., self.N:])  # (B,L,N)
        c = self.proj_C(x); Cc = torch.complex(c[..., :self.N], c[..., self.N:])  # (B,L,N)
        return u, Bc, Cc

    # ---- 单步 segment cell ----
    def _step(self, lam, h_pi, t_pi, u_pi, B_pi, C_pi,
              t_k, u_k, B_k, C_k, eps_override=None):
        """一步：返回 (h_cur, y_k, gate, r, 新提交状态...)。全部 vectorized over (B,H,N)。"""
        # 离散化步长（消融）：continuous=真实Δt / fixed=常数 / learned=输入依赖(Mamba风格)
        if self.disc_mode == "learned":
            dt_eff = F.softplus(self.proj_dt(u_k)).clamp(1e-3, 1e3)   # (B,H)
        else:
            delta = (t_k - t_pi).clamp_min(0.0)                      # (B,)
            if self.disc_mode == "fixed":
                delta = torch.full_like(delta, float(self.dt_init))
            dt_eff = effective_dt(delta.unsqueeze(-1), self.log_dt_scale)  # (B,H)

        # 预测分支（保持上次输入 u_pi、B_pi）
        dA_p, dB_p = zoh_discretize(lam, dt_eff, B_pi)       # (B,H,N)
        x_hat = dA_p * h_pi + dB_p * u_pi.unsqueeze(-1)      # (B,H,N)
        y_hat = _readout(x_hat, C_pi)                        # (B,H)

        # 门控（观测=当前输入 u_k；统计只读，更新统一在 forward 里做一次，避免检查点双更新/推理误更新）
        obs = self.obs_norm.apply_stats(u_k)
        g, r = self.gate(obs, self.obs_norm.apply_stats(y_hat), eps_override=eps_override)  # g:(B,), r:(B,)

        # 更新分支（注入当前输入 u_k、B_k）
        dA_u, dB_u = zoh_discretize(lam, dt_eff, B_k)
        h_upd = dA_u * h_pi + dB_u * u_k.unsqueeze(-1)       # (B,H,N)

        gs = g.view(-1, 1, 1)                                # 广播到 (B,H,N)
        gv = g.view(-1, 1)                                   # 广播到 (B,H)/(B,N)
        h_cur = gs * h_upd + (1 - gs) * x_hat                # 当前状态（输出用）
        # 软提交：更新则提交←当前，跳过则提交不变
        h_pi_n = gs * h_upd + (1 - gs) * h_pi
        t_pi_n = g * t_k + (1 - g) * t_pi
        u_pi_n = gv * u_k + (1 - gv) * u_pi
        B_pi_n = gv * B_k + (1 - gv) * B_pi
        C_pi_n = gv * C_k + (1 - gv) * C_pi

        y_k = _readout(h_cur, C_k) + self.D * u_k            # (B,H)
        return h_cur, y_k, g, r, h_pi_n, t_pi_n, u_pi_n, B_pi_n, C_pi_n

    def _scan_range(self, lam, state, u, Bc, Cc, t, k0, k1):
        """扫描 [k0,k1) 区间，返回 (ys, gs, rs, 末态)。state=(h_pi,t_pi,u_pi,B_pi,C_pi)。

        robust_guard 时维护连续残差暴涨计数：连续 robust_k 帧 r 阶跃暴涨 → 该步降 ε 进高密度更新，
        平稳后自动恢复（设计 §4.4）。注：分块检查点下 spike 计数在块边界重置（兜底机制，影响小）。
        """
        h_pi, t_pi, u_pi, B_pi, C_pi = state
        ys, gs, rs = [], [], []
        spike = torch.zeros(u.shape[0], device=u.device)
        prev_r = None
        for k in range(k0, k1):
            eps_ov = None
            if self.robust_guard:
                base = self.gate.eps
                eps_ov = torch.where(spike >= self.robust_k,
                                     base * self.robust_eps_factor, base)   # (B,)
            h_cur, y_k, g, r, h_pi, t_pi, u_pi, B_pi, C_pi = self._step(
                lam, h_pi, t_pi, u_pi, B_pi, C_pi,
                t[:, k], u[:, k], Bc[:, k], Cc[:, k], eps_override=eps_ov)
            ys.append(y_k); gs.append(g); rs.append(r)
            if self.robust_guard:
                if prev_r is not None:
                    is_spike = r > prev_r * self.robust_spike_ratio          # 残差阶跃暴涨
                    spike = torch.where(is_spike, spike + 1.0, torch.zeros_like(spike))
                prev_r = r.detach()
        ys = torch.stack(ys, 1); gs = torch.stack(gs, 1); rs = torch.stack(rs, 1)
        return ys, gs, rs, (h_pi, t_pi, u_pi, B_pi, C_pi)

    def forward(self, x: Tensor, timestamps: Tensor) -> EACSOutput:
        """x:(B,L,d_model) timestamps:(B,L) 秒。返回 EACSOutput（y 已含残差连接）。"""
        B, L, _ = x.shape
        # 推理路径：GPU 上有融合内核则整段一次跑完（显存 O(B·H·N)）；否则走下方序列 cell
        if (not self.training) and x.is_cuda and self._fast_paths_ok():
            try:
                from ..ops.eacs_gated_scan import gated_kernel_available
                if gated_kernel_available():
                    return self._forward_fused(x, timestamps)
            except Exception:
                pass
        x_in = x
        x = self.norm(x)
        # SSM 核心用 float32 复数
        u, Bc, Cc = self._project(x)
        u = u.float(); Bc = Bc.to(torch.complex64); Cc = Cc.to(torch.complex64)
        lam = self.lam().to(torch.complex64)
        t = timestamps.float()

        # 训练可微融合路：常数图显存的 autograd.Function（backward 逆时重算+逐步VJP）
        if self.training and self.fused_train and self._fast_paths_ok():
            _ = self.obs_norm(u.reshape(-1, self.H))          # 更新观测运行统计（融合内部用固定buffer）
            from ..ops.eacs_cell_train import fused_gated_scan
            ys, gs, rs = fused_gated_scan(
                u, Bc, Cc, t, lam, self.log_dt_scale, self.D, self.gate.eps,
                self.obs_norm.running_mean, self.obs_norm.running_var,
                float(self.gate.temperature), self.gate.norm_eta,
                self.obs_norm.eps, self.dt_init, gate_mode="ste", use_kernel=True)
            delta = self.proj_out(ys.to(x_in.dtype))
            y = x_in + delta if self.add_residual else delta
            return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs)

        # 初始提交：把提交时刻回退 dt_init，使第 0 帧输入即被注入
        if self.training:
            with torch.no_grad():                       # 统一更新观测统计一次（scan 内只读）
                self.obs_norm(u.reshape(-1, self.H))
        dt0 = self.dt_init
        h_pi = torch.zeros(B, self.H, self.N, dtype=torch.complex64, device=x.device)
        t_pi = t[:, 0] - dt0
        u_pi = u[:, 0]
        B_pi = Bc[:, 0]
        C_pi = Cc[:, 0]
        state = (h_pi, t_pi, u_pi, B_pi, C_pi)

        cs = self.chunk_size
        if self.training and cs and cs > 0 and L > cs:
            # 分块梯度检查点：只保存块边界状态，块内重算 → 省激活显存
            ys_all, gs_all, rs_all = [], [], []
            for k0 in range(0, L, cs):
                k1 = min(k0 + cs, L)
                ys, gs, rs, state = ckpt.checkpoint(
                    self._scan_range, lam, state, u, Bc, Cc, t, k0, k1,
                    use_reentrant=False)
                ys_all.append(ys); gs_all.append(gs); rs_all.append(rs)
            ys = torch.cat(ys_all, 1); gs = torch.cat(gs_all, 1); rs = torch.cat(rs_all, 1)
        else:
            ys, gs, rs, state = self._scan_range(lam, state, u, Bc, Cc, t, 0, L)

        delta = self.proj_out(ys.to(x_in.dtype))            # (B,L,d) 分支增量
        y = x_in + delta if self.add_residual else delta
        return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs)

    @torch.no_grad()
    def _forward_fused(self, x: Tensor, timestamps: Tensor) -> EACSOutput:
        """推理专用：调用融合门控内核整段扫描，等价于 eval 模式的序列 cell 前向。"""
        from ..ops.eacs_gated_scan import eacs_cell_forward
        x_in = x
        xn = self.norm(x)
        u, Bc, Cc = self._project(xn)
        ys, gates, resid = eacs_cell_forward(
            self.lam(), self.log_dt_scale, u, Bc, Cc, timestamps,
            self.D, self.obs_norm.running_mean, self.obs_norm.running_var,
            eps=float(self.gate.eps.item()), eta=self.gate.norm_eta,
            var_eps=self.obs_norm.eps, dt_init=self.dt_init)
        delta = self.proj_out(ys.to(x_in.dtype))
        y = x_in + delta if self.add_residual else delta
        return EACSOutput(y=y, gates=gates, update_rate=gates.mean(), residual=resid)

    # ---- 谱正则：惩罚特征值实部离开目标频段 ----
    def spectral_reg(self) -> Tensor:
        re = -torch.exp(self.a_log_neg_real)                 # Re(λ)
        below = torch.relu(self.band_lo - re)                # < lo（更负）
        above = torch.relu(re - self.band_hi)                # > hi（更接近0）
        return (below + above).pow(2).mean()

    @torch.no_grad()
    def run_with_commits(self, x: Tensor, timestamps: Tensor):
        """推理：运行前向并记录每帧提交状态，供任意时刻连续查询（设计方案 §4.3）。

        返回 (EACSOutput, commits)；commits 字典含逐帧提交 (h,t,u,B,C) 与 lam，
        对查询时刻 t*∈[t_k,t_{k+1}) 取 index k 的提交演化即可。
        """
        B, L, _ = x.shape
        x_in = x
        xn = self.norm(x)
        u, Bc, Cc = self._project(xn)
        u = u.float(); Bc = Bc.to(torch.complex64); Cc = Cc.to(torch.complex64)
        lam = self.lam().to(torch.complex64); t = timestamps.float()
        h_pi = torch.zeros(B, self.H, self.N, dtype=torch.complex64, device=x.device)
        t_pi = t[:, 0] - self.dt_init; u_pi = u[:, 0]; B_pi = Bc[:, 0]; C_pi = Cc[:, 0]
        ys, gs, rs = [], [], []
        H_pi, T_pi, U_pi, B_pis, C_pis = [], [], [], [], []
        for k in range(L):
            h_cur, y_k, g, r, h_pi, t_pi, u_pi, B_pi, C_pi = self._step(
                lam, h_pi, t_pi, u_pi, B_pi, C_pi, t[:, k], u[:, k], Bc[:, k], Cc[:, k])
            ys.append(y_k); gs.append(g); rs.append(r)
            H_pi.append(h_pi); T_pi.append(t_pi); U_pi.append(u_pi); B_pis.append(B_pi); C_pis.append(C_pi)
        ys = torch.stack(ys, 1); gs = torch.stack(gs, 1); rs = torch.stack(rs, 1)
        delta = self.proj_out(ys.to(x_in.dtype))
        y = x_in + delta if self.add_residual else delta
        commits = dict(h=torch.stack(H_pi, 1), t=torch.stack(T_pi, 1), u=torch.stack(U_pi, 1),
                       B=torch.stack(B_pis, 1), C=torch.stack(C_pis, 1), lam=lam,
                       frame_t=t, log_dt_scale=self.log_dt_scale.detach())
        return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs), commits
