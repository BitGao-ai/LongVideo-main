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

from ..ops.discretization import (make_lambda, zoh_kernels, zoh_apply_B,
                                  effective_dt)
from ..ops.spectral_init import BranchSpec, init_branch_lambda, target_real_band
from .event_gate import EventGate, RunningStandardizer


@dataclass
class EACSOutput:
    y: Tensor              # (B, L, d_model) 时序建模输出（已含残差）
    gates: Tensor          # (B, L) 每步硬/软门控（1=更新, 0=跳过）
    update_rate: Tensor    # 标量，有效更新率 mean(gate)
    residual: Tensor       # (B, L) 归一化预测残差 r_k


def _readout(h: Tensor, C: Tensor) -> Tensor:
    """复数状态读出为实特征：y = Re(Σ_n C_n · h_n)。

    形状对 (h, C) 支持两档，取决于是否并轴：
        单分支   h:(B,H,N)   C:(B,N)   → (B,H)
        多分支并轴 h:(B,S,H,N) C:(B,S,N) → (B,S,H)
    两档走同一个 einsum 归约轴 n，实测逐位相同（见 regression_branch_merge）。
    """
    eq = "bn,bhn->bh" if h.dim() == 3 else "bsn,bshn->bsh"
    return torch.einsum(eq, C, h).real


@dataclass
class StepParams:
    """一步 segment cell 所需的全部参数——**单分支与多分支并轴共用同一份数学**。

    形状约定：并轴时在 H 之前多一个分支轴 S，其余靠广播吃掉：
        lam (H,N)→(S,H,N)   log_dt_scale/D/obs_* (H,)→(S,H)   dt_max 标量→(S,1)
    eps / temperature 为 None 时表示"用 gate 自己的"（单分支）；并轴时传各分支堆叠后的
    张量，这样 ε 的梯度仍各自回到自己的 eps_raw，退火乘子也各归各的。

    存在的理由：三个分支各跑一个 Python 逐帧循环时，90% 的单步耗时花在扫描上，而其中
    绝大部分是小张量的调度开销而非浮点运算。把分支并进一个循环后循环次数 3→1、每个
    kernel 大 3 倍。要点是**不能**为此复制一份 cell 数学——N1 就是"改写时把同一段逻辑
    抄到两处"造成的回归。所以这里让两条路径调用同一个 eacs_step。
    """
    lam: Tensor
    inv_lam: Tensor
    log_dt_scale: Tensor
    D: Tensor
    obs_mean: Tensor
    obs_var: Tensor
    obs_eps: float
    gate: EventGate
    disc_mode: str
    dt_init: float
    dt_max: Tensor | float
    proj_dt: nn.Module | None = None
    eps: Tensor | None = None
    temperature: Tensor | None = None


@dataclass
class RobustCfg:
    """§4.4 残差暴涨兜底的三个超参（仅逐分支路径使用）。"""
    k: int
    spike_ratio: float
    eps_factor: float


def _standardize(x: Tensor, p: StepParams) -> Tensor:
    """等价于 RunningStandardizer.apply_stats，但吃并轴后的 (S,H) 统计。"""
    return (x - p.obs_mean) / torch.sqrt(p.obs_var + p.obs_eps)


def eacs_step(p: StepParams, h_pi, t_pi, u_pi, B_pi, C_pi,
              t_k, u_k, B_k, C_k, eps_override=None, cpi_k=None):
    """一步 segment cell：返回 (h_cur, y_k, gate, r, 新提交状态...)。

    对形状不做任何假设——所有运算都是逐元素或沿最后一维的归约，因此单分支 (B,·) 与
    多分支并轴 (B,S,·) 走的是同一段代码、同一组 kernel。
    """
    # 离散化步长（消融）：continuous=真实Δt / fixed=常数 / learned=输入依赖(Mamba风格)
    if p.disc_mode == "learned":
        dt_eff = F.softplus(p.proj_dt(u_k)).clamp(1e-3, 1e3)     # (…,H)
    else:
        delta = (t_k - t_pi).clamp_min(0.0)                      # (…,)
        if p.disc_mode == "fixed":
            delta = torch.full_like(delta, float(p.dt_init))
        dt_eff = effective_dt(delta.unsqueeze(-1), p.log_dt_scale,
                              dt_max=p.dt_max)                   # (…,H)

    # 预测/更新两分支从同一个提交时刻起算、共用同一个 Δ，故离散化核只算一次：
    # dA=exp(λΔ) 与 dB_bar=(exp(λΔ)-1)/λ 都与选择性 B 无关，只在最后乘各自的 B。
    # （此前两分支各调一次 zoh_discretize，z/exp/expm1 全套中间量白算一遍。）
    # 1/λ 由 StepParams 在扫描外算一次带进来——λ 是参数，逐帧重算复数除法是纯浪费。
    dA, dB_bar = zoh_kernels(p.lam, dt_eff, p.inv_lam)   # (…,H,N)

    # 自由演化项 Ā(Δ)·x_π 同样是两分支共有的——预测与更新只差"注入哪个输入"。
    # 此前 x_hat 与 h_upd 各写了一次 dA*h_pi，即每步多一次 (…,H,N) 复数乘法、
    # 多一份留给反传的中间量；提出来复用后逐位不变。
    h_free = dA * h_pi                                   # (…,H,N)

    # 预测分支（保持上次输入 u_pi、B_pi）
    x_hat = h_free + zoh_apply_B(dB_bar, B_pi) * u_pi.unsqueeze(-1)
    y_hat = _readout(x_hat, C_pi)                        # (…,H)

    # 门控（观测=当前输入 u_k；统计只读，更新统一在 forward 里做一次，避免检查点双更新/推理误更新）
    obs = _standardize(u_k, p)
    eps_ov = eps_override if eps_override is not None else p.eps
    g, r = p.gate(obs, _standardize(y_hat, p), eps_override=eps_ov,
                  cpi=cpi_k, temp_override=p.temperature)   # g,r:(…,)

    # 更新分支（注入当前输入 u_k、B_k）——复用上面的 dA·x_π 与 dA/dB_bar
    h_upd = h_free + zoh_apply_B(dB_bar, B_k) * u_k.unsqueeze(-1)

    gs = g[..., None, None]                              # 广播到 (…,H,N)
    gv = g[..., None]                                    # 广播到 (…,H)/(…,N)
    # h_cur 与 h_pi_n 的 g·h_upd 项完全相同，只有 (1−g) 乘的对象不同（当前预测 vs 旧提交），
    # 提出来复用又省掉一次 (…,H,N) 复数乘法；逐位不变。
    g_upd = gs * h_upd
    one_g = 1 - gs
    h_cur = g_upd + one_g * x_hat                        # 当前状态（输出用）
    # 软提交：更新则提交←当前，跳过则提交不变
    h_pi_n = g_upd + one_g * h_pi
    t_pi_n = g * t_k + (1 - g) * t_pi
    u_pi_n = gv * u_k + (1 - gv) * u_pi
    B_pi_n = gv * B_k + (1 - gv) * B_pi
    C_pi_n = gv * C_k + (1 - gv) * C_pi

    y_k = _readout(h_cur, C_k) + p.D * u_k               # (…,H)
    return h_cur, y_k, g, r, h_pi_n, t_pi_n, u_pi_n, B_pi_n, C_pi_n


def scan_range(p: StepParams, state, u, Bc, Cc, t, k0, k1,
               cpi=None, robust: RobustCfg | None = None):
    """扫描 [k0,k1)，返回 (ys, gs, rs, 末态)。state=(h_pi,t_pi,u_pi,B_pi,C_pi)。

    单分支与多分支并轴共用：p 与输入的形状决定走哪一档，循环体本身完全相同。
    robust_guard 时维护连续残差暴涨计数：连续 robust.k 帧 r 阶跃暴涨 → 该步降 ε 进
    高密度更新，平稳后自动恢复（设计 §4.4）。注：分块检查点下 spike 计数在块边界重置
    （兜底机制，影响小）。
    """
    h_pi, t_pi, u_pi, B_pi, C_pi = state
    ys, gs, rs = [], [], []
    spike = prev_r = base_eps = None
    if robust is not None:
        spike = torch.zeros_like(t_pi)
        # ε 在整段扫描里是常数（sigmoid(eps_raw)·anneal），逐步重算是白算一次前向图
        base_eps = p.gate.eps if p.eps is None else p.eps
    for k in range(k0, k1):
        eps_ov = None
        if robust is not None:
            eps_ov = torch.where(spike >= robust.k,
                                 base_eps * robust.eps_factor, base_eps)
        h_cur, y_k, g, r, h_pi, t_pi, u_pi, B_pi, C_pi = eacs_step(
            p, h_pi, t_pi, u_pi, B_pi, C_pi,
            t[:, k], u[:, k], Bc[:, k], Cc[:, k], eps_override=eps_ov,
            cpi_k=(cpi[:, k] if cpi is not None else None))
        ys.append(y_k); gs.append(g); rs.append(r)
        if robust is not None:
            if prev_r is not None:
                is_spike = r > prev_r * robust.spike_ratio          # 残差阶跃暴涨
                spike = torch.where(is_spike, spike + 1.0, torch.zeros_like(spike))
            prev_r = r.detach()
    ys = torch.stack(ys, 1); gs = torch.stack(gs, 1); rs = torch.stack(rs, 1)
    return ys, gs, rs, (h_pi, t_pi, u_pi, B_pi, C_pi)



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
        # 记下谱初始化开关：λ 一旦初始化完就看不出来源了，而消融脚本需要能**事后核对**
        # 模型的实际构造与 cfg 一致（见 models.assert_config_applied）。普通属性，不进 state_dict。
        self.use_spectral_init = use_spectral_init
        # 有效 Δt 的上界。_step 里的 delta 是「距上次提交」而非帧间隔——事件门控连续跳过时
        # 它单调累积，长视频静态段轻易超过旧的固定上界 1000s；而 long 分支 τ_max=3600s
        # 恰好最容易撞上，clamp 处梯度为 0，log_dt_scale 在该区域也学不动。
        # 取 10·τ_max：此时 |exp(λΔ)| ≤ e^{-10} ≈ 4.5e-5，状态已实质衰减完，
        # 再往上截断不会丢失任何动力学信息，同时避免复指数相位精度失控。
        self.dt_max = max(1e3, 10.0 * float(spec.tau_max))

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
        """消融/鲁棒/CPI 调制模式下强制走序列 cell。
    
        融合/CUDA 快路仅支持 continuous + event + 无兖底；且快路内核不接受 cpi 参数——
        一旦启用 CPI 调制（cpi_modulation>0）必须回退序列路径，否则 CPI 信号被静默吞掉
        （门控行为与不传 cpi 完全相同，创新点 P1.5 失效）。
        """
        return (self.disc_mode == "continuous"
                and getattr(self.gate, "gate_kind", "event") == "event"
                and not self.robust_guard
                and getattr(self.gate, "cpi_modulation", 0.0) == 0)

    def _cuda_fused_ok(self, x: Tensor) -> bool:
        """推理侧 CUDA 融合门控内核当前可用吗（不含调用本身可能抛的异常）。"""
        if self.training or not x.is_cuda or not self._fast_paths_ok():
            return False
        try:
            from ..ops.eacs_gated_scan import gated_kernel_available
            return bool(gated_kernel_available())
        except Exception:
            return False

    def uses_sequential_scan(self, x: Tensor) -> bool:
        """本层在当前状态下会走逐帧序列扫描（而非 CUDA 融合前向 / 可微融合训练）吗？

        供 MultiScaleEACS 决定能否把各分支并进一个循环：那条并轴路径本身也是 Python
        逐帧循环，只有在**本来就要逐帧扫**的时候才是净收益；已有的融合快路（整段一次
        跑完）比它快得多，不能被抢走。
        """
        if self._cuda_fused_ok(x):
            return False
        return not (self.training and self.fused_train and self._fast_paths_ok())

    def _project(self, x: Tensor):
        # autocast(bf16) 下 Linear 输出为 bf16，而 torch.complex 不支持 bf16 输入，
        # 必须先升精度 float32 再组装复数（SSM 核心本就按 float32 复数计算）。
        u = self.proj_u(x).float()                                   # (B,L,H)
        b = self.proj_B(x).float()
        Bc = torch.complex(b[..., :self.N], b[..., self.N:])         # (B,L,N) complex64
        c = self.proj_C(x).float()
        Cc = torch.complex(c[..., :self.N], c[..., self.N:])         # (B,L,N) complex64
        return u, Bc, Cc

    # ---- 单步 segment cell ----
    def step_params(self, lam: Tensor) -> StepParams:
        """把本层的（未并轴）参数打包成 eacs_step 的入参。eps/temperature 留 None
        表示用 self.gate 自己的值——并轴路径才需要传堆叠后的版本。"""
        return StepParams(
            lam=lam, inv_lam=torch.reciprocal(lam), log_dt_scale=self.log_dt_scale, D=self.D,
            obs_mean=self.obs_norm.running_mean, obs_var=self.obs_norm.running_var,
            obs_eps=self.obs_norm.eps, gate=self.gate, disc_mode=self.disc_mode,
            dt_init=self.dt_init, dt_max=self.dt_max, proj_dt=self.proj_dt)

    def _robust_cfg(self) -> RobustCfg | None:
        return (RobustCfg(self.robust_k, self.robust_spike_ratio, self.robust_eps_factor)
                if self.robust_guard else None)

    def _step(self, lam, h_pi, t_pi, u_pi, B_pi, C_pi,
              t_k, u_k, B_k, C_k, eps_override=None, cpi_k=None):
        """一步 segment cell（薄封装，数学在模块级 eacs_step）。"""
        return eacs_step(self.step_params(lam), h_pi, t_pi, u_pi, B_pi, C_pi,
                         t_k, u_k, B_k, C_k, eps_override=eps_override, cpi_k=cpi_k)

    def _scan_range(self, lam, state, u, Bc, Cc, t, k0, k1, cpi=None):
        """扫描 [k0,k1)（薄封装，循环体在模块级 scan_range）。

        cpi: 可选 (B,L) 帧级 CPI 信号（P1.5 统一主线）。
        """
        return scan_range(self.step_params(lam), state, u, Bc, Cc, t, k0, k1,
                          cpi=cpi, robust=self._robust_cfg())

    def _update_obs_stats(self, u: Tensor, frame_mask: Tensor | None) -> None:
        """用当前 batch 的观测更新运行统计，**只取有效帧**。

        这份统计决定门控残差 r=‖obs-pred‖/(‖obs‖+η) 的尺度，进而决定与 ε 比较的结果，
        也就是"有效更新率"这个核心效率指标。若把 collate 填零的 padding 帧也算进去，
        统计会随批内 padding 比例漂移，更新率跨 batch/跨实验都不可比。
        """
        obs = u.reshape(-1, self.H) if frame_mask is None else u[frame_mask.bool()]
        if obs.numel():
            self.obs_norm(obs)

    def forward(self, x: Tensor, timestamps: Tensor, cpi: Tensor | None = None,
                frame_mask: Tensor | None = None) -> EACSOutput:
        """x:(B,L,d_model) timestamps:(B,L) 秒。cpi: 可选(B,L) CPI 信号（P1.5）。

        frame_mask: 可选 (B,L) bool 有效帧掩码。**只用于观测运行统计**——扫描本身是因果的，
        尾部 padding 不会影响前面有效帧的输出（已验证），但会污染 obs_norm 的统计。
        """
        B, L, _ = x.shape
        # 推理路径：GPU 上有融合内核则整段一次跑完（显存 O(B·H·N)）；否则走下方序列 cell。
        # try 仍包住调用本身：内核可能在首次 JIT 编译时才失败，那时应静默回退而非崩掉推理。
        if self._cuda_fused_ok(x):
            try:
                return self._forward_fused(x, timestamps)
            except Exception:
                pass
        x_in = x
        x = self.norm(x)
        # SSM 核心用 float32 复数（_project 内部已升精度）
        u, Bc, Cc = self._project(x)
        lam = self.lam().to(torch.complex64)
        t = timestamps.float()

        # 训练可微融合路：常数图显存的 autograd.Function（backward 逆时重算+逐步VJP）
        if self.training and self.fused_train and self._fast_paths_ok():
            with torch.no_grad():        # 更新观测运行统计（融合内部用固定 buffer）
                self._update_obs_stats(u, frame_mask)
            from ..ops.eacs_cell_train import fused_gated_scan
            ys, gs, rs = fused_gated_scan(
                u, Bc, Cc, t, lam, self.log_dt_scale, self.D, self.gate.eps,
                self.obs_norm.running_mean, self.obs_norm.running_var,
                float(self.gate.temperature), self.gate.norm_eta,
                self.obs_norm.eps, self.dt_init, gate_mode="ste", use_kernel=True,
                dt_max=self.dt_max)
            delta = self.proj_out(ys.to(x_in.dtype))
            y = x_in + delta if self.add_residual else delta
            return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs)

        # 初始提交：把提交时刻回退 dt_init，使第 0 帧输入即被注入
        if self.training:
            with torch.no_grad():                       # 统一更新观测统计一次（scan 内只读）
                self._update_obs_stats(u, frame_mask)
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
                    self._scan_range, lam, state, u, Bc, Cc, t, k0, k1, cpi,
                    use_reentrant=False)
                ys_all.append(ys); gs_all.append(gs); rs_all.append(rs)
            ys = torch.cat(ys_all, 1); gs = torch.cat(gs_all, 1); rs = torch.cat(rs_all, 1)
        else:
            ys, gs, rs, state = self._scan_range(lam, state, u, Bc, Cc, t, 0, L, cpi=cpi)

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
            var_eps=self.obs_norm.eps, dt_init=self.dt_init, dt_max=self.dt_max)
        delta = self.proj_out(ys.to(x_in.dtype))
        y = x_in + delta if self.add_residual else delta
        return EACSOutput(y=y, gates=gates, update_rate=gates.mean(), residual=resid)

    # ---- 谱正则：惩罚特征值实部离开目标频段 ----
    def spectral_reg(self) -> Tensor:
        re = -torch.exp(self.a_log_neg_real)                 # Re(λ)
        below = torch.relu(self.band_lo - re)                # < lo（更负）
        above = torch.relu(re - self.band_hi)                # > hi（更接近0）
        return (below + above).pow(2).mean()

    def _commit_range(self, lam, state, u, Bc, Cc, t, k0, k1, cpi=None):
        """扫描 [k0,k1) 并同时记录逐帧提交。返回 8 元组（全部为张量，供 checkpoint 使用）。

        末态即最后一帧的提交（h_pi 在处理完第 k 步后就等于 index k 的提交），
        故下一块的初始 state 直接取各序列的 [:, -1]，无需额外返回。

        cpi: 可选 (B,L) 帧级 CPI 信号，与 forward/_scan_range 同口径地送进 EventGate。
        缺了它，定位路径（continuous_readout → run_with_commits）的门控行为会和 QA 路径
        不一致——同一套 EACS 权重在两种门控规则下跑，P1.5 的 CPI 调制只对一半流程生效。
        """
        h_pi, t_pi, u_pi, B_pi, C_pi = state
        ys, gs, rs = [], [], []
        H, T, U, Bs, Cs = [], [], [], [], []
        p = self.step_params(lam)          # 逐步重建这个打包对象是白费，扫描期间它不变
        for k in range(k0, k1):
            _, y_k, g, r, h_pi, t_pi, u_pi, B_pi, C_pi = eacs_step(
                p, h_pi, t_pi, u_pi, B_pi, C_pi, t[:, k], u[:, k], Bc[:, k], Cc[:, k],
                cpi_k=(cpi[:, k] if cpi is not None else None))
            ys.append(y_k); gs.append(g); rs.append(r)
            H.append(h_pi); T.append(t_pi); U.append(u_pi); Bs.append(B_pi); Cs.append(C_pi)
        return (torch.stack(ys, 1), torch.stack(gs, 1), torch.stack(rs, 1),
                torch.stack(H, 1), torch.stack(T, 1), torch.stack(U, 1),
                torch.stack(Bs, 1), torch.stack(Cs, 1))

    def run_with_commits(self, x: Tensor, timestamps: Tensor, cpi: Tensor | None = None,
                         use_chunk: bool | None = None):
        """运行前向并记录每帧提交状态，供任意时刻连续查询（设计方案 §4.3）。

        返回 (EACSOutput, commits)；commits 字典含逐帧提交 (h,t,u,B,C) 与 lam，
        对查询时刻 t*∈[t_k,t_{k+1}) 取 index k 的提交演化即可。

        **梯度语义**：本方法不再自带 @torch.no_grad()。定位训练（grounding_forward →
        continuous_readout）必须让梯度经 commits 回传到 EACS/空间编码，否则整条时序
        主干拿不到梯度、只有打分头被训练（连续读出就成了随机特征）。推理侧由调用方
        自行包 `with torch.no_grad():`——scripts/infer_query、benchmarks/infer_grounding、
        models.encode_temporal 的 ssm 上下文分支、Trainer.validate 均已如此。

        use_chunk：None=按 chunk_size 自动决定；False=强制不分块。
        **调用方已经把本次调用包在外层梯度检查点里时必须传 False**：两层检查点会叠乘，
        外层重算一遍、内层再重算一遍，同一段扫描要跑三遍（实测定位训练每步 9×L 个 cell
        步，而下限是 3×L）。而内层分块在这条路上几乎不省显存——大头是本方法的**返回值**
        (B,L,H,N) 提交轨迹，那是输出不是内部激活，分块压不到（实测 8.176 vs 8.170
        MB/帧，差 0.07%）。所以在外层检查点里内层分块是纯付出、无收益。

        **显存**：需要梯度且启用分块时，逐帧循环按块做梯度检查点（与 forward 同策略），
        块内每步约 20 个 (B,H,N) 复数中间量不再常驻，只在反向逐块重算。不可省的是提交轨迹
        本身（(B,L,H,N) 复数，是本方法的返回值），它随 L 线性增长。
        """
        B, L, _ = x.shape
        x_in = x
        xn = self.norm(x)
        u, Bc, Cc = self._project(xn)
        lam = self.lam().to(torch.complex64); t = timestamps.float()
        h_pi = torch.zeros(B, self.H, self.N, dtype=torch.complex64, device=x.device)
        state = (h_pi, t[:, 0] - self.dt_init, u[:, 0], Bc[:, 0], Cc[:, 0])

        cs = self.chunk_size
        chunked = (torch.is_grad_enabled() and cs and cs > 0 and L > cs
                   and use_chunk is not False)
        if chunked:
            parts = []
            for k0 in range(0, L, cs):
                out = ckpt.checkpoint(self._commit_range, lam, state, u, Bc, Cc, t,
                                      k0, min(k0 + cs, L), cpi, use_reentrant=False)
                parts.append(out)
                state = tuple(o[:, -1] for o in out[3:])   # 末帧提交 = 下一块初始 state
            ys, gs, rs, H, T, U, Bs, Cs = [
                torch.cat([p[i] for p in parts], 1) for i in range(8)]
        else:
            ys, gs, rs, H, T, U, Bs, Cs = self._commit_range(
                lam, state, u, Bc, Cc, t, 0, L, cpi)

        delta = self.proj_out(ys.to(x_in.dtype))
        y = x_in + delta if self.add_residual else delta
        commits = dict(h=H, t=T, u=U, B=Bs, C=Cs, lam=lam,
                       frame_t=t, log_dt_scale=self.log_dt_scale, dt_max=self.dt_max)
        return EACSOutput(y=y, gates=gs, update_rate=gs.mean(), residual=rs), commits
