"""门控 EACS cell 的可微融合训练路径（设计方案 §4.1 训练）。

把 L 步门控扫描封装成单个 torch.autograd.Function：
  - forward：无梯度顺序扫描（可用融合 CUDA 前向内核加速），**不建全局计算图**；
  - backward：逆时重算 + **逐步 autograd 求 VJP** 的反向递推，只携带状态伴随（**autograd 图显存 O(1)**、
    与 L 无关；另需保存状态轨迹 O(L·H·N)），正确构造（用 autograd 求每步 VJP，无需手推公式），CPU/GPU 通用。

相比"朴素 Python 循环 + 全局 autograd 图"（激活图显存 O(L·每步激活)，远大于状态轨迹），本 Function
是门控 cell 的可微融合训练实现。门控用直通估计（STE）：forward 硬门控、backward 软门控梯度。

验证（CPU，无需 GPU）：
  1) soft 模式（纯 sigmoid 门控，全可微）→ torch.autograd.gradcheck(double)；
  2) ste 模式 → 梯度与现有 eacs.py `_scan_range` 的 autograd 路径逐元素一致。
"""
from __future__ import annotations

import torch
from torch import Tensor

from .discretization import complex_expm1


def _step_functional(state, inputs, params, T: float, eta: float,
                     var_eps: float, mean: Tensor, var: Tensor,
                     gate_mode: str = "ste", dt_max: float = 1e3,
                     g_hard: Tensor | None = None,
                     inv_lam: Tensor | None = None):
    """单步 segment cell（与 EACSLayer._step / modules.eacs.eacs_step 同构）。
    state/inputs/params 为张量元组。

    g_hard：可选的**外部硬门控**。硬门控 `r > eps` 是不连续的，只要前向（可能走 CUDA
    融合内核）与反向重算在 r 上差 1 ulp，某一步的门控就会翻转、之后整条轨迹分叉，
    backward 算的就是另一条轨迹的梯度且不会报任何错。把前向真实产生的 gate 传进来
    强制复用，即可保证两条路径逐步一致；STE 的软梯度分量仍照常参与反传。

    inv_lam：可选的预算 1/λ（复数除法约是复数乘法的 8 倍成本，而 λ 在整段扫描里不变）。
    **只有无梯度的前向重算循环能传**——反向 VJP 循环里 lam_ 是每步新建的叶子，必须让
    1/λ 由它现算出来，否则 dBbar 对 λ 的依赖被切断、dlam 少掉这一路贡献且不会报错。
    """
    h_pi, t_pi, u_pi, B_pi, C_pi = state
    u_l, B_l, C_l, t_l = inputs
    lam, log_dt, D, eps = params

    delta = (t_l - t_pi).clamp_min(0.0)
    dt_eff = (delta.unsqueeze(-1) * torch.exp(log_dt)).clamp(1e-3, dt_max)   # (B,H)
    z = lam * dt_eff.unsqueeze(-1)                                          # (B,H,N)
    dA = torch.exp(z)
    if inv_lam is None:
        inv_lam = torch.reciprocal(lam)
    dBbar = complex_expm1(z, exp_z=dA) * inv_lam    # 复用 dA，不重复算 exp(z)
    h_free = dA * h_pi                              # 预测/更新两分支共有，只算一次
    xhat = h_free + dBbar * B_pi.unsqueeze(1) * u_pi.unsqueeze(-1)
    yhat = (C_pi.unsqueeze(1) * xhat).real.sum(-1)                          # (B,H)

    s = torch.sqrt(var + var_eps)
    diff = (u_l - yhat) / s
    obs = (u_l - mean) / s
    r = diff.norm(dim=-1) / (obs.norm(dim=-1) + eta)                        # (B,)
    soft = torch.sigmoid((r - eps) / max(T, 1e-4))
    if gate_mode == "soft":
        g = soft
    else:                                                                  # STE：前向硬、反向软
        hard = (r > eps).to(r.dtype) if g_hard is None else g_hard.to(r.dtype)
        g = hard + (soft - soft.detach())

    hupd = h_free + dBbar * B_l.unsqueeze(1) * u_l.unsqueeze(-1)
    gs = g.view(-1, 1, 1); gv = g.view(-1, 1)
    g_upd = gs * hupd                               # hcur 与 h_pi_n 共有，只算一次
    one_g = 1 - gs
    hcur = g_upd + one_g * xhat
    h_pi_n = g_upd + one_g * h_pi
    t_pi_n = g * t_l + (1 - g) * t_pi
    u_pi_n = gv * u_l + (1 - gv) * u_pi
    B_pi_n = gv * B_l + (1 - gv) * B_pi
    C_pi_n = gv * C_l + (1 - gv) * C_pi
    y = (C_l.unsqueeze(1) * hcur).real.sum(-1) + D * u_l
    return (h_pi_n, t_pi_n, u_pi_n, B_pi_n, C_pi_n), y, g, r


def _init_state(u, Bc, Cc, t, dt_init):
    B, L, H = u.shape
    N = Bc.shape[-1]
    h_pi = torch.zeros(B, H, N, dtype=Bc.dtype, device=u.device)
    return (h_pi, t[:, 0] - dt_init, u[:, 0], Bc[:, 0], Cc[:, 0])


def _or_zeros(x, ref):
    """梯度缺失（allow_unused 返回 None）时补一份**零**，形状同 ref。

    注意不能写成 `ref if x is None else x`：调用点传的 ref 往往就是累加器本身
    （dlam/dlog/dD/deps 跨 L 步累加），None 时会变成 `dlam += dlam` 直接把已累积的
    梯度翻倍。目前 lam 每步都被用到所以触发不了，但这是不会报错的定时炸弹。
    """
    return torch.zeros_like(ref) if x is None else x


class GatedEACSCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                T, eta, var_eps, dt_init, gate_mode, use_kernel, dt_max):
        B, L, H = u.shape
        # 前向：无梯度顺序扫描（可用 CUDA 融合前向内核）
        if use_kernel and u.is_cuda and gate_mode == "ste":
            try:
                from .eacs_gated_scan import gated_kernel_available, eacs_cell_forward
                if gated_kernel_available():
                    ys, gs, rs = eacs_cell_forward(
                        lam, log_dt, u, Bc, Cc, t, D, mean, var,
                        eps=float(eps), eta=eta, var_eps=var_eps, dt_init=dt_init,
                        dt_max=dt_max)
                    _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                              T, eta, var_eps, dt_init, gate_mode, L, dt_max, gs)
                    return ys, gs, rs
            except Exception:
                pass
        with torch.no_grad():
            state = _init_state(u, Bc, Cc, t, dt_init)
            ys, gs, rs = [], [], []
            inv_lam = torch.reciprocal(lam)      # λ 扫描期间不变，倒数只算一次
            for l in range(L):
                state, y, g, r = _step_functional(
                    state, (u[:, l], Bc[:, l], Cc[:, l], t[:, l]),
                    (lam, log_dt, D, eps), T, eta, var_eps, mean, var, gate_mode,
                    dt_max=dt_max, inv_lam=inv_lam)
                ys.append(y); gs.append(g); rs.append(r)
            ys = torch.stack(ys, 1); gs = torch.stack(gs, 1); rs = torch.stack(rs, 1)
        _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                  T, eta, var_eps, dt_init, gate_mode, L, dt_max, gs)
        return ys, gs, rs

    @staticmethod
    def backward(ctx, g_ys, g_gs, g_rs):
        u, Bc, Cc, t, lam, log_dt, D, eps, mean, var, gs_fwd = ctx.saved_tensors
        T, eta, var_eps, dt_init, gate_mode, L = ctx.T, ctx.eta, ctx.var_eps, ctx.dt_init, ctx.gate_mode, ctx.L
        dt_max = ctx.dt_max
        B, _, H = u.shape
        # 复用前向真实产生的硬门控：前向可能走的是 CUDA 内核，若让重算自行判 r>eps，
        # 1 ulp 的差异就会让轨迹分叉、梯度对不上前向输出（且不报错）。soft 模式无此问题。
        def _g(l):
            return None if gate_mode == "soft" else gs_fwd[:, l]

        # 重算前向状态轨迹（每步"上一状态"），无梯度
        with torch.no_grad():
            state = _init_state(u, Bc, Cc, t, dt_init)
            prev = []
            inv_lam = torch.reciprocal(lam)      # 同上；VJP 循环里不能这么做（见 _step_functional）
            for l in range(L):
                prev.append(state)
                state, _, _, _ = _step_functional(
                    state, (u[:, l], Bc[:, l], Cc[:, l], t[:, l]),
                    (lam, log_dt, D, eps), T, eta, var_eps, mean, var, gate_mode,
                    dt_max=dt_max, g_hard=_g(l), inv_lam=inv_lam)

        du = torch.zeros_like(u); dBc = torch.zeros_like(Bc); dCc = torch.zeros_like(Cc); dt = torch.zeros_like(t)
        dlam = torch.zeros_like(lam); dlog = torch.zeros_like(log_dt); dD = torch.zeros_like(D); deps = torch.zeros_like(eps)
        # 状态伴随（初始 0：末状态未被使用）
        ah = torch.zeros_like(prev[0][0]); at = torch.zeros_like(prev[0][1])
        au = torch.zeros_like(prev[0][2]); aB = torch.zeros_like(prev[0][3]); aC = torch.zeros_like(prev[0][4])

        for l in reversed(range(L)):
            ph = prev[l][0].detach().requires_grad_(True)
            pt = prev[l][1].detach().requires_grad_(True)
            pu = prev[l][2].detach().requires_grad_(True)
            pB = prev[l][3].detach().requires_grad_(True)
            pC = prev[l][4].detach().requires_grad_(True)
            ul = u[:, l].detach().requires_grad_(True); Bl = Bc[:, l].detach().requires_grad_(True)
            Cl = Cc[:, l].detach().requires_grad_(True); tl = t[:, l].detach().requires_grad_(True)
            lam_ = lam.detach().requires_grad_(True); log_ = log_dt.detach().requires_grad_(True)
            D_ = D.detach().requires_grad_(True); eps_ = eps.detach().requires_grad_(True)
            with torch.enable_grad():
                (nh, nt, nu, nB, nC), y, g, r = _step_functional(
                    (ph, pt, pu, pB, pC), (ul, Bl, Cl, tl), (lam_, log_, D_, eps_),
                    T, eta, var_eps, mean, var, gate_mode,
                    dt_max=dt_max, g_hard=_g(l))
            outs = [nh, nt, nu, nB, nC, y, g, r]
            gouts = [ah, at, au, aB, aC, g_ys[:, l],
                     g_gs[:, l] if g_gs is not None else torch.zeros_like(g),
                     g_rs[:, l] if g_rs is not None else torch.zeros_like(r)]
            grads = torch.autograd.grad(outs, [ph, pt, pu, pB, pC, ul, Bl, Cl, tl, lam_, log_, D_, eps_],
                                        gouts, allow_unused=True, retain_graph=False)
            ah = _or_zeros(grads[0], ah); at = _or_zeros(grads[1], at)
            au = _or_zeros(grads[2], au); aB = _or_zeros(grads[3], aB); aC = _or_zeros(grads[4], aC)
            du[:, l] += _or_zeros(grads[5], du[:, l]); dBc[:, l] += _or_zeros(grads[6], dBc[:, l])
            dCc[:, l] += _or_zeros(grads[7], dCc[:, l]); dt[:, l] += _or_zeros(grads[8], dt[:, l])
            dlam += _or_zeros(grads[9], dlam); dlog += _or_zeros(grads[10], dlog)
            dD += _or_zeros(grads[11], dD); deps += _or_zeros(grads[12], deps)

        # 初始状态伴随回流到 inputs[:,0]（h_pi 初值=0 无梯度）
        dt[:, 0] += at; du[:, 0] += au; dBc[:, 0] += aB; dCc[:, 0] += aC
        # forward 有 17 个输入，返回值个数必须与之一一对应
        return (du, dBc, dCc, dt, dlam, dlog, dD, deps, None, None,
                None, None, None, None, None, None, None)


# 保存反向所需（张量走 save_for_backward，标量挂 ctx 属性）
def _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
              T, eta, var_eps, dt_init, gate_mode, L, dt_max, gs):
    # gs（前向真实门控）也存下来：反向重算必须复用它，否则硬门控的不连续会让轨迹分叉
    ctx.save_for_backward(u, Bc, Cc, t, lam, log_dt, D, eps, mean, var, gs)
    ctx.T, ctx.eta, ctx.var_eps, ctx.dt_init, ctx.gate_mode, ctx.L = T, eta, var_eps, dt_init, gate_mode, L
    ctx.dt_max = dt_max


def fused_gated_scan(u: Tensor, Bc: Tensor, Cc: Tensor, t: Tensor,
                     lam: Tensor, log_dt: Tensor, D: Tensor, eps: Tensor,
                     mean: Tensor, var: Tensor, T: float, eta: float,
                     var_eps: float = 1e-5, dt_init: float = 1.0,
                     gate_mode: str = "ste", use_kernel: bool = True,
                     dt_max: float = 1e3):
    """可微融合门控扫描。返回 (ys[B,L,H], gates[B,L], resid[B,L])。"""
    return GatedEACSCellFunction.apply(u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                                       T, eta, var_eps, dt_init, gate_mode, use_kernel,
                                       dt_max)
