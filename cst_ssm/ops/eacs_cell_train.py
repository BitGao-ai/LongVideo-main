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
                     gate_mode: str = "ste"):
    """单步 segment cell（与 EACSLayer._step 同构）。state/inputs/params 为张量元组。"""
    h_pi, t_pi, u_pi, B_pi, C_pi = state
    u_l, B_l, C_l, t_l = inputs
    lam, log_dt, D, eps = params

    delta = (t_l - t_pi).clamp_min(0.0)
    dt_eff = (delta.unsqueeze(-1) * torch.exp(log_dt)).clamp(1e-3, 1e3)     # (B,H)
    z = lam * dt_eff.unsqueeze(-1)                                          # (B,H,N)
    dA = torch.exp(z)
    dBbar = complex_expm1(z) / lam
    xhat = dA * h_pi + dBbar * B_pi.unsqueeze(1) * u_pi.unsqueeze(-1)
    yhat = (C_pi.unsqueeze(1) * xhat).real.sum(-1)                          # (B,H)

    s = torch.sqrt(var + var_eps)
    diff = (u_l - yhat) / s
    obs = (u_l - mean) / s
    r = diff.norm(dim=-1) / (obs.norm(dim=-1) + eta)                        # (B,)
    soft = torch.sigmoid((r - eps) / max(T, 1e-4))
    if gate_mode == "soft":
        g = soft
    else:                                                                  # STE：前向硬、反向软
        g = (r > eps).to(r.dtype) + (soft - soft.detach())

    hupd = dA * h_pi + dBbar * B_l.unsqueeze(1) * u_l.unsqueeze(-1)
    gs = g.view(-1, 1, 1); gv = g.view(-1, 1)
    hcur = gs * hupd + (1 - gs) * xhat
    h_pi_n = gs * hupd + (1 - gs) * h_pi
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


def _zeros_like(x, ref):
    return ref if x is None else x


class GatedEACSCellFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                T, eta, var_eps, dt_init, gate_mode, use_kernel):
        B, L, H = u.shape
        # 前向：无梯度顺序扫描（可用 CUDA 融合前向内核）
        if use_kernel and u.is_cuda and gate_mode == "ste":
            try:
                from .eacs_gated_scan import gated_kernel_available, eacs_cell_forward
                if gated_kernel_available():
                    ys, gs, rs = eacs_cell_forward(
                        lam, log_dt, u, Bc, Cc, t, D, mean, var,
                        eps=float(eps), eta=eta, var_eps=var_eps, dt_init=dt_init)
                    _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var, T, eta, var_eps, dt_init, gate_mode, L)
                    return ys, gs, rs
            except Exception:
                pass
        with torch.no_grad():
            state = _init_state(u, Bc, Cc, t, dt_init)
            ys, gs, rs = [], [], []
            for l in range(L):
                state, y, g, r = _step_functional(
                    state, (u[:, l], Bc[:, l], Cc[:, l], t[:, l]),
                    (lam, log_dt, D, eps), T, eta, var_eps, mean, var, gate_mode)
                ys.append(y); gs.append(g); rs.append(r)
            ys = torch.stack(ys, 1); gs = torch.stack(gs, 1); rs = torch.stack(rs, 1)
        _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var, T, eta, var_eps, dt_init, gate_mode, L)
        return ys, gs, rs

    @staticmethod
    def backward(ctx, g_ys, g_gs, g_rs):
        u, Bc, Cc, t, lam, log_dt, D, eps, mean, var = ctx.saved_tensors
        T, eta, var_eps, dt_init, gate_mode, L = ctx.T, ctx.eta, ctx.var_eps, ctx.dt_init, ctx.gate_mode, ctx.L
        B, _, H = u.shape

        # 重算前向状态轨迹（每步"上一状态"），无梯度
        with torch.no_grad():
            state = _init_state(u, Bc, Cc, t, dt_init)
            prev = []
            for l in range(L):
                prev.append(state)
                state, _, _, _ = _step_functional(
                    state, (u[:, l], Bc[:, l], Cc[:, l], t[:, l]),
                    (lam, log_dt, D, eps), T, eta, var_eps, mean, var, gate_mode)

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
                    T, eta, var_eps, mean, var, gate_mode)
            outs = [nh, nt, nu, nB, nC, y, g, r]
            gouts = [ah, at, au, aB, aC, g_ys[:, l],
                     g_gs[:, l] if g_gs is not None else torch.zeros_like(g),
                     g_rs[:, l] if g_rs is not None else torch.zeros_like(r)]
            grads = torch.autograd.grad(outs, [ph, pt, pu, pB, pC, ul, Bl, Cl, tl, lam_, log_, D_, eps_],
                                        gouts, allow_unused=True, retain_graph=False)
            ah = _zeros_like(grads[0], ah * 0); at = _zeros_like(grads[1], at * 0)
            au = _zeros_like(grads[2], au * 0); aB = _zeros_like(grads[3], aB * 0); aC = _zeros_like(grads[4], aC * 0)
            du[:, l] += _zeros_like(grads[5], du[:, l]); dBc[:, l] += _zeros_like(grads[6], dBc[:, l])
            dCc[:, l] += _zeros_like(grads[7], dCc[:, l]); dt[:, l] += _zeros_like(grads[8], dt[:, l])
            dlam += _zeros_like(grads[9], dlam); dlog += _zeros_like(grads[10], dlog)
            dD += _zeros_like(grads[11], dD); deps += _zeros_like(grads[12], deps)

        # 初始状态伴随回流到 inputs[:,0]（h_pi 初值=0 无梯度）
        dt[:, 0] += at; du[:, 0] += au; dBc[:, 0] += aB; dCc[:, 0] += aC
        return (du, dBc, dCc, dt, dlam, dlog, dD, deps, None, None,
                None, None, None, None, None, None)


# 保存反向所需（张量走 save_for_backward，标量挂 ctx 属性）
def _ctx_save(ctx, u, Bc, Cc, t, lam, log_dt, D, eps, mean, var, T, eta, var_eps, dt_init, gate_mode, L):
    ctx.save_for_backward(u, Bc, Cc, t, lam, log_dt, D, eps, mean, var)
    ctx.T, ctx.eta, ctx.var_eps, ctx.dt_init, ctx.gate_mode, ctx.L = T, eta, var_eps, dt_init, gate_mode, L


def fused_gated_scan(u: Tensor, Bc: Tensor, Cc: Tensor, t: Tensor,
                     lam: Tensor, log_dt: Tensor, D: Tensor, eps: Tensor,
                     mean: Tensor, var: Tensor, T: float, eta: float,
                     var_eps: float = 1e-5, dt_init: float = 1.0,
                     gate_mode: str = "ste", use_kernel: bool = True):
    """可微融合门控扫描。返回 (ys[B,L,H], gates[B,L], resid[B,L])。"""
    return GatedEACSCellFunction.apply(u, Bc, Cc, t, lam, log_dt, D, eps, mean, var,
                                       T, eta, var_eps, dt_init, gate_mode, use_kernel)
