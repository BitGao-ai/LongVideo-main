"""创新点1 CPIB-Distill 训练损失：条件 InfoNCE + 信息瓶颈 KL + 反事实一致性。

设计文档（CHST_创新点细化.pdf §1）总损失：
    L = L_task + λ1·L_NCE + λ2·R_KL + λ3·L_cf

各分项：
  - L_NCE（条件 InfoNCE）：让 CGU 保留的表征 z_t 能预测未来 τ 窗口表征 z_{t+1:t+τ}，
    同时远离 batch 内负样本。双线性 critic f(z,z')=z^T W z'。
  - R_KL（信息瓶颈率预算）：KL(Bern(s_{t,i}) ‖ Bern(ρ))，约束全局压缩率。
    ρ 为保留预算超参（如 0.3 = 保留 30% 信息）。
  - L_cf（反事实一致性）：MC 随机消融 K 次，度量边际效应 Δ_{t,i}，
    约束 CGU 分数 s 与因果效应一致：L_cf = (1/K)Σ(s - sg[norm(Δ)])²。
    仅前若干 epoch 启用（退火关闭），避免后期过约束。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ============================ 条件 InfoNCE ============================

class BilinearCritic(nn.Module):
    """双线性打分函数 f(z, z') = z^T W z'（对称可选）。"""

    def __init__(self, dim: int, symmetric: bool = False):
        super().__init__()
        self.W = nn.Parameter(torch.randn(dim, dim) * (dim ** -0.5))
        self.symmetric = symmetric

    @property
    def effective_W(self) -> Tensor:
        """实际参与打分的矩阵（symmetric 时为 W+Wᵀ）。批量打分请直接用它做 matmul。"""
        return self.W + self.W.T if self.symmetric else self.W

    def forward(self, z: Tensor, zp: Tensor) -> Tensor:
        """z: (…, d), zp: (…, d) → scores: (…,)"""
        return torch.einsum("...d,de,...e->...", z, self.effective_W, zp)


def conditional_infonce_loss(z: Tensor, z_future: Tensor, critic: BilinearCritic,
                             temperature: float = 0.1,
                             frame_mask: Tensor | None = None) -> Tensor:
    """条件 InfoNCE：z_t 预测未来帧表征（正样本），batch 内其余为负样本。

    参数：
        z         : (B, L, d)  当前帧表征（CGU 蒸馏后的帧级特征）
        z_future  : (B, L, d)  **已对齐的未来表征**：z_future[t] = 帧 t+1 的特征（detach）。
                    调用方（losses._compute_cpib）已完成 shift，本函数内部不得再次偏移，
                    否则正样本会错成 t+2（双重 shift 破坏预测语义）。
        critic    : 双线性打分
        temperature: softmax 温度
        frame_mask: (B, L) bool，True=有效帧
    返回：
        标量损失
    """
    B, L, d = z.shape
    if L < 2:
        return z.sum() * 0.0

    # 正样本对：(z_t, z_future[t])，取 t ∈ [0, L-2]；z_future[t] 已是帧 t+1 的特征
    z_t = z[:, :-1].reshape(-1, d)              # (B*(L-1), d)
    z_pos = z_future[:, :-1].reshape(-1, d).detach()  # (B*(L-1), d)

    # 有效 mask
    if frame_mask is not None:
        valid = (frame_mask[:, :-1] & frame_mask[:, 1:]).reshape(-1)  # (B*(L-1),)
        if not valid.any():
            return z.sum() * 0.0
        z_t = z_t[valid]
        z_pos = z_pos[valid]

    N = z_t.shape[0]
    if N < 2:
        return z.sum() * 0.0

    # 打分矩阵：(N, N)，对角线为正样本。
    # 用两次 matmul 而不是 einsum("...d,de,...e->...", z.unsqueeze(1), W, zp.unsqueeze(0))：
    # 后者靠广播凑出 (N,N)，torch 虽然没有物化 (N,N,d)，但调度开销大得多
    # （实测 N=1024/d=384 时 0.040s vs 0.003s）。数值完全一致。
    logits = (z_t @ critic.effective_W) @ z_pos.t() / temperature        # (N, N)
    labels = torch.arange(N, device=z.device)
    loss = F.cross_entropy(logits, labels)
    return loss


# ============================ 信息瓶颈 KL ============================

def information_bottleneck_kl(scores: Tensor, rho: float = 0.3,
                              frame_mask: Tensor | None = None) -> Tensor:
    """率预算正则：R_KL = Σ_{t,i} KL(Bern(s_{t,i}) ‖ Bern(ρ))。

    参数：
        scores     : (B, L, P)  CGU 输出的逐 token 保留分数
        rho        : 全局压缩预算（目标保留率）
        frame_mask : (B, L) 有效帧
    返回：
        标量（每 token 平均 KL）
    """
    eps = 1e-7
    s = scores.clamp(eps, 1 - eps)
    # KL(Bern(s) ‖ Bern(ρ)) = s·log(s/ρ) + (1-s)·log((1-s)/(1-ρ))
    kl = s * torch.log(s / rho) + (1 - s) * torch.log((1 - s) / (1 - rho))  # (B,L,P)

    if frame_mask is not None:
        # 只对有效帧计算
        mask = frame_mask.unsqueeze(-1).expand_as(kl)   # (B,L,P)
        kl = kl * mask
        n = mask.sum().clamp_min(1)
    else:
        n = kl.numel()
    return kl.sum() / n


# ============================ 反事实一致性 ============================

def counterfactual_consistency_loss(
    scores: Tensor,
    frame_feat: Tensor,
    tokens: Tensor,
    distiller_forward,
    K: int = 2,
    frame_mask: Tensor | None = None,
    ablate_frac: float = 0.1,
) -> Tensor:
    """反事实一致性损失：约束 CGU 分数与因果边际效应一致。

    方法（设计 §1.4）：
      1. 对每帧随机消融 n_ab = max(1, round(ablate_frac·P)) 个 token（设 score=0），重新蒸馏
      2. 边际效应 Δ_t = ‖frame_feat − frame_feat_cf‖（被消融那批 token 的整体重要性）
      3. L_cf = MSE(mean(s_被消融), sg[Δ_t / ‖frame_feat_t‖])

    **归一化按帧、不按时间**：目标必须只依赖该帧自身。沿时间维归一化会让帧 t 的目标
    取决于整段里 Δ 最大的那一帧，同一帧在不同 batch 组合下目标值都不同（详见实现处注释）。

    **一次消融一批而不是一个**：TokenDistiller 的保留分支是 P 个 token 的加权均值，
    单个 token 的边际影响只有约 1/P——P=9 时 Δ 已经很小，P=64 时基本是数值噪声，
    回归的目标全是噪声，这一项等于没在学。按比例消融把 Δ 放大到可用量级。

    参数：
        scores          : (B, L, P) CGU 分数
        frame_feat      : (B, L, d) 正常蒸馏输出
        tokens          : (B, L, P, d) 原始 token
        distiller_forward: callable(tokens, scores) → (frame_feat, cpi)
        K               : MC 采样次数
        frame_mask      : (B, L)
        ablate_frac     : 每帧消融的 token 比例
    返回：
        标量损失
    """
    B, L, P, d = tokens.shape
    if P < 2:
        return scores.sum() * 0.0
    n_ab = max(1, min(P - 1, int(round(ablate_frac * P))))   # 至少 1、至多 P-1

    total_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
    for _ in range(K):
        # 每帧无放回地随机选 n_ab 个 token（argsort 随机键 = 随机排列取前 n_ab 个）
        ablate_idx = torch.rand(B, L, P, device=scores.device).argsort(dim=-1)[..., :n_ab]
        scores_cf = scores.clone()
        scores_cf.scatter_(2, ablate_idx, 0.0)

        # 重新蒸馏
        with torch.no_grad():
            feat_cf, _ = distiller_forward(tokens, scores_cf)

        # 边际效应：frame_feat 与 feat_cf 的差异（逐帧）
        delta = torch.linalg.vector_norm(
            frame_feat.detach() - feat_cf, dim=-1)        # (B, L)

        # 被消融那批 token 的平均分数应高（因果重要）
        s_ablated = scores.gather(2, ablate_idx).mean(dim=-1)               # (B, L)

        # 归一化 Δ → [0,1]：除以**该帧自身**的特征范数，得到"相对变化量"。
        # 早先是 delta / delta.amax(dim=1)，沿**时间**维归一化——目标 s_ablated 是逐帧
        # 逐 token 分数的均值（每帧独立的 [0,1] 量），而沿时间归一化让帧 t 的目标取决于
        # 整段里 Δ 最大的那一帧：同一帧在不同 batch 组合下目标值不同，且只有全局最大的
        # 那帧目标为 1、其余被压小，净效果就是把 s 拽向一个常数。改成逐帧的相对变化后，
        # 目标只依赖该帧自身，与 s_ablated 的语义维度一致。
        denom = torch.linalg.vector_norm(frame_feat.detach(), dim=-1)       # (B, L)
        delta_norm = (delta / denom.clamp_min(1e-8)).clamp(0.0, 1.0)

        # MSE
        diff = (s_ablated - delta_norm.detach()) ** 2     # (B, L)
        if frame_mask is not None:
            diff = diff * frame_mask
            n = frame_mask.sum().clamp_min(1)
        else:
            n = diff.numel()
        total_loss = total_loss + diff.sum() / n

    return total_loss / K


# ============================ 组合：CPIB 损失 ============================

@dataclass
class CPIBWeights:
    """创新点1 各损失项权重。"""
    nce: float = 0.1          # λ1：InfoNCE
    kl: float = 0.01          # λ2：信息瓶颈率预算
    cf: float = 0.05          # λ3：反事实一致性
    cf_ablate_frac: float = 0.1   # 反事实每帧消融的 token 比例（太小则 Δ 淹没在噪声里）
    rho: float = 0.3          # 信息瓶颈保留预算
    cf_anneal_steps: int = 2000   # L_cf 退火步数（之后权重→0）


def cpib_loss(
    scores: Tensor,
    frame_feat: Tensor,
    tokens: Tensor,
    z_future: Tensor,
    distiller_forward,
    w: CPIBWeights,
    step: int = 0,
    frame_mask: Tensor | None = None,
    critic: BilinearCritic | None = None,
) -> tuple[Tensor, dict]:
    """计算创新点1 完整附加损失。

    返回 (total_loss, components_dict)。
    """
    comp = {}
    total = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

    # L_NCE
    if critic is not None and w.nce > 0:
        l_nce = conditional_infonce_loss(frame_feat, z_future, critic,
                                         frame_mask=frame_mask)
        total = total + w.nce * l_nce
        comp["nce"] = l_nce.detach()

    # R_KL
    if w.kl > 0:
        r_kl = information_bottleneck_kl(scores, rho=w.rho, frame_mask=frame_mask)
        total = total + w.kl * r_kl
        comp["ib_kl"] = r_kl.detach()

    # L_cf（带退火）
    cf_weight = w.cf * max(0.0, 1.0 - step / max(w.cf_anneal_steps, 1))
    if cf_weight > 1e-8:
        l_cf = counterfactual_consistency_loss(
            scores, frame_feat, tokens, distiller_forward,
            K=2, frame_mask=frame_mask, ablate_frac=w.cf_ablate_frac)
        total = total + cf_weight * l_cf
        comp["cf"] = l_cf.detach()

    comp["cpib_total"] = total.detach()
    return total, comp
