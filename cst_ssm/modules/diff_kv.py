"""创新点3：误差有界差分 KV 缓存 + CPI 稀疏注意力（设计文档 §3）。

核心思想：
  - 长视频推理时 KV 缓存线性增长是显存瓶颈。
  - 每隔 G 帧存完整 K,V（基准帧）；中间帧只存低秩残差 K̂_t = K_base + U_t·S_t^T（秩 r≪d）。
  - 引理1（误差界）：‖Attn(Q,K̂,V̂) - Attn(Q,K,V)‖ ≤ C·δ，δ=残差重构误差。
  - 当残差能量超阈值 → 自动插入新基准帧（自适应 G）。
  - CPI 稀疏注意力：高 CPI token 走全注意力，低 CPI 用 SSM 摘要替代（性质3）。

本模块为推理侧效率组件，训练时可选启用（蒸馏预测头），推理时直接使用。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DiffKVConfig:
    rank: int = 8                 # 低秩残差秩 r
    base_interval: int = 8        # 初始基准帧间隔 G
    error_threshold: float = 0.1  # 残差能量阈值 ε（超过则插入新基准帧）
    cpi_sparse: bool = True       # 是否启用 CPI 稀疏注意力
    cpi_keep_ratio: float = 0.3   # 高 CPI token 保留比例（走全注意力）


class LowRankResidualHead(nn.Module):
    """低秩残差预测头：从帧差特征生成 (U_t, S_t, V_t)，使 ΔK ≈ U_t·diag(S_t)·V_t^T·1。

    设计文档 §3 要求：K̂_t = K_base + U_t·diag(S_t)·V_t^T（秩 r≪d 的截断 SVD 形式）。
    输入：frame_diff (B, d) = feat_t - feat_base（当前帧与基准帧的特征差）
    输出：U (B, d, r), S (B, r), V (B, d, r)  低秩双因子
    重构：ΔK = (U * S) @ V^T → (B, d, d) 低秩矩阵；取对角或乘以查询向量得 (B, d) 残差
    参数量：3·d·r + r（极轻量）
    """

    def __init__(self, d: int, rank: int = 8):
        super().__init__()
        self.rank = rank
        self.d = d
        # 帧差 → 低秩双因子 + 奇异值
        self.proj_U = nn.Linear(d, d * rank, bias=False)
        self.proj_V = nn.Linear(d, d * rank, bias=False)
        self.proj_S = nn.Linear(d, rank)
        # 初始化小权重使初始残差接近零
        nn.init.normal_(self.proj_U.weight, std=0.01)
        nn.init.normal_(self.proj_V.weight, std=0.01)
        nn.init.zeros_(self.proj_S.bias)

    def forward(self, frame_diff: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """frame_diff: (B, d) → U: (B, d, r), S: (B, r), V: (B, d, r)"""
        B = frame_diff.shape[0]
        U = self.proj_U(frame_diff).view(B, self.d, self.rank)   # (B, d, r)
        V = self.proj_V(frame_diff).view(B, self.d, self.rank)   # (B, d, r)
        S = self.proj_S(frame_diff)                              # (B, r)
        return U, S, V

    def reconstruct(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        """ΔK = U·diag(S)·V^T → (B, d, d)（秩 r 低秩矩阵）。

        高效实现：不显式构造 d×d，用 (B,d,r) × (B,r) × (B,r,d) 形式存储。
        """
        return (U * S.unsqueeze(1)) @ V.transpose(1, 2)   # (B, d, d)

    def reconstruct_vec(self, U: Tensor, S: Tensor, V: Tensor) -> Tensor:
        """向量级重构：Δk = (U·diag(S)·V^T)·1/√d → (B, d)。

        用于 KV 缓存场景：每帧 K/V 为 (B,d) 向量，低秩残差沿所有方向施加。
        等价于 V 各列加权求和后再经 U·S 线性组合。
        """
        # (U*S): (B,d,r); V: (B,d,r) → sum over r of (U*S)_{:,i} * V_{:,i} 的外积取对角
        # 高效：Δk_i = Σ_r U_{i,r} * S_r * (Σ_j V_{j,r}) / √d
        V_sum = V.sum(dim=1) / (self.d ** 0.5)   # (B, r)
        return (U * S.unsqueeze(1) * V_sum.unsqueeze(1)).sum(dim=-1)  # (B, d)


class DifferentialKVCache(nn.Module):
    """差分 KV 缓存管理器：基准帧 + 低秩残差 + 自适应基准帧插入。

    用法（推理时）：
        cache = DifferentialKVCache(cfg, d_model)
        for t in range(L):
            k_t, v_t = compute_kv(feat_t)
            k_cached = cache.update(k_t, v_t, feat_t, t)
            # k_cached: 重构的完整 K（用于注意力计算）

    训练时：预测头学习从帧差重构 K 残差（MSE 损失）。
    """

    def __init__(self, cfg: DiffKVConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d = d_model
        self.head_K = LowRankResidualHead(d_model, cfg.rank)
        self.head_V = LowRankResidualHead(d_model, cfg.rank)
        # 运行时状态（推理时填充）
        self.reset()

    def reset(self):
        """清空缓存状态。"""
        self._base_K: Tensor | None = None
        self._base_V: Tensor | None = None
        self._base_feat: Tensor | None = None
        self._base_idx: int = 0
        self._residuals_K: list[tuple[Tensor, Tensor]] = []  # [(U, S), ...]
        self._residuals_V: list[tuple[Tensor, Tensor]] = []
        self._n_base_frames: int = 0

    def should_insert_base(self, feat_t: Tensor) -> bool:
        """判断是否应插入新基准帧（残差能量超阈值）。"""
        if self._base_feat is None:
            return True
        diff = feat_t - self._base_feat
        energy = torch.linalg.vector_norm(diff, dim=-1).mean()
        return energy.item() > self.cfg.error_threshold

    def update(self, K_t: Tensor, V_t: Tensor, feat_t: Tensor,
               t: int) -> tuple[Tensor, Tensor]:
        """更新缓存并返回重构的 (K_full, V_full)。

        K_t, V_t: (B, d) 当前帧的完整 K/V
        feat_t: (B, d) 当前帧特征（用于判断基准帧插入）
        返回重构的 K, V（基准 + 低秩残差 U·diag(S)·V^T 向量级重构）
        """
        if self.should_insert_base(feat_t):
            # 插入新基准帧
            self._base_K = K_t.detach().clone()
            self._base_V = V_t.detach().clone()
            self._base_feat = feat_t.detach().clone()
            self._base_idx = t
            self._residuals_K.clear()
            self._residuals_V.clear()
            self._n_base_frames += 1
            return K_t, V_t

        # 中间帧：计算低秩残差（U·diag(S)·V^T 双因子）
        diff_K = K_t - self._base_K
        diff_V = V_t - self._base_V
        U_K, S_K, Vf_K = self.head_K(diff_K)
        U_V, S_V, Vf_V = self.head_V(diff_V)
        self._residuals_K.append((U_K.detach(), S_K.detach(), Vf_K.detach()))
        self._residuals_V.append((U_V.detach(), S_V.detach(), Vf_V.detach()))

        # 向量级重构：ΔK = (U·diag(S)·V^T)·1/√d
        K_recon = self._base_K + self.head_K.reconstruct_vec(U_K, S_K, Vf_K)
        V_recon = self._base_V + self.head_V.reconstruct_vec(U_V, S_V, Vf_V)
        return K_recon, V_recon

    @property
    def compression_ratio(self) -> float:
        """当前压缩率：存储量 / 全量存储。"""
        if self._n_base_frames == 0:
            return 1.0
        total_frames = self._base_idx + len(self._residuals_K) + 1
        # 基准帧存 d 维，残差帧存 2·d·r 维（U+S）
        full_storage = total_frames * self.d
        actual_storage = (self._n_base_frames * self.d +
                          len(self._residuals_K) * 2 * self.d * self.cfg.rank)
        return actual_storage / max(full_storage, 1)

    def reconstruction_loss(self, K_true: Tensor, V_true: Tensor,
                            feat_t: Tensor) -> Tensor:
        """训练用：预测头重构损失 MSE(K_true, K_base + ΔK)，ΔK = U·diag(S)·V^T 向量级。"""
        if self._base_K is None:
            self._base_K = K_true.detach().clone()
            self._base_V = V_true.detach().clone()
            self._base_feat = feat_t.detach().clone()
            return torch.tensor(0.0, device=K_true.device)

        diff_K = K_true - self._base_K
        diff_V = V_true - self._base_V
        U_K, S_K, Vf_K = self.head_K(diff_K)
        U_V, S_V, Vf_V = self.head_V(diff_V)

        # 向量级重构：ΔK ≈ (U·diag(S)·V^T)·1/√d
        recon_K = self.head_K.reconstruct_vec(U_K, S_K, Vf_K)   # (B, d)
        recon_V = self.head_V.reconstruct_vec(U_V, S_V, Vf_V)   # (B, d)

        loss = F.mse_loss(recon_K, diff_K) + F.mse_loss(recon_V, diff_V)
        return loss


class CPISparseAttention(nn.Module):
    """因果 CPI 稀疏注意力：高 CPI token 走因果全注意力，低 CPI 用摘要替代。

    设计§3 性质3：摘要替代误差 ≤ 簇内方差。
    实现：按 CPI 分数排序，top-k 走因果注意力（下三角 mask），其余用加权均值摘要代替。
    训练与推理均启用 CPI 稀疏（推理时更需要效率）。

    输入：Q (B, N, d), K (B, N, d), V (B, N, d), cpi (B, N)
    输出：attn_out (B, N, d)
    """

    def __init__(self, d: int, n_heads: int = 8, keep_ratio: float = 0.3,
                 causal: bool = True):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal    # 因果 mask（CHST 架构要求）
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def _causal_mask(self, N: int, device) -> Tensor:
        """生成因果下三角 mask：位置 i 只能 attend to j≤i。"""
        return torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: Tensor, cpi: Tensor | None = None) -> Tensor:
        """x: (B, N, d), cpi: (B, N) 可选。训练和推理均支持 CPI 稀疏。"""
        B, N, d = x.shape
        Q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        if cpi is None:
            # 无 CPI 信号：全注意力（+ 因果 mask）
            attn = (Q @ K.transpose(-2, -1)) * self.scale
            if self.causal:
                attn = attn.masked_fill(self._causal_mask(N, x.device), float("-inf"))
            attn = attn.softmax(-1)
            out = (attn @ V).transpose(1, 2).reshape(B, N, d)
            return self.out_proj(out)

        # CPI 稀疏：top-k 因果注意力 + 其余摘要（训练和推理均启用）
        k = max(1, int(N * self.keep_ratio))
        _, topk_idx = cpi.topk(k, dim=-1)                # (B, k)

        # 高 CPI 子集的投影
        topk_idx_exp = topk_idx.unsqueeze(-1).expand(B, k, d)
        Q_top = self.q_proj(x).gather(1, topk_idx_exp).view(B, k, self.n_heads, self.head_dim).transpose(1, 2)
        K_top = self.k_proj(x).gather(1, topk_idx_exp).view(B, k, self.n_heads, self.head_dim).transpose(1, 2)
        V_top = self.v_proj(x).gather(1, topk_idx_exp).view(B, k, self.n_heads, self.head_dim).transpose(1, 2)

        # 低 CPI 摘要（加权均值）
        mask_low = torch.ones(B, N, device=x.device, dtype=torch.bool)
        mask_low.scatter_(1, topk_idx, False)
        cpi_low = cpi * mask_low.float()
        w_low = cpi_low / (cpi_low.sum(dim=-1, keepdim=True) + 1e-8)  # (B, N)
        V_summary = (x * w_low.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B, 1, d)
        V_summary = V_summary.expand(B, k, d)

        # 因果注意力：高CPI queries attend to 高CPI keys（+ 因果 mask）+ 摘要
        attn = (Q_top @ K_top.transpose(-2, -1)) * self.scale
        if self.causal:
            # 对 top-k 子集施加因果约束：按原始位置顺序，只 attend 到位置 ≤ 自己的 key
            # topk_idx: (B, k)，构造 (B, k, k) 因果 mask
            pos_q = topk_idx.unsqueeze(2)   # (B, k, 1)
            pos_k = topk_idx.unsqueeze(1)   # (B, 1, k)
            causal = pos_k > pos_q           # key 位置 > query 位置 → 遮蔽
            attn = attn.masked_fill(causal.unsqueeze(1), float("-inf"))  # (B, heads, k, k)
        attn = attn.softmax(-1)
        out_top = (attn @ V_top).transpose(1, 2).reshape(B, k, d)

        # 拼回（低CPI位置用摘要填充）
        out = V_summary.new_zeros(B, N, d)
        out.scatter_(1, topk_idx_exp, out_top)
        # 低CPI位置用摘要
        out = out + mask_low.unsqueeze(-1).float() * V_summary[:, :1].expand(B, N, d) * mask_low.unsqueeze(-1).float()

        return self.out_proj(out)


def diffkv_reconstruction_loss(model: nn.Module, K_seq: Tensor, V_seq: Tensor,
                               feat_seq: Tensor) -> Tensor:
    """训练辅助损失：让低秩预测头学会重构 K/V 残差。

    K_seq, V_seq: (B, L, d) 完整 K/V 序列
    feat_seq: (B, L, d) 帧特征序列
    """
    if not hasattr(model, 'diff_kv'):
        return torch.tensor(0.0, device=K_seq.device)
    cache: DifferentialKVCache = model.diff_kv
    B, L, d = K_seq.shape
    total_loss = torch.tensor(0.0, device=K_seq.device)
    n = 0
    cache.reset()
    # 第 0 帧作为基准
    cache._base_K = K_seq[:, 0].detach()
    cache._base_V = V_seq[:, 0].detach()
    cache._base_feat = feat_seq[:, 0].detach()
    for t in range(1, L):
        loss = cache.reconstruction_loss(K_seq[:, t], V_seq[:, t], feat_seq[:, t])
        total_loss = total_loss + loss
        n += 1
    return total_loss / max(n, 1)
