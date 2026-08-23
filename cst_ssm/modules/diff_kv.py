"""创新点3：误差有界差分 KV 缓存 + CPI 稀疏注意力（设计文档 §3）。

核心思想：
  - 长视频推理时 KV 缓存线性增长是显存瓶颈。
  - 基准帧存完整 K,V；中间帧只存 **r 维残差系数**（共享低秩基 basis，秩 r≪d）。
  - 存储账（真压缩）：每中间帧存 r 系数 ≪ d 全量，compression_ratio < 1。
  - 引理1（误差界）：‖Attn(Q,K̂,V̂) - Attn(Q,K,V)‖ ≤ C·δ，δ=残差重构误差
    （δ = ‖Δk − basis·coeffs‖，由重构损失压低、由阈值机制上界）。
  - 当残差能量超阈值 → 自动插入新基准帧（自适应 G）。
  - CPI 稀疏注意力：**严格因果**——低 CPI 位置输出=其之前全部低 CPI 内容在值空间的
    因果累积摘要；高 CPI 位置走因果 top-k 注意力，并额外可见自己的因果摘要。

本模块为推理侧效率组件；训练时用无状态的序列重构损失（不跨 batch 残留状态）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class DiffKVConfig:
    rank: int = 8                 # 低秩残差秩 r（每帧存储系数数）
    base_interval: int = 8        # 初始基准帧间隔 G（阈值机制自适应）
    error_threshold: float = 0.1  # 残差能量阈值 ε（超过则插入新基准帧）
    cpi_sparse: bool = True       # 是否启用 CPI 稀疏注意力
    cpi_keep_ratio: float = 0.3   # 高 CPI token 保留比例（走全注意力）


class LowRankResidualHead(nn.Module):
    """低秩残差编解码头：共享低秩基 + 逐帧 r 维系数。

    设计文档 §3 的"截断 SVD 形式"在向量级 KV 残差上的正确落地：
      存储账——全量每帧 d floats；本方案每帧仅 r 个系数（r ≪ d），才是真压缩。
      重构——Δk ≈ basis @ coeffs；误差 = 投影残差（Lemma 1 的 δ）。
    初始化：basis 取 QR 正交基（数值稳定），encoder 置零 → 初始残差≈0，
    从"只用基准帧"的平凡解出发渐进学习补偿残差，训练稳定。
    """

    def __init__(self, d: int, rank: int = 8):
        super().__init__()
        if rank >= d:
            raise ValueError(f"rank({rank}) 必须 < 特征维({d})，否则无压缩收益")
        self.d = d
        self.rank = rank
        # 共享低秩基 (d, r)：reduced QR 正交初始化。只需要 r 列正交基，所以对 (d,r)
        # 做 QR 即可——对 (d,d) 做完整 QR 再切前 r 列，d=3584 时白白多花一个 51MB
        # 的随机矩阵和一次完整分解，而 head_K/head_V 各要来一遍。
        q, _ = torch.linalg.qr(torch.randn(d, rank))
        self.basis = nn.Parameter(q.contiguous())
        # 帧差 → 系数（零初始化：初始残差为 0）
        self.encoder = nn.Linear(d, rank)
        nn.init.zeros_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

    def encode(self, diff: Tensor) -> Tensor:
        """(…, d) 残差向量 → (…, r) 系数（唯一需要存储的量）。"""
        return self.encoder(diff)

    def decode(self, coeffs: Tensor) -> Tensor:
        """(…, r) 系数 → (…, d) 重构残差。"""
        return F.linear(coeffs, self.basis)

    @property
    def per_frame_storage(self) -> int:
        """每帧存储量（floats）：仅 r 个系数。"""
        return self.rank


class DifferentialKVCache(nn.Module):
    """差分 KV 缓存管理器：基准帧(全量 K,V) + 中间帧(r 维系数) + 自适应基准帧插入。

    用法（推理，流式）：
        cache = DifferentialKVCache(cfg, d_model)
        for t in range(L):
            k_full, v_full = cache.update(k_t, v_t, feat_t, t)  # 重构的完整 K/V

    训练：sequence_reconstruction_loss(K_seq, V_seq)——无状态、逐 batch 独立：
    第 0 帧做基准，学习低秩重构后续帧残差。不会跨 batch 残留缓存状态。
    """

    def __init__(self, cfg: DiffKVConfig, d_model: int):
        super().__init__()
        self.cfg = cfg
        self.d = d_model
        self.head_K = LowRankResidualHead(d_model, cfg.rank)
        self.head_V = LowRankResidualHead(d_model, cfg.rank)
        # 运行时状态（推理流式填充；训练损失路径不使用）
        self.reset()

    def reset(self):
        """清空缓存状态。"""
        self._base_K: Tensor | None = None
        self._base_V: Tensor | None = None
        self._base_feat: Tensor | None = None
        self._base_idx: int = 0                          # 上一个基准帧的帧序号
        # 逐帧残差系数 (r,)：**跨基准段累积、插入新基准时不清空**。
        # 早期版本每插一个基准就 clear()，等于把此前所有中间帧的 K/V 永久丢弃——
        # 那样它就不是缓存（服务不了历史），只是个逐帧编解码器，压缩率也算不对。
        self._residual_K: list[Tensor] = []
        self._residual_V: list[Tensor] = []
        self._seg_of_frame: list[int] = []               # 每条残差挂在第几个基准段
        self._bases: list[tuple[Tensor, Tensor]] = []    # 各段基准 (K,V) 全量
        self._n_frames: int = 0
        self._n_base_frames: int = 0
        self._n_res_frames: int = 0                      # 累计中间帧数

    def should_insert_base(self, feat_t: Tensor) -> bool:
        """是否插入新基准帧。两条判据任一满足即插：

        ① **相对**残差能量 ‖feat_t − feat_base‖ / (‖feat_base‖ + η) > ε。
           必须用相对量：早期版本拿未归一化的 d_model 空间 L2 范数与常数 0.1 比，
           而随便一个特征向量的范数都远超 0.1 —— 实测 128 帧全部成为基准帧、
           compression_ratio 恒为 1.0，低秩残差路径一次都没走过。
        ② 距上一个基准已满 base_interval 帧（论文里的 G）。给自适应间隔一个上界，
           同时让"固定 G vs 自适应 G"的消融有真实可比的对照物（此前 G 是死字段）。
        """
        if self._base_feat is None:
            return True
        diff = torch.linalg.vector_norm(feat_t - self._base_feat, dim=-1)
        base = torch.linalg.vector_norm(self._base_feat, dim=-1)
        rel = (diff / (base + 1e-6)).mean()
        if bool(rel.item() > self.cfg.error_threshold):
            return True
        G = int(self.cfg.base_interval)
        return G > 0 and (self._n_frames - self._base_idx) >= G

    def update(self, K_t: Tensor, V_t: Tensor, feat_t: Tensor,
               t: int) -> tuple[Tensor, Tensor]:
        """流式更新缓存并返回重构的完整 (K_full, V_full)。

        K_t, V_t: (B, d) 当前帧完整 K/V；feat_t: (B, d) 帧特征（基准帧插入判据）。
        中间帧仅把 r 维系数入缓存（detach），不存全量 → 显存 O(n_base·d + n_res·r)。
        """
        self._n_frames += 1
        if self.should_insert_base(feat_t):
            # 插入新基准帧（全量存储）
            self._base_K = K_t.detach().clone()
            self._base_V = V_t.detach().clone()
            self._base_feat = feat_t.detach().clone()
            self._base_idx = self._n_frames
            self._bases.append((self._base_K, self._base_V))
            self._n_base_frames += 1
            return K_t, V_t

        # 中间帧：编码残差 → r 维系数（真正的存储单元）
        coeff_K = self.head_K.encode(K_t - self._base_K)
        coeff_V = self.head_V.encode(V_t - self._base_V)
        self._residual_K.append(coeff_K.detach())
        self._residual_V.append(coeff_V.detach())
        self._seg_of_frame.append(len(self._bases) - 1)
        self._n_res_frames += 1
        # 重构完整 K/V 供注意力使用
        return (self._base_K + self.head_K.decode(coeff_K),
                self._base_V + self.head_V.decode(coeff_V))

    @torch.no_grad()
    def reconstruct_residual_frames(self) -> tuple[Tensor | None, Tensor | None]:
        """把缓存里**所有历史中间帧**的 K/V 从 (所属基准 + 低秩系数) 重建出来。

        返回 (K, V)，形状均为 (n_res, B, d)；无中间帧时返回 (None, None)。
        这是"它确实是个缓存"的可验证证据——历史帧能被重建，而不是算完就丢。
        """
        if not self._residual_K:
            return None, None
        K = torch.stack([self._bases[s][0] + self.head_K.decode(c)
                         for s, c in zip(self._seg_of_frame, self._residual_K)])
        V = torch.stack([self._bases[s][1] + self.head_V.decode(c)
                         for s, c in zip(self._seg_of_frame, self._residual_V)])
        return K, V

    @property
    def compression_ratio(self) -> float:
        """存储比 = 实际存储 / 全量存储（K+V 双通道计）。r ≪ d 时 < 1（真压缩）。

        用**累计**计数器而非当前列表长度：残差列表已不再被 clear，两者现在一致，
        但保留独立计数器可避免将来任何裁剪逻辑再次把这个指标算成偏乐观的值。
        """
        if self._n_frames == 0:
            return 1.0
        full = self._n_frames * 2 * self.d
        actual = (self._n_base_frames * 2 * self.d
                  + self._n_res_frames * 2 * self.cfg.rank)
        return actual / full

    @property
    def n_base_frames(self) -> int:
        """已插入的基准帧数（全量存储的那些）。"""
        return self._n_base_frames

    @property
    def n_residual_frames(self) -> int:
        """走低秩残差路径的帧数（每帧只存 r 个系数）。"""
        return self._n_res_frames

    @property
    def stats(self) -> dict:
        """缓存统计（效率评测用）。"""
        return {"n_frames": self._n_frames, "n_base": self._n_base_frames,
                "n_residual": self._n_res_frames, "rank": self.cfg.rank,
                "d": self.d, "compression_ratio": self.compression_ratio}

    def sequence_reconstruction_loss(self, K_seq: Tensor, V_seq: Tensor) -> Tensor:
        """训练损失（无状态）：第 0 帧为基准，低秩头重构后续帧的 K/V 残差。

        K_seq, V_seq: (B, L, d)。基准帧切片 detach——基准帧全量存储、无需梯度，
        只让残差重构路径学习。逐 batch 独立，不依赖/不修改运行时缓存状态。
        """
        if K_seq.shape[1] < 2:
            return K_seq.sum() * 0.0
        diff_K = K_seq[:, 1:] - K_seq[:, :1].detach()
        diff_V = V_seq[:, 1:] - V_seq[:, :1].detach()
        recon_K = self.head_K.decode(self.head_K.encode(diff_K))
        recon_V = self.head_V.decode(self.head_V.encode(diff_V))
        return F.mse_loss(recon_K, diff_K) + F.mse_loss(recon_V, diff_V)


class CPISparseAttention(nn.Module):
    """因果 CPI 稀疏注意力：高 CPI 走因果注意力，低 CPI 用**因果值空间摘要**替代。

    设计§3 性质3：摘要替代误差 ≤ 簇内方差。实现严格因果：
      - 低 CPI 位置 i 的输出 = 位置 ≤ i 的所有低 CPI 内容在投影值 V 空间的
        CPI 加权累积摘要（cumsum 实现，O(N)）；无历史时回退自身 V；
      - 高 CPI 位置（top-k）：在高 CPI keys 之间做因果注意力（key 位置 ≤ query 位置），
        并额外把"自己的因果摘要"作为一个 key/value 参与注意力；
      - 摘要在值空间计算并真实参与注意力 → 与全注意力的语义一致；
      - Q/K/V 只投影一次后 gather，无重复计算。

    输入：x (B, N, d), cpi (B, N) 可选。输出：(B, N, d)。
    """

    def __init__(self, d: int, n_heads: int = 8, keep_ratio: float = 0.3,
                 causal: bool = True):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d({d}) 必须能被 n_heads({n_heads}) 整除")
        self.keep_ratio = keep_ratio
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal    # 因果约束（CHST 架构要求）
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x: Tensor, cpi: Tensor | None = None) -> Tensor:
        """x: (B, N, d), cpi: (B, N) 可选。无 cpi 时退化为（因果）全注意力。"""
        B, N, d = x.shape
        h, hd = self.n_heads, self.head_dim
        Q = self.q_proj(x).view(B, N, h, hd)
        K = self.k_proj(x).view(B, N, h, hd)
        V = self.v_proj(x).view(B, N, h, hd)

        if cpi is None:
            # 无 CPI 信号：全注意力（+ 因果 mask）
            Qt, Kt, Vt = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
            attn = (Qt @ Kt.transpose(-2, -1)) * self.scale
            if self.causal:
                mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool),
                                  diagonal=1)
                attn = attn.masked_fill(mask, float("-inf"))
            out = (attn.softmax(-1) @ Vt).transpose(1, 2).reshape(B, N, d)
            return self.out_proj(out)

        # ---- CPI 稀疏（严格因果）----
        k = max(1, int(N * self.keep_ratio))
        _, topk_idx = cpi.topk(k, dim=-1)                      # (B, k)

        # 低 CPI 值的因果累积摘要：位置 i 处 = Σ_{j≤i, j∈low} cpi_j·V_j / Σ cpi_j
        is_low = torch.ones(B, N, dtype=torch.bool, device=x.device)
        is_low.scatter_(1, topk_idx, False)
        w = cpi.clamp_min(0.0) * is_low                        # (B, N)
        cum_wV = (V * w[:, :, None, None]).cumsum(dim=1)       # (B,N,h,hd)
        cum_w = w.cumsum(dim=1)                                # (B, N)
        has_hist = cum_w > 1e-8                                # (B, N)
        summary = cum_wV / cum_w[:, :, None, None].clamp_min(1e-8)  # (B,N,h,hd)

        # 低 CPI 位置输出 = 该位置的因果摘要；无低 CPI 历史时回退自身 V
        out_low = torch.where(has_hist[:, :, None, None], summary, V)  # (B,N,h,hd)

        # top-k 查询：每个查询可见全部 top-k keys（因果遮蔽）+ 自己的因果摘要作额外 key
        idx = topk_idx.unsqueeze(-1).unsqueeze(-1).expand(B, k, h, hd)
        Q_top = Q.gather(1, idx).transpose(1, 2)               # (B,h,k,hd)
        K_top = K.gather(1, idx).transpose(1, 2)
        V_top = V.gather(1, idx).transpose(1, 2)
        S_top = summary.gather(1, idx).transpose(1, 2)         # (B,h,k,hd) 各查询自己的因果摘要

        # 每个查询的 key 集 = k 个 top-k keys + 1 个"自身的因果摘要"。
        # 注意不要 cat：K_top.unsqueeze(2).expand(B,h,k,k,hd) 是视图，但 torch.cat 会把它
        # **完整物化**成 O(B·k²·d)——d=3584 时 L=512 就要 1.35GB，比它想替代的全注意力
        # 还贵两个数量级。改成两块分别算 logits，峰值回到 O(B·h·k²)：
        #   ① top-k keys 之间：标准 (B,h,k,hd)@(B,h,hd,k) → (B,h,k,k)
        #   ② 自身摘要那一列：逐查询点积 (Q_top*S_top).sum(-1) → (B,h,k,1)
        attn_kk = (Q_top @ K_top.transpose(-2, -1)) * self.scale      # (B,h,k,k)
        attn_s = (Q_top * S_top).sum(-1, keepdim=True) * self.scale   # (B,h,k,1)
        if self.causal:
            # 高 CPI keys 之间的因果约束：key 位置 > query 位置 → 遮蔽
            # 追加的摘要 key 天然因果（内容只含 ≤ 自身位置），永不遮蔽 → 每个查询至少有一个可见 key
            pos_q = topk_idx.unsqueeze(2)                      # (B,k,1)
            pos_k = topk_idx.unsqueeze(1)                      # (B,1,k)
            block = (pos_k > pos_q).view(B, 1, k, k)           # (B,1,k,k)
            attn_kk = attn_kk.masked_fill(block, float("-inf"))
        # softmax 在 [top-k keys | 自身摘要] 拼起来的 k+1 维上做一次，再拆回两块加权求和，
        # 语义与拼接后 softmax 完全一致，但从不物化 (B,h,k,k+1,hd) 的 K_all/V_all。
        w = torch.softmax(torch.cat([attn_kk, attn_s], dim=-1), dim=-1)   # (B,h,k,k+1)
        w_kk, w_s = w[..., :k], w[..., k:]                                # (B,h,k,k) / (B,h,k,1)
        out_top = w_kk @ V_top + w_s * S_top                              # (B,h,k,hd)
        out_top = out_top.transpose(1, 2).reshape(B, k, d)   # (B,h,k,hd)：头在 dim1，须先转置

        # 拼回：低 CPI 位置 = 因果摘要；top-k 位置 = 注意力输出
        # out_low 是 (B,N,h,hd)——**头已经在 dim2、时间在 dim1**，直接 reshape 即可。
        # 上一行 out_top 是 (B,h,k,hd) 才需要 transpose(1,2)；两者形状语义不同，
        # 套用同一个写法会把「头」和「时间」两个轴搅在一起：输出位置 n 会读到源位置
        # ((n·d+j) mod (N·hd))//hd 的内容，既是错值，又让未来帧泄漏到过去位置
        # （实测 L=16/h=8 时扰动末帧会改变位置 5,7,9,11,13），直接破坏本模块声称的严格因果。
        out = out_low.reshape(B, N, d)
        # 用非就地 scatter：out_low 连续，上面的 reshape 返回的是**视图**，就地写会改到
        # torch.where 的输出本身。当前虽无别处引用、autograd 也扛得住，但等价的非就地写法
        # 峰值显存与修复前的 transpose+reshape（必然拷贝）持平，没有理由冒这个险。
        out = out.scatter(1, topk_idx.unsqueeze(-1).expand(B, k, d), out_top)
        return self.out_proj(out)


def diffkv_reconstruction_loss(model: nn.Module, K_seq: Tensor, V_seq: Tensor,
                               feat_seq: Tensor | None = None) -> Tensor:
    """训练辅助损失：让低秩编解码头学会重构 K/V 残差（无状态，逐 batch 独立）。

    K_seq, V_seq: (B, L, d) 完整 K/V 序列。feat_seq 保留参数仅为向后兼容。
    """
    cache = getattr(model, "diff_kv", None)
    if cache is None:
        return K_seq.sum() * 0.0
    return cache.sequence_reconstruction_loss(K_seq, V_seq)
