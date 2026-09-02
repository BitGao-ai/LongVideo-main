"""LLM 推理层（设计方案 §3.4）：文本 Token 全自注意力 + 视觉状态交叉注意力（无视觉历史 KV 缓存）。

- GatedCrossAttention : Flamingo 风格 tanh 门控交叉注意力，初始门=0（初始等价原 LLM，训练稳定）。
- VisualConditionedLM : 自包含 decoder-only LLM，层间插入门控交叉注意力到视觉状态。默认小规模，
                        既是可训练的具体实现，也是无需下载 7B 权重即可 smoke test 的 stand-in。
- 低耦合：VisionBackbone / LLMBackbone 为结构化协议，可换任意骨干（含 HF 因果 LM，见 README 集成说明）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


# --------------------------- 低耦合协议 ---------------------------
@runtime_checkable
class VisionBackbone(Protocol):
    def forward(self, frames: Tensor) -> Tensor:  # (B,L,3,H,W) 或 (B,L,P,d) → (B,L,d)
        ...


@runtime_checkable
class LLMBackbone(Protocol):
    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None, labels: Tensor | None = None,
                need_logits: bool = True) -> dict:
        ...


# --------------------------- 模块 ---------------------------
def _safe_softmax(attn: Tensor, dim: int = -1) -> Tensor:
    """softmax，但"整行 key 都被遮蔽"时输出全 0 而不是 NaN。

    全 -inf 行的 softmax 是 0/0 → NaN，会顺着反传把整个梯度污染成 NaN，且没有任何报错。
    触发条件很现实：某个样本的 visual_mask 全 False（整段视频都是 padding）、
    或外部注入的 attention_mask 把某个 query 位置的全部 key 都挡掉。
    这类行本就没有可见的 key，输出 0（对残差不做贡献）是唯一有意义的取值。

    保留本函数是因为它定义了本文件两处注意力的**遮蔽语义**；实际计算已改走
    `_sdpa`（见其 docstring），那里在 SDPA 之后按同样规则把死行清零。
    """
    visible = torch.isfinite(attn).any(dim=dim, keepdim=True)
    w = attn.masked_fill(~visible, 0.0).softmax(dim)   # 先避开 0/0
    return w * visible                                 # 再把这些行整体归零


def _sdpa(q: Tensor, k: Tensor, v: Tensor, scale: float,
          allow: Tensor | None = None, is_causal: bool = False) -> Tensor:
    """scaled_dot_product_attention + "整行被遮蔽则输出 0" 的语义补齐。

    **为什么必须换掉手写注意力**：原实现显式物化 `(B,heads,T,T)` 的分数矩阵，而
    `_safe_softmax` 在其上又连开三份（masked_fill / softmax / 乘 visible），加上进来时
    已有的因果与 padding 两次 masked_fill，一层内同时在世约 6 份全尺寸张量。
    按 configs/default.yaml 的 max_len=8704、B=2、8 头算，单份 bf16 就是 2.42 GB，
    一层约 14.5 GB——CST-SSM 每帧产出 1 个 soft token，所以 T 就是帧数，长视频下
    这是必然触发的。SDPA 的显存是 O(T)，不物化分数矩阵。

    allow: 可选 bool 掩码，True=该 (query,key) 对可见，须可广播到 (B,heads,Lq,Lk)。
           与 is_causal 互斥（SDPA 的约束），所以因果+padding 的组合要合成到 allow 里。
    语义补齐：SDPA 对"整行皆不可见"的 query 会输出 NaN（softmax 的 0/0），与
    _safe_softmax 约定的 0 不一致，故在此显式清零。allow 为 None 时不可能出现死行
    （is_causal 下每个 query 至少能看到自己），跳过该步。

    清零是**无条件**执行的，不加 `if dead.any()` 那种快路：`.any()` 要把结果取回 host，
    在 GPU 上是每层一次强制同步，会把整条流水线卡住（同 qwen3_vl._scatter_visual 里
    避免逐步 D2H 的理由）。而 masked_fill 作用在 (B,heads,Lq,hd) 上，是 O(T·d) 而非
    O(T²)，那点写入远比一次同步便宜。
    """
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow,
                                         is_causal=is_causal, scale=scale)
    if allow is not None:
        dead = ~allow.any(dim=-1, keepdim=True)        # (…,Lq,1) 可广播
        out = out.masked_fill(dead, 0.0)
    return out


class GatedCrossAttention(nn.Module):
    """文本查询 → 视觉状态键/值的门控交叉注意力。tanh 门初始 0，保证初始不改变 LLM 行为。"""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.gate = nn.Parameter(torch.zeros(1))   # tanh 门，初始 0

    def forward(self, text: Tensor, visual: Tensor,
                visual_mask: Tensor | None = None) -> Tensor:
        B, Lt, d = text.shape
        Lv = visual.shape[1]
        q = self.q(self.norm_q(text)).view(B, Lt, self.n_heads, -1).transpose(1, 2)
        kv = self.kv(self.norm_kv(visual)).view(B, Lv, 2, self.n_heads, -1)
        k, v = kv.permute(2, 0, 3, 1, 4).unbind(0)     # (B,heads,Lv,hd)
        # 掩码只依赖 key（哪些帧有效），与 query 位置无关 → (B,1,1,Lv) 靠广播即可，
        # 完全不用物化 (B,heads,Lt,Lv)。Lt=8704/Lv=8192 时那是 2.28 GB/份。
        allow = None if visual_mask is None else visual_mask[:, None, None, :]
        out = _sdpa(q, k, v, self.scale, allow=allow)
        out = out.transpose(1, 2).reshape(B, Lt, d)
        return text + torch.tanh(self.gate) * self.proj(out)


class _CausalSelfAttn(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        B, T, d = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, d // self.n_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        if key_padding_mask is None:
            # 没有 padding：直接用 SDPA 的内建因果掩码，连 (T,T) 都不建
            out = _sdpa(q, k, v, self.scale, is_causal=True)
        else:
            # SDPA 不允许 attn_mask 与 is_causal 并用，故把因果与 padding 合成一张。
            # 它是 (B,1,T,T) 的 **bool**（1 字节/元素、跨头广播），T=8704/B=2 时 151 MB；
            # 而旧实现是 6 份 (B,heads,T,T) 的分数张量，同口径约 14.5 GB。
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril_()
            allow = causal & key_padding_mask[:, None, None, :]        # True=可见
            out = _sdpa(q, k, v, self.scale, allow=allow)
        out = out.transpose(1, 2).reshape(B, T, d)
        return self.proj(out)


class _DecoderLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CausalSelfAttn(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass
class LLMConfig:
    vocab_size: int = 32000
    dim: int = 512
    n_layer: int = 6
    n_head: int = 8
    cross_every: int = 2       # 每隔多少层插入一次门控交叉注意力
    max_len: int = 2048
    # LLM 段逐层梯度检查点。与 CSTSSMConfig.eacs_chunk 同性质：**数值上逐位等价**，
    # 只用重算换显存（反向时重跑一遍层内前向，约 +30% 前向计算）。
    #
    # 为什么长视频必须有它：CST-SSM 每帧产出 1 个 soft token（cst_ssm_model.py 的
    # encode_visual），所以 LLM 序列长度 == 帧数。不开检查点时 LLM 段每 token 要留
    # n_layer × O(hidden) 的激活，L=4000 时它是 CST-SSM 段（已被 eacs_chunk 压到
    # 0.057 MB/帧）的数倍——真正的长视频显存天花板在这里，不在时序段。
    #
    # 实现固定用 use_reentrant=False，原因见 VisualConditionedLM.forward 的注释：
    # 可重入版本与 DDP(find_unused_parameters=True) 不兼容，而 Trainer 正是这么配的。
    grad_checkpoint: bool = False
    # ---- 显存主开关之三：LM 头 logits 分块（长视频 + 大词表的真正天花板）----
    # 一次性算 (B,T,V) 的 logits 再求 CE，峰值约 B·T·V×12 字节，来源有四份：
    #   ① bf16 logits            B·T·V×2
    #   ② logits[:, :-1].reshape 是**复制**而非视图（切片非连续）  B·T·V×2
    #   ③ F.cross_entropy 在 autocast 名单里被提升到 fp32          B·T·V×4
    #   ④ log_softmax 的 fp32 输出存给反向                         B·T·V×4
    # Qwen3-VL 的 V=151936，B=2/T=8192 时这一项就是 29.9GB——比 4B 底座权重还大，
    # 40G 卡上光这一项加权重就已 38.4GB，激活无处安放。
    #
    # 分块后按 loss_chunk 个 token 一段算 CE 求和，每段套梯度检查点（反向逐段重算），
    # 峰值降到 loss_chunk·V×10：1024 时约 1.5GB，与 T 无关。
    # 数值等价（fp32 分段求和，与一次求和的差异在末位）；0=关闭，回到一次性路径。
    # 只在**训练**（grad 开启）时生效——推理/验证要拿 logits 做 MCQ 打分，且无反向图，
    # 一次性物化反而更便宜。
    loss_chunk: int = 1024


def _ce_chunk_sum(hidden: Tensor, labels: Tensor, lm_head: nn.Module,
                  ignore_index: int) -> Tensor:
    """一段 token 的 CE 求和。单独成函数是为了能整段套 checkpoint（反向再算一遍）。"""
    return F.cross_entropy(lm_head(hidden), labels,
                           ignore_index=ignore_index, reduction="sum")


def chunked_lm_loss(hidden: Tensor, labels: Tensor, lm_head: nn.Module,
                    chunk_size: int, ignore_index: int = -100) -> Tensor:
    """分块算 next-token 交叉熵，**不物化整块 (B,T,V) logits**。

    等价于：
        logits = lm_head(hidden)
        F.cross_entropy(logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1),
                        ignore_index=ignore_index)

    差异只有两处，都在预期内：
      · fp32 求和被拆成若干段再相加，末位可能有 ~1e-7 相对差（与 eacs_chunk 那种
        逐位等价不同，这里做不到逐位，因为浮点加法不满足结合律）。
      · 全 batch 都是 ignore_index 时，原实现返回 nan（0/0），这里返回 0。
        后者更安全：nan 会顺着反向污染全部梯度且不报错。

    梯度检查点是**必需的而非优化**：不套的话每段的 log_softmax fp32 输出照样都要
    存给反向，累计起来和不分块一模一样，分块就白做了。
    """
    B, T, d = hidden.shape
    if T < 2:                                   # 没有 next-token 对，返回一个连着图的 0
        return hidden.sum() * 0.0
    flat_h = hidden[:, :-1].reshape(-1, d)      # (B·(T-1), d) 预测位
    flat_y = labels[:, 1:].reshape(-1)          # (B·(T-1),)   目标
    n_valid = (flat_y != ignore_index).sum()
    # requires_grad 为 False（纯推理路径误调）时 checkpoint 会告警且毫无收益，直接跳过
    use_ckpt = torch.is_grad_enabled() and flat_h.requires_grad
    total = flat_h.new_zeros((), dtype=torch.float32)
    for s in range(0, flat_h.shape[0], chunk_size):
        hs, ys = flat_h[s:s + chunk_size], flat_y[s:s + chunk_size]
        if use_ckpt:
            part = checkpoint(_ce_chunk_sum, hs, ys, lm_head, ignore_index,
                              use_reentrant=False)
        else:
            part = _ce_chunk_sum(hs, ys, lm_head, ignore_index)
        total = total + part
    return total / n_valid.clamp_min(1).to(total.dtype)


def use_chunked_loss(loss_chunk: int, labels: Tensor | None,
                     need_logits: bool = True) -> bool:
    """是否走分块 CE。

    三个条件：开关 >0、有 labels、且**这一步的 logits 没有消费者**。后者由两条独立的
    证据判定，满足任意一条即可：

      · `torch.is_grad_enabled()` —— 训练态。训练侧没有任何代码读 out["logits"]
        （finetune_loss 只读 out["loss"]），所以恒走分块。这条是原有行为，不变。
      · `not need_logits` —— 调用方显式声明不要 logits。**这是新增的一条**，专为
        `Trainer.validate` 而设：它跑在 @torch.no_grad() 下，但同样只读 out["loss"]，
        旧闸门却因为"没有梯度"就去物化整块 (B,T,V)。按 LLMConfig.loss_chunk 的字节账，
        no_grad 下约 8 B/元素（比训练态少了"存给反向的 log_softmax"那份），
        B=2/T=8192/V=151936 就是 **≈20 GB**——40G 卡上训练步正常、一开 --val-manifest
        就在验证步 OOM，是最难排查的那类失败。

    默认 need_logits=True 保证**所有既有调用方逐字不变**：eval_benchmark 的 MCQ 打分要在
    答案首 token 位置比较各选项概率，那里必须拿到 logits，且 no_grad 下没有反向图、
    一次性物化用完即释放，峰值远低于训练。
    """
    if loss_chunk <= 0 or labels is None:
        return False
    return torch.is_grad_enabled() or not need_logits


class VisualConditionedLM(nn.Module):
    """decoder-only LLM，层间插入门控交叉注意力到视觉状态。"""

    def __init__(self, cfg: LLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_len, cfg.dim))
        self.layers = nn.ModuleList([_DecoderLayer(cfg.dim, cfg.n_head) for _ in range(cfg.n_layer)])
        self.cross = nn.ModuleDict({
            str(i): GatedCrossAttention(cfg.dim, cfg.n_head)
            for i in range(cfg.n_layer) if i % cfg.cross_every == 0
        })
        self.norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight            # 权重绑定省显存

    def _maybe_ckpt(self, module, *args):
        """按 cfg.grad_checkpoint 决定是否对一层做梯度检查点。

        只在**训练且梯度开启**时启用：推理下 checkpoint 既省不到显存（本就没有反向图），
        还会因"没有输入 requires_grad"而告警。

        use_reentrant=False 是硬性要求而非偏好：可重入实现要求 DDP 关闭
        find_unused_parameters 并开 static_graph，而 Trainer 用的是
        find_unused_parameters=True（CPIB/grounding 属可选模块，不是每步都参与）。
        非重入版本与该配置兼容，且同样保留 RNG 状态（dropout 等随机层数值不变）。
        """
        if not (self.cfg.grad_checkpoint and self.training and torch.is_grad_enabled()):
            return module(*args)
        return checkpoint(module, *args, use_reentrant=False)

    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None,
                visual_mask: Tensor | None = None,
                labels: Tensor | None = None,
                need_logits: bool = True) -> dict:
        """need_logits: 调用方是否会读 out["logits"]。给 False 时（如 Trainer.validate）
        no_grad 下也走分块 CE，不物化 (B,T,V)。默认 True = 既有行为逐字不变。"""
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            raise ValueError(
                f"文本长度 T={T} 超过 LLMConfig.max_len={self.cfg.max_len}（位置编码只有这么长）。"
                f"请调大 max_len，或把数据侧的 max_text_len 降到 ≤{self.cfg.max_len}。")
        h = self.embed(input_ids) + self.pos[:, :T]
        for i, layer in enumerate(self.layers):
            h = self._maybe_ckpt(layer, h, attention_mask)
            if str(i) in self.cross:
                h = self._maybe_ckpt(self.cross[str(i)], h, visual_states, visual_mask)
        h = self.norm(h)
        # 训练走分块 CE：不物化 (B,T,V)，峰值与 T 无关（见 LLMConfig.loss_chunk）。
        # 此时不返回 logits——训练侧没有任何消费者（finetune_loss 只读 out["loss"]），
        # 返回它就等于把刚省下的那份显存又原样占回去。
        if use_chunked_loss(self.cfg.loss_chunk, labels, need_logits):
            return {"loss": chunked_lm_loss(h, labels, self.lm_head, self.cfg.loss_chunk)}
        logits = self.lm_head(h)
        out = {"logits": logits}
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            out["loss"] = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return out
