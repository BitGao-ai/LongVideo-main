"""接入 Qwen3-VL 系列（替换自包含 stand-in 的视觉塔 + LLM）。

集成策略（与 CST-SSM 设计一致：视觉先经 EACS 连续时序压缩，再喂 LLM）：
  视觉端：用 Qwen3-VL 视觉塔离线抽帧级特征（feat_dim=out_hidden_size，默认 3584）→ 存 .npz 特征缓存
          → CST-SSM 段1(FeatureAdapter)/段2(EACS) 做**时长无关**的连续时序压缩。
  语言端：Qwen3VLLanguageModel 包装 Qwen3-VL 的 LLM，把 CST-SSM 压缩后的视觉状态（每帧 1 个向量，
          远少于 Qwen 原生 帧×patch 个视觉 token）经 **video 占位符位置注入 inputs_embeds**（soft-token），
          再走 Qwen3 解码器 + LM 头算 loss。相比原生一次性喂全部帧 patch，KV 规模大幅下降。

依赖：transformers（含 Qwen3-VL）。lazy import，未安装时给出清晰报错，不影响其余模块。
关键常量（来自 Qwen3VLConfig 默认）：video_token_id=151656, image_token_id=151655, out_hidden_size=3584。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..modules.llm_interface import chunked_lm_loss, use_chunked_loss

# Qwen3 解码器里可挂 LoRA 的线性层名（供 utils.apply_lora 使用）
QWEN3_LORA_TARGETS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
})
VIDEO_TOKEN_ID = 151656
IMAGE_TOKEN_ID = 151655
QWEN3VL_OUT_HIDDEN = 3584          # 视觉塔输出维（feat_dim）


def _import_hf():
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        return Qwen3VLForConditionalGeneration, AutoProcessor
    except Exception as e:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "接入 Qwen3-VL 需要 transformers（含 Qwen3-VL）：pip install -U transformers。"
            f"\n原始错误: {e}")


# 加载底座时可能不被当前环境支持的可选 kwarg：不认就逐个摘掉重试，而不是让加载失败。
# 顺序即摘除优先级——先摘性能优化项，保留 torch_dtype 这类影响正确性/显存的关键项。
_OPTIONAL_LOAD_KWARGS = ("device_map", "low_cpu_mem_usage")


def _from_pretrained_resilient(cls, model_name: str, load_kw: dict):
    """`cls.from_pretrained(model_name, **load_kw)`，可选 kwarg 不被支持时逐个摘掉重试。

    需要这层是因为两个 kwarg 的可用性都取决于环境而非代码：
      · low_cpu_mem_usage —— 老版本 transformers 的签名里没有；
      · device_map        —— 需要装了 accelerate，缺了会抛 ImportError/ValueError。
    两者都**只影响加载时的峰值内存与耗时，不影响权重正确性**，所以退回比报错更合适；
    但每退一步都要出声，否则"我明明开了 device_map"会毫无提示地落空。
    """
    kw = dict(load_kw)
    while True:
        try:
            return cls.from_pretrained(model_name, **kw)
        except (TypeError, ImportError, ValueError) as e:
            dropped = next((k for k in _OPTIONAL_LOAD_KWARGS if k in kw), None)
            if dropped is None:
                raise            # 已经没有可摘的可选项了，说明是真错误，原样抛出
            kw.pop(dropped)
            print(f"[qwen3vl] 当前环境不支持 from_pretrained 的 {dropped}"
                  f"（{type(e).__name__}: {str(e)[:120]}），已摘掉重试。"
                  f"{'装 accelerate 可启用直接落卡加载。' if dropped == 'device_map' else ''}")


# ============================ 语言端 ============================
class Qwen3VLLanguageModel(nn.Module):
    """包装 Qwen3-VL LLM：把 CST-SSM 压缩视觉状态注入 video 占位符位置，返回 {logits, loss}。

    forward 签名与 modules.llm_interface.VisualConditionedLM 完全一致，可直接作为 CSTSSMModel 的 llm 注入。
    """

    def __init__(self, hf_model, video_token_id: int | None = None,
                 train_base: bool = False, grad_checkpoint: bool = False,
                 loss_chunk: int = 1024):
        super().__init__()
        self.hf = hf_model
        # 从 config 读 video_token_id（不同 Qwen3-VL 大小/变体更稳健），回退常量
        self.video_token_id = (video_token_id if video_token_id is not None
                               else getattr(hf_model.config, "video_token_id", VIDEO_TOKEN_ID))
        self.hidden = hf_model.config.text_config.hidden_size
        if not train_base:                              # 默认冻结基座（配合 LoRA）
            for p in self.hf.parameters():
                p.requires_grad_(False)
        self.grad_checkpoint = bool(grad_checkpoint)
        # LM 头 logits 分块（见 LLMConfig.loss_chunk）。Qwen3-VL 的词表 151936 使这一项
        # 成为长视频训练的第一显存瓶颈：B=2/T=8192 时一次性路径要 29.9GB，比底座权重还大。
        self.loss_chunk = int(loss_chunk)
        if self.grad_checkpoint:
            self._enable_grad_checkpoint()

    def _enable_grad_checkpoint(self) -> None:
        """对 Qwen3-VL 开逐层梯度检查点。数值等价，只用重算换显存。

        两个细节都不是可选项：
          1) use_reentrant=False —— 可重入实现与 DDP(find_unused_parameters=True) 不兼容，
             而 Trainer 正是这么配的（CPIB/grounding 是可选模块，不保证每步参与）。
          2) use_cache=False —— 训练时 KV cache 与检查点互斥，HF 会打印告警并强行关掉；
             这里提前显式关闭，既免告警也避免白占一份 cache 显存。

        冻结基座（LoRA）下依然需要它：LoRA 挂在各层内部，反向要穿过整个解码器，
        所有中间激活照样要留——冻结省的是优化器状态，不是激活。
        """
        cfg = getattr(self.hf, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "use_cache"):
            text_cfg.use_cache = False
        enable = getattr(self.hf, "gradient_checkpointing_enable", None)
        if enable is None:
            raise AttributeError(
                "该 HF 模型没有 gradient_checkpointing_enable；请升级 transformers，"
                "或把 llm.grad_checkpoint 设为 false")
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            # 老版本 transformers 的签名没有 gradient_checkpointing_kwargs。此时它会走
            # 可重入实现——与 find_unused_parameters=True 冲突，宁可显式报错也不要在
            # 8 卡上跑出"某些参数没有梯度"的诡异报错。
            raise RuntimeError(
                "当前 transformers 的 gradient_checkpointing_enable 不支持 "
                "use_reentrant=False（版本过旧）。可重入检查点与 Trainer 的 "
                "DDP(find_unused_parameters=True) 不兼容，请升级 transformers "
                "或关闭 llm.grad_checkpoint")

    # -- 组件访问（跨 transformers 版本做 getattr 兜底）--
    def _embed_tokens(self):
        return self.hf.get_input_embeddings()

    def _text_model(self):
        base = getattr(self.hf, "model", self.hf)
        return getattr(base, "language_model", base)     # Qwen3VLTextModel 或等价

    def _lm_head(self):
        return self.hf.get_output_embeddings()

    @staticmethod
    def _scatter_visual(inputs_embeds: Tensor, placeholder_mask: Tensor,
                        visual_states: Tensor, visual_mask: Tensor | None) -> Tensor:
        """把 visual_states 写入 inputs_embeds 中 placeholder_mask 为 True 的位置（行主序对齐）。

        约定：每个样本的 video 占位符数量 == 该样本有效视觉状态数（由数据侧保证，见 README）。
        逐样本强校验该约定：失配时直接 raise——否则视觉状态会跨样本串位写入
        别的样本的占位符位置，且被全局补零掩盖产生静默错训。
        """
        B, T, d = inputs_embeds.shape
        if visual_mask is not None:
            vis = visual_states[visual_mask]             # (n_valid, d)
            n_vis_per = visual_mask.sum(dim=1)           # (B,) 逐样本有效状态数
        else:
            vis = visual_states.reshape(-1, d)
            n_vis_per = torch.full((B,), visual_states.shape[1],
                                   device=inputs_embeds.device, dtype=torch.long)
        n_slot_per = placeholder_mask.sum(dim=1)         # (B,) 逐样本占位符数
        n_slots = int(n_slot_per.sum().item())
        if n_slots == 0:
            return inputs_embeds                          # 无占位符：纯文本
        # 校验在**同设备**上做：`n_vis_per.cpu() vs n_slot_per.cpu()` 每步两次 D2H 同步，
        # 会把整条流水线卡住；只在真的失配时才取回详情。
        if not bool(torch.equal(n_vis_per, n_slot_per.to(n_vis_per.dtype))):
            bad = (n_vis_per != n_slot_per).nonzero(as_tuple=True)[0].tolist()
            ns, nv = n_slot_per.cpu().tolist(), n_vis_per.cpu().tolist()
            detail = ", ".join(f"样本{i}: 占位符={ns[i]} vs 有效视觉状态={nv[i]}" for i in bad[:8])
            raise ValueError(
                "[Qwen3VL] 逐样本 video 占位符数与有效视觉状态数不一致（失配样本："
                f"{detail}）。请检查数据侧帧数/占位符生成与 max_video_tokens 截断，"
                "避免视觉状态跨样本串位。")
        if vis.shape[0] < n_slots:                        # 校验后理论不可达，保留兜底防回归
            vis = torch.cat([vis, vis.new_zeros(n_slots - vis.shape[0], d)], 0)
        vis = vis[:n_slots].to(inputs_embeds.dtype)
        # masked_scatter 直接产出新张量，省掉 clone 那一份 (B,T,d)
        # （T=2048/d=3584/bf16 时每步 28MB）。inputs_embeds 是 embedding 的输出、
        # 本就是新张量，不存在改坏调用方数据的风险。
        return inputs_embeds.masked_scatter(placeholder_mask.unsqueeze(-1), vis)

    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None,
                visual_mask: Tensor | None = None,
                labels: Tensor | None = None,
                need_logits: bool = True) -> dict:
        """need_logits: 调用方是否会读 out["logits"]（见 llm_interface.use_chunked_loss）。
        Trainer.validate 传 False，于是 no_grad 下也走分块 CE——Qwen3-VL 的 V=151936，
        B=2/T=8192 一次性物化约 20 GB，而验证侧只需要一个标量 loss。"""
        embed = self._embed_tokens()
        inputs_embeds = embed(input_ids)                              # (B,T,d)
        mask = (input_ids == self.video_token_id)                    # (B,T) 占位符
        inputs_embeds = self._scatter_visual(inputs_embeds, mask, visual_states, visual_mask)

        text_model = self._text_model()
        hidden = text_model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask).last_hidden_state
        # 训练走分块 CE：不物化 (B,T,151936) 的 logits（B=2/T=8192 时那是 29.9GB，
        # 40G 卡上加权重就已 38.4GB）。推理/验证保持原路径——那里 logits 有消费者
        # （eval_benchmark 的 MCQ 打分），且 no_grad 下没有反向图，峰值低得多。
        if use_chunked_loss(self.loss_chunk, labels, need_logits):
            return {"loss": chunked_lm_loss(hidden, labels, self._lm_head(),
                                            self.loss_chunk)}
        logits = self._lm_head()(hidden)
        out = {"logits": logits}
        if labels is not None:
            sl = logits[:, :-1].reshape(-1, logits.size(-1))
            sll = labels[:, 1:].reshape(-1)
            out["loss"] = F.cross_entropy(sl, sll, ignore_index=-100)
        return out


# ============================ 视觉端（离线特征抽取）============================
class Qwen3VLVisionFeatureExtractor:
    """用 Qwen3-VL 视觉塔离线抽取帧级特征，写入 .npz 特征缓存（供 VideoTemporalDataset 读取）。

    推荐用法（真实按帧抽取，与 data_pipeline 的自适应变步长采样对齐）：
        ex = Qwen3VLVisionFeatureExtractor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
        # frame_idx: 采样器选出的原视频帧索引；ts: 对应真实秒级时间戳（严格单调）
        feats, ts = ex.encode_video_at("video.mp4", frame_idx, ts, target_patches=9)  # feats:[L,P,d]
        np.savez(out, features=feats.astype('float16'), timestamps=ts)

    关键设计：**每个采样帧当作独立"图像"预处理**，使视觉塔的 temporal_patch_size(=2) 不会把两
    个非均匀采样的相邻帧合并成一个时间栅格——这样每帧独立成一个时间戳，真实可变 Δt 得以保真
    （这正是 CST-SSM 区别于"喂 Δt 给 Mamba"的物理前提）。encode_video(均匀 fps) 仅作 stand-in 保留。
    """

    # 每帧编码 token 数上限（合并后口径）。默认 256 ≈ 512×512 输入，足够池化到 P<=64；
    # 不设限时 image_processor 会按原分辨率给出数百 token/帧，而下游只保留 P 个 → 算力全浪费。
    DEFAULT_MAX_FRAME_TOKENS = 256

    def __init__(self, hf_model, processor):
        self.hf = hf_model
        self.processor = processor
        self.visual = getattr(getattr(hf_model, "model", hf_model), "visual", None)
        if self.visual is None:
            raise AttributeError("未找到 Qwen3-VL 视觉塔（model.visual）。")
        vc = getattr(getattr(hf_model, "config", None), "vision_config", None)
        self._merge = int(getattr(vc, "spatial_merge_size", 2)) if vc else 2
        self._tp = int(getattr(vc, "temporal_patch_size", 2)) if vc else 2
        self._ps = int(getattr(vc, "patch_size", 16)) if vc else 16
        self.max_frame_tokens = self.DEFAULT_MAX_FRAME_TOKENS

    @classmethod
    def from_pretrained(cls, name: str, **kw):
        Qwen3VLForConditionalGeneration, AutoProcessor = _import_hf()
        model = Qwen3VLForConditionalGeneration.from_pretrained(name, **kw).eval()
        proc = AutoProcessor.from_pretrained(name)
        return cls(model, proc)

    @classmethod
    def from_components(cls, visual, processor, spatial_merge_size: int = 2,
                        temporal_patch_size: int = 2, patch_size: int = 16,
                        max_frame_tokens: int | None = None):
        """绕过 HF 加载，直接注入视觉塔 + processor（用于测试/高级定制）。"""
        self = cls.__new__(cls)
        self.hf = None
        self.processor = processor
        self.visual = visual
        self._merge = int(spatial_merge_size)
        self._tp = int(temporal_patch_size)
        self._ps = int(patch_size)
        self.max_frame_tokens = (cls.DEFAULT_MAX_FRAME_TOKENS if max_frame_tokens is None
                                 else int(max_frame_tokens))
        return self

    # -------------------- 设备/预处理/池化 内部件 --------------------
    def _device(self):
        try:
            return next(self.visual.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _preprocess_frames(self, frames, max_tokens: int | None = None):
        """把 L 帧 RGB（HxWx3 uint8）过 image_processor → (pixel_values, image_grid_thw)。

        以"图像批"路径处理：每帧被 image_processor 内部按 temporal_patch_size 自复制成一个独立
        时间栅格（T=1/帧），故 grid 有 L 行、每行 (1,Hp,Wp)。跨版本用键名兜底。

        max_tokens>0 时给 processor 传 min/max_pixels，把每帧限制到约 max_tokens 个**合并后**
        token。不限制的话 480×864 的帧会编出 405 token/帧，而我们只留 P(=9) 个——97% 的视觉塔
        算力白烧，这是抽取阶段最大的一笔无效开销。老版本 processor 不认这两个 kwarg 时自动退回。
        """
        ip = getattr(self.processor, "image_processor", self.processor)
        kw = {}
        if max_tokens and int(max_tokens) > 0:
            unit = (self._merge * self._ps) ** 2          # 一个合并 token 覆盖的像素数
            mx = max(int(max_tokens), 4) * unit
            cur_min = getattr(ip, "min_pixels", None) or unit
            kw = {"min_pixels": min(int(cur_min), mx), "max_pixels": mx}
        try:
            enc = ip(images=list(frames), return_tensors="pt", **kw)
        except TypeError:                                  # processor 不支持 min/max_pixels
            enc = ip(images=list(frames), return_tensors="pt")
        pv = enc.get("pixel_values", None) if hasattr(enc, "get") else enc["pixel_values"]
        grid = enc.get("image_grid_thw", None) if hasattr(enc, "get") else enc["image_grid_thw"]
        if pv is None or grid is None:
            raise KeyError("image_processor 未返回 pixel_values/image_grid_thw；请核对 transformers 版本")
        return pv, grid

    @staticmethod
    def _factor_grid(P: int) -> tuple[int, int]:
        """把目标 patch 数 P 分解成最接近正方的 (gh,gw)：9→3×3，64→8×8，6→2×3，质数→1×P。"""
        gh = 1
        for k in range(1, int(P ** 0.5) + 1):
            if P % k == 0:
                gh = k
        return gh, P // gh

    @classmethod
    def _pool_tokens(cls, tok: Tensor, target_P: int, hw: tuple[int, int] | None = None) -> Tensor:
        """把单帧的 (n,d) 视觉 token 池化到 (target_P,d)。

        hw=(h',w') 给出该帧**合并后**的 token 网格时走真正的 2D 自适应平均池化：P=9 得到
        3×3 的空间块。此前用 tensor_split 在展平序列上切 P 段，切点不落在行边界上（405=15×27
        切 9 段每段 45，而一行 27），产出的是跨行锯齿条带、没有空间语义——那是个实打实的 bug。
        hw 未知时退回旧的分组均值。
        """
        n, d = tok.shape
        if target_P == 1:
            return tok.mean(0, keepdim=True)
        if hw is not None and hw[0] * hw[1] == n:
            gh, gw = cls._factor_grid(target_P)
            x = tok.view(1, hw[0], hw[1], d).permute(0, 3, 1, 2).float()   # (1,d,h',w')
            y = F.adaptive_avg_pool2d(x, (gh, gw))                         # (1,d,gh,gw)
            return y.permute(0, 2, 3, 1).reshape(gh * gw, d).to(tok.dtype)
        if n == target_P:
            return tok
        if n < target_P:
            reps = (target_P + n - 1) // n
            return tok.repeat(reps, 1)[:target_P]
        groups = torch.tensor_split(tok, target_P, dim=0)
        return torch.stack([g.mean(0) for g in groups], 0)

    def _align_hidden(self, hidden: Tensor, grid_thw: Tensor):
        """把视觉塔输出对齐到**空间合并后**的 token 口径 → (hidden, sizes, hw)。

        不同 transformers 版本里 Qwen3VLVisionModel 的返回约定不一致：有的已经过 patch merger
        （行数 = Σprod/merge²），有的吐 merger 之前的 hidden（行数 = Σprod）。按实际比例判定，
        后者补跑 self.visual.merger——**必须补跑而不是将就着切**，因为 merger 带 LayerNorm+MLP，
        正是投影到 LLM 维度 out_hidden_size 的那一步，跳过它存下来的特征维度和语义都跟 LLM 对不上。
        """
        merge = self._merge
        merge2 = merge * merge
        g = [(int(t), int(h), int(w)) for (t, h, w) in grid_thw.tolist()]
        prods = [t * h * w for (t, h, w) in g]
        total, n = sum(prods), int(hidden.shape[0])
        if n == total // merge2:
            pass                                            # 已合并：预期路径
        elif n == total:
            merger = getattr(self.visual, "merger", None)
            if merger is None:
                raise RuntimeError(
                    f"视觉塔输出 {n} 个 token = grid patch 总数，说明未做空间合并，"
                    f"但 self.visual 上找不到 merger 可补跑；请核对 transformers 版本的 "
                    f"Qwen3VLVisionModel.forward 返回约定")
            hidden = merger(hidden)
            if int(hidden.shape[0]) != total // merge2:
                raise RuntimeError(
                    f"补跑 merger 后 token 数 {int(hidden.shape[0])} 仍不等于期望 "
                    f"{total // merge2}（spatial_merge_size={merge}）")
        else:
            raise ValueError(
                f"token 总数 {n} 既不等于 {total // merge2}（已合并）也不等于 {total}（未合并）；"
                f"spatial_merge_size={merge}，请核对 transformers 版本与视觉塔输出约定")
        sizes = [p // merge2 for p in prods]
        # t>1 的栅格（video 路径）一行含多帧，无法当成单张 2D 图池化 → 该行不给 hw
        hw = [(h // merge, w // merge) if t == 1 else None for (t, h, w) in g]
        return hidden, sizes, hw

    def _pool_all(self, hidden: Tensor, sizes, hw, target_P: int) -> Tensor:
        """按 sizes 把 (total_tokens,d) 切成逐帧段并各自池化 → (L,target_P,d)。"""
        out, off = [], 0
        for s, g in zip(sizes, hw):
            out.append(self._pool_tokens(hidden[off:off + s], target_P, g))
            off += s
        return torch.stack(out, 0)

    def _segment_and_pool(self, hidden: Tensor, grid_thw: Tensor, target_P: int) -> Tensor:
        """对齐 + 逐帧池化的合并入口 → (L,target_P,d)。兼容 Qwen 动态分辨率（逐帧 Hp/Wp 可不同）。"""
        hidden, sizes, hw = self._align_hidden(hidden, grid_thw)
        return self._pool_all(hidden, sizes, hw, target_P)

    def _run_visual(self, pv: Tensor, grid: Tensor) -> Tensor:
        dev = self._device()
        hidden = self.visual(pv.to(dev), grid_thw=grid.to(dev))
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        return getattr(hidden, "last_hidden_state", hidden)

    # ----------------------------- 公开 API -----------------------------
    @torch.no_grad()
    def encode_frames(self, frames_rgb, timestamps, target_patches: int | None = None,
                      max_tokens_per_frame: int | None = None):
        """L 帧 RGB + 对应真实时间戳 → 帧级特征 [L,P,d] 与 ts[L]（ts 原样返回，不重算）。"""
        import numpy as np
        frames = [np.asarray(f) for f in frames_rgb]
        ts = np.asarray(timestamps, dtype=np.float32)
        if len(frames) != len(ts):
            raise ValueError(f"帧数 {len(frames)} 与时间戳数 {len(ts)} 必须一致")
        budget = self.max_frame_tokens if max_tokens_per_frame is None else max_tokens_per_frame
        pv, grid = self._preprocess_frames(frames, budget)
        hidden = self._run_visual(pv, grid)
        hidden, sizes, hw = self._align_hidden(hidden, grid)
        # 默认 P 取各帧最小 token 数（而非 grid[0]）：动态分辨率下逐帧 Hp/Wp 可不同，
        # 按首帧定 P 会让其余帧被拉伸/截断到一个它们并不具备的分辨率。
        P = int(target_patches) if target_patches else int(min(sizes))
        feats = self._pool_all(hidden, sizes, hw, P)             # (L,P,d)
        # 先 .cpu() 再 .float()：bf16 下 D2H 传输量减半，也免掉 GPU 上那份 float32 副本
        return feats.cpu().float().numpy(), ts

    @staticmethod
    def make_reader(video_path: str):
        """建一个持有 decord.VideoReader 的可复用 reader(video_path, frame_idx)->list[HxWx3]。

        分块编码时必须复用：否则每个 chunk 都重新打开视频、重建帧索引（512 帧 / chunk 64 = 8 次）。
        """
        import numpy as np  # noqa: F401  (decord 的 asnumpy 依赖)
        try:
            import decord  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("按 frame_idx 精确取帧需要 decord：pip install decord。原始错误: " + str(e))
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)

        def _read(_path, frame_idx):
            batch = vr.get_batch([int(i) for i in frame_idx]).asnumpy()   # (L,H,W,3) uint8
            return [batch[i] for i in range(batch.shape[0])]
        return _read

    @classmethod
    def _decode_frames(cls, video_path: str, frame_idx):
        """用 decord 按帧索引**精确随机访问**解码 RGB 帧（只读需要的帧，不整段解码）。"""
        return cls.make_reader(video_path)(video_path, frame_idx)

    @torch.no_grad()
    def encode_video_at(self, video_path: str, frame_idx, timestamps,
                        target_patches: int | None = None, reader=None,
                        max_tokens_per_frame: int | None = None):
        """**推荐入口**：按采样器给的 frame_idx 精确取帧过视觉塔。

        reader 可注入自定义取帧函数 reader(video_path, frame_idx)->list[HxWx3 uint8]
        （测试用，或分块编码时用 make_reader 复用同一个 VideoReader；默认走 decord）。
        返回 feats[L,P,d], ts[L]，L==len(frame_idx)，ts==timestamps。
        """
        frames = (reader(video_path, frame_idx) if reader is not None
                  else self._decode_frames(video_path, frame_idx))
        return self.encode_frames(frames, timestamps, target_patches, max_tokens_per_frame)

    @torch.no_grad()
    def encode_video(self, video_path: str, fps: float = 2.0):
        """[stand-in] 均匀 fps 采样抽特征；时间戳按 tp/fps 推算（等间隔）。

        仅用于快速试跑；正式抽取请用 encode_video_at 配合自适应变步长采样，以获得真实可变 Δt。
        """
        import numpy as np
        messages = [{"role": "user", "content": [{"type": "video", "video": video_path}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", video_fps=fps)
        pv = inputs["pixel_values_videos"]
        grid = inputs["video_grid_thw"]                          # (num_videos,3)=(T,H,W)
        hidden = self._run_visual(pv, grid)
        hidden, _, _ = self._align_hidden(hidden, grid)          # 与 encode_frames 同一套对齐
        T, H, W = [int(x) for x in grid[0]]
        P = (H // self._merge) * (W // self._merge)
        d = hidden.shape[-1]
        feats = hidden[: T * P].reshape(T, P, d).cpu().float().numpy()
        ts = (np.arange(T) * self._tp / float(fps)).astype("float32")
        return feats, ts


# ============================ 工厂 ============================
def build_cstssm_qwen3vl(model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
                         d_model: int = 768,
                         branches=None,
                         gate_init_eps: float = 0.1,
                         eacs_chunk: int = 32,
                         lora: bool = True,
                         lora_r: int = 16, lora_alpha: int = 32,
                         dtype=None, base_cfg=None, grad_checkpoint: bool | None = None,
                         device_map=None,
                         **hf_kwargs):
    """构建"Qwen3-VL 视觉塔特征 + CST-SSM 时序 + Qwen3-VL LLM"的端到端模型。

    返回 CSTSSMModel（feature 模式，feat_dim=视觉塔输出维；LLM 段替换为 Qwen3-VL）。
    数据侧需提供 Qwen3-VL 视觉特征 .npz + prompt 内含 Lv 个 video 占位符（Lv=帧数）。

    base_cfg: 可选的完整 CSTSSMConfig。给了就以它为准，工厂**只接管三个由底座决定的字段**
        （input_mode / feat_dim / llm），其余一律沿用——包括 cpib_distill、diff_kv、
        eacs_gate_kind、eacs_disc_mode、cpi_modulation 等全部消融开关，此时
        d_model / branches / gate_init_eps / eacs_chunk 这几个形参被忽略（值取自 base_cfg）。

        这个参数是必要的：这些开关都在 CSTSSMModel.__init__ 里生效，调用方拿到模型后再改
        cfg 不会重建任何模块。消融脚本此前正是那样写的，导致 --real 路径下各臂完全相同。
        构造完请用 models.assert_config_applied(model, cfg) 核对一次。

    device_map: 交给 HF `from_pretrained` 的设备映射，如 {"": "cuda:3"}。给了就让权重
        **直接落到该卡**，省掉"先在 CPU 物化 8.8GB 再 .to(cuda)"这一趟（8 卡同时做就是
        约 70GB 主机内存尖峰 + 8 份读盘）。

        ⚠ 默认 None（仍走 CPU 加载）**不是保守，是必需**：训练脚本刻意把
        "fork DataLoader worker" 排在 "CUDA 初始化" 之前，而 device_map 会在加载底座时
        就建起 CUDA 上下文。num_workers>0 时 fork 出的子进程会继承这个上下文——本仓库的
        worker 只产出 CPU 张量，实践中不会崩，但那是 PyTorch 明确警告的用法。
        要用请走训练脚本的 --load-to-device，那里带了互锁与告警。

        accelerate 未装或 transformers 版本不认这个 kwarg 时自动退回 CPU 加载路径
        （只影响峰值内存与启动耗时，不影响正确性）。
    """
    from dataclasses import replace
    from ..ops.spectral_init import DEFAULT_BRANCHES
    from ..modules.llm_interface import LLMConfig
    from ..models.cst_ssm_model import CSTSSMModel, CSTSSMConfig
    from ..utils.lora import apply_lora, mark_only_lora_trainable

    Qwen3VLForConditionalGeneration, _ = _import_hf()
    # low_cpu_mem_usage=True：8 卡训练时 8 个进程会**同时**各加载一份底座，默认路径
    # 会先在 CPU 上物化完整 fp32/bf16 权重再搬到 GPU，峰值主机内存 ≈ 8×9GB。
    # 该开关改成按分片流式加载到 meta 设备再填充，峰值降到单份大小量级。
    # 老版本 transformers 不认这个 kwarg 时自动退回（不影响正确性，只是更吃内存）。
    _load_kw = {"low_cpu_mem_usage": True}
    _load_kw.update({"torch_dtype": dtype} if dtype else {})
    if device_map is not None:
        _load_kw["device_map"] = device_map
    _load_kw.update(hf_kwargs)
    hf = _from_pretrained_resilient(Qwen3VLForConditionalGeneration, model_name, _load_kw)
    hidden = hf.config.text_config.hidden_size
    # out_hidden_size 缺失时不能安静地用 QWEN3VL_OUT_HIDDEN=3584 顶上：那是 Qwen3VLConfig
    # 的占位默认值，与实测的 Qwen3-VL-4B=2560 不同。顶上去的后果是模型按 3584 建
    # FeatureAdapter，而特征是 2560——直到第 0 步前向才炸。缺了就说出来。
    feat_dim = getattr(hf.config.vision_config, "out_hidden_size", None)
    if not feat_dim:
        feat_dim = QWEN3VL_OUT_HIDDEN
        print(f"[qwen3vl] ⚠ {model_name} 的 vision_config 没有 out_hidden_size，"
              f"退回兜底常量 {QWEN3VL_OUT_HIDDEN}。该值未必等于此底座视觉塔的真实输出维，"
              f"请核对特征维（抽特征与训练必须同源）")
    # LLMConfig 仅用于 projector 目标维（=Qwen hidden）；实际 LLM 用 Qwen3-VL。
    # grad_checkpoint / loss_chunk 必须从 base_cfg.llm 透传：这里是新建 LLMConfig，
    # 不带过来的话 YAML 里的 model.llm.* 会被这一行悄悄丢掉（改配置无效果）。
    _base_llm = getattr(base_cfg, "llm", None)
    _gc = bool(getattr(_base_llm, "grad_checkpoint", False))
    _lc = int(getattr(_base_llm, "loss_chunk", LLMConfig.loss_chunk))
    if grad_checkpoint is not None:                 # 显式形参优先于 base_cfg
        _gc = bool(grad_checkpoint)
    llm_cfg = LLMConfig(vocab_size=hf.config.text_config.vocab_size, dim=hidden,
                        grad_checkpoint=_gc, loss_chunk=_lc)

    if base_cfg is not None:
        cfg = replace(base_cfg, input_mode="feature", feat_dim=feat_dim, llm=llm_cfg)
    else:
        cfg = CSTSSMConfig(
            input_mode="feature", feat_dim=feat_dim, d_model=d_model,
            branches=branches or DEFAULT_BRANCHES, gate_init_eps=gate_init_eps,
            eacs_chunk=eacs_chunk, llm=llm_cfg)

    qwen_llm = Qwen3VLLanguageModel(hf, train_base=not lora, grad_checkpoint=_gc,
                                    loss_chunk=_lc)
    model = CSTSSMModel(cfg, llm=qwen_llm)               # 视觉走特征缓存，故不注入 vision

    if lora:
        # 只给 Qwen3 解码器挂 LoRA；CST-SSM 时序/投影小参数默认可训
        apply_lora(model.llm, targets=QWEN3_LORA_TARGETS, r=lora_r, alpha=lora_alpha)
        mark_only_lora_trainable(model)
    return model
