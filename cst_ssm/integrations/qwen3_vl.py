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


# ============================ 语言端 ============================
class Qwen3VLLanguageModel(nn.Module):
    """包装 Qwen3-VL LLM：把 CST-SSM 压缩视觉状态注入 video 占位符位置，返回 {logits, loss}。

    forward 签名与 modules.llm_interface.VisualConditionedLM 完全一致，可直接作为 CSTSSMModel 的 llm 注入。
    """

    def __init__(self, hf_model, video_token_id: int | None = None,
                 train_base: bool = False):
        super().__init__()
        self.hf = hf_model
        # 从 config 读 video_token_id（不同 Qwen3-VL 大小/变体更稳健），回退常量
        self.video_token_id = (video_token_id if video_token_id is not None
                               else getattr(hf_model.config, "video_token_id", VIDEO_TOKEN_ID))
        self.hidden = hf_model.config.text_config.hidden_size
        if not train_base:                              # 默认冻结基座（配合 LoRA）
            for p in self.hf.parameters():
                p.requires_grad_(False)

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
        """
        B, T, d = inputs_embeds.shape
        if visual_mask is not None:
            vis = visual_states[visual_mask]             # (n_valid, d)
        else:
            vis = visual_states.reshape(-1, d)
        n_slots = int(placeholder_mask.sum().item())
        n_vis = vis.shape[0]
        if n_slots == 0:
            return inputs_embeds                          # 无占位符：纯文本
        if n_vis < n_slots:                               # 数量不足则右侧补零对齐（稳健兜底）
            vis = torch.cat([vis, vis.new_zeros(n_slots - n_vis, d)], 0)
        vis = vis[:n_slots].to(inputs_embeds.dtype)
        out = inputs_embeds.clone()
        out[placeholder_mask] = vis
        return out

    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None,
                visual_mask: Tensor | None = None,
                labels: Tensor | None = None) -> dict:
        embed = self._embed_tokens()
        inputs_embeds = embed(input_ids)                              # (B,T,d)
        mask = (input_ids == self.video_token_id)                    # (B,T) 占位符
        inputs_embeds = self._scatter_visual(inputs_embeds, mask, visual_states, visual_mask)

        text_model = self._text_model()
        hidden = text_model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask).last_hidden_state
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

    def __init__(self, hf_model, processor):
        self.hf = hf_model
        self.processor = processor
        self.visual = getattr(getattr(hf_model, "model", hf_model), "visual", None)
        if self.visual is None:
            raise AttributeError("未找到 Qwen3-VL 视觉塔（model.visual）。")
        vc = getattr(getattr(hf_model, "config", None), "vision_config", None)
        self._merge = int(getattr(vc, "spatial_merge_size", 2)) if vc else 2
        self._tp = int(getattr(vc, "temporal_patch_size", 2)) if vc else 2

    @classmethod
    def from_pretrained(cls, name: str, **kw):
        Qwen3VLForConditionalGeneration, AutoProcessor = _import_hf()
        model = Qwen3VLForConditionalGeneration.from_pretrained(name, **kw).eval()
        proc = AutoProcessor.from_pretrained(name)
        return cls(model, proc)

    @classmethod
    def from_components(cls, visual, processor, spatial_merge_size: int = 2,
                        temporal_patch_size: int = 2):
        """绕过 HF 加载，直接注入视觉塔 + processor（用于测试/高级定制）。"""
        self = cls.__new__(cls)
        self.hf = None
        self.processor = processor
        self.visual = visual
        self._merge = int(spatial_merge_size)
        self._tp = int(temporal_patch_size)
        return self

    # -------------------- 设备/预处理/池化 内部件 --------------------
    def _device(self):
        try:
            return next(self.visual.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _preprocess_frames(self, frames):
        """把 L 帧 RGB（HxWx3 uint8）过 image_processor → (pixel_values, image_grid_thw)。

        以"图像批"路径处理：每帧被 image_processor 内部按 temporal_patch_size 自复制成一个独立
        时间栅格（T=1/帧），故 grid 有 L 行、每行 (1,Hp,Wp)。跨版本用键名兜底。
        """
        ip = getattr(self.processor, "image_processor", self.processor)
        enc = ip(images=list(frames), return_tensors="pt")
        pv = enc.get("pixel_values", None) if hasattr(enc, "get") else enc["pixel_values"]
        grid = enc.get("image_grid_thw", None) if hasattr(enc, "get") else enc["image_grid_thw"]
        if pv is None or grid is None:
            raise KeyError("image_processor 未返回 pixel_values/image_grid_thw；请核对 transformers 版本")
        return pv, grid

    @staticmethod
    def _pool_tokens(tok: Tensor, target_P: int) -> Tensor:
        """把单帧的 (n,d) 视觉 token 池化到 (target_P,d)。
        target_P==1 → 全局均值；n==P → 原样；n<P → 循环补齐；n>P → 均匀分组均值（近似空间池化）。"""
        n, d = tok.shape
        if target_P == 1:
            return tok.mean(0, keepdim=True)
        if n == target_P:
            return tok
        if n < target_P:
            reps = (target_P + n - 1) // n
            return tok.repeat(reps, 1)[:target_P]
        groups = torch.tensor_split(tok, target_P, dim=0)
        return torch.stack([g.mean(0) for g in groups], 0)

    def _segment_and_pool(self, hidden: Tensor, grid_thw: Tensor, target_P: int) -> Tensor:
        """按 grid 把 (total_tokens,d) 切成逐帧段并各自池化 → (L,target_P,d)。

        每帧 token 数 = prod(t,h,w)//merge²（视觉塔空间合并后）。兼容 Qwen 动态分辨率
        （逐帧 Hp/Wp 可不同），这是相比固定 reshape 的关键稳健性。
        """
        merge2 = self._merge * self._merge
        sizes = [int(t) * int(h) * int(w) // merge2 for (t, h, w) in grid_thw.tolist()]
        if sum(sizes) != int(hidden.shape[0]):
            raise ValueError(f"token 总数 {int(hidden.shape[0])} 与 grid 推断 {sum(sizes)} 不一致；"
                             f"检查 spatial_merge_size(={self._merge}) 与视觉塔输出约定")
        out, off = [], 0
        for s in sizes:
            out.append(self._pool_tokens(hidden[off:off + s], target_P))
            off += s
        return torch.stack(out, 0)

    def _run_visual(self, pv: Tensor, grid: Tensor) -> Tensor:
        dev = self._device()
        hidden = self.visual(pv.to(dev), grid_thw=grid.to(dev))
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        return getattr(hidden, "last_hidden_state", hidden)

    # ----------------------------- 公开 API -----------------------------
    @torch.no_grad()
    def encode_frames(self, frames_rgb, timestamps, target_patches: int | None = None):
        """L 帧 RGB + 对应真实时间戳 → 帧级特征 [L,P,d] 与 ts[L]（ts 原样返回，不重算）。"""
        import numpy as np
        frames = [np.asarray(f) for f in frames_rgb]
        ts = np.asarray(timestamps, dtype=np.float32)
        if len(frames) != len(ts):
            raise ValueError(f"帧数 {len(frames)} 与时间戳数 {len(ts)} 必须一致")
        pv, grid = self._preprocess_frames(frames)
        hidden = self._run_visual(pv, grid)
        P = int(target_patches) if target_patches else int(
            (int(grid[0][1]) * int(grid[0][2])) // (self._merge * self._merge))
        feats = self._segment_and_pool(hidden, grid, P)          # (L,P,d)
        return feats.float().cpu().numpy(), ts

    @staticmethod
    def _decode_frames(video_path: str, frame_idx):
        """用 decord 按帧索引**精确随机访问**解码 RGB 帧（只读需要的帧，不整段解码）。"""
        import numpy as np
        try:
            import decord  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("按 frame_idx 精确取帧需要 decord：pip install decord。原始错误: " + str(e))
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        idx = [int(i) for i in frame_idx]
        batch = vr.get_batch(idx).asnumpy()                      # (L,H,W,3) uint8
        return [batch[i] for i in range(batch.shape[0])]

    @torch.no_grad()
    def encode_video_at(self, video_path: str, frame_idx, timestamps,
                        target_patches: int | None = None, reader=None):
        """**推荐入口**：按采样器给的 frame_idx 精确取帧过视觉塔。

        reader 可注入自定义取帧函数 reader(video_path, frame_idx)->list[HxWx3 uint8]
        （测试用；默认走 decord）。返回 feats[L,P,d], ts[L]，L==len(frame_idx)，ts==timestamps。
        """
        frames = (reader(video_path, frame_idx) if reader is not None
                  else self._decode_frames(video_path, frame_idx))
        return self.encode_frames(frames, timestamps, target_patches)

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
        T, H, W = [int(x) for x in grid[0]]
        P = (H // self._merge) * (W // self._merge)
        d = hidden.shape[-1]
        feats = hidden[: T * P].reshape(T, P, d).float().cpu().numpy()
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
                         dtype=None, **hf_kwargs):
    """构建"Qwen3-VL 视觉塔特征 + CST-SSM 时序 + Qwen3-VL LLM"的端到端模型。

    返回 CSTSSMModel（feature 模式，feat_dim=视觉塔输出维；LLM 段替换为 Qwen3-VL）。
    数据侧需提供 Qwen3-VL 视觉特征 .npz + prompt 内含 Lv 个 video 占位符（Lv=帧数）。
    """
    from ..ops.spectral_init import DEFAULT_BRANCHES
    from ..modules.llm_interface import LLMConfig
    from ..models.cst_ssm_model import CSTSSMModel, CSTSSMConfig
    from ..utils.lora import apply_lora, mark_only_lora_trainable

    Qwen3VLForConditionalGeneration, _ = _import_hf()
    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, **({"torch_dtype": dtype} if dtype else {}), **hf_kwargs)
    hidden = hf.config.text_config.hidden_size
    feat_dim = getattr(hf.config.vision_config, "out_hidden_size", QWEN3VL_OUT_HIDDEN)

    cfg = CSTSSMConfig(
        input_mode="feature", feat_dim=feat_dim, d_model=d_model,
        branches=branches or DEFAULT_BRANCHES, gate_init_eps=gate_init_eps,
        eacs_chunk=eacs_chunk,
        # LLMConfig 仅用于 projector 目标维（=Qwen hidden）；实际 LLM 用 Qwen3-VL
        llm=LLMConfig(vocab_size=hf.config.text_config.vocab_size, dim=hidden),
    )
    qwen_llm = Qwen3VLLanguageModel(hf, train_base=not lora)   # video_token_id 从 hf.config 读
    model = CSTSSMModel(cfg, llm=qwen_llm)               # 视觉走特征缓存，故不注入 vision

    if lora:
        # 只给 Qwen3 解码器挂 LoRA；CST-SSM 时序/投影小参数默认可训
        apply_lora(model.llm, targets=QWEN3_LORA_TARGETS, r=lora_r, alpha=lora_alpha)
        mark_only_lora_trainable(model)
    return model
