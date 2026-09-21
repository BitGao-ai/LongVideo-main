"""Qwen3-VL integration: offline visual features plus wrapped Qwen3 LLM."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..modules.llm_interface import chunked_lm_loss, use_chunked_loss

QWEN3_LORA_TARGETS = frozenset({
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
})
VIDEO_TOKEN_ID = 151656
IMAGE_TOKEN_ID = 151655
QWEN3VL_OUT_HIDDEN = 3584


def _import_hf():
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        return Qwen3VLForConditionalGeneration, AutoProcessor
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Qwen3-VL integration requires transformers with Qwen3-VL: "
            f"pip install -U transformers.\nOriginal error: {e}")


_OPTIONAL_LOAD_KWARGS = ("device_map", "low_cpu_mem_usage")


def _from_pretrained_resilient(cls, model_name: str, load_kw: dict):
    """from_pretrained with graceful fallback for unsupported optional kwargs."""
    kw = dict(load_kw)
    while True:
        try:
            return cls.from_pretrained(model_name, **kw)
        except (TypeError, ImportError, ValueError) as e:
            dropped = next((k for k in _OPTIONAL_LOAD_KWARGS if k in kw), None)
            if dropped is None:
                raise
            kw.pop(dropped)
            print(f"[qwen3vl] from_pretrained kwarg {dropped} unsupported "
                  f"({type(e).__name__}: {str(e)[:120]}); retrying without it.")


class Qwen3VLLanguageModel(nn.Module):
    """Wraps the Qwen3-VL LLM; injects compressed visual states at video tokens.

    Forward signature matches VisualConditionedLM, so it plugs into CSTSSMModel directly.
    """

    def __init__(self, hf_model, video_token_id: int | None = None,
                 train_base: bool = False, grad_checkpoint: bool = False,
                 loss_chunk: int = 1024):
        super().__init__()
        self.hf = hf_model
        self.video_token_id = (video_token_id if video_token_id is not None
                               else getattr(hf_model.config, "video_token_id", VIDEO_TOKEN_ID))
        self.hidden = hf_model.config.text_config.hidden_size
        if not train_base:
            for p in self.hf.parameters():
                p.requires_grad_(False)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.loss_chunk = int(loss_chunk)
        if self.grad_checkpoint:
            self._enable_grad_checkpoint()

    def _enable_grad_checkpoint(self) -> None:
        """Enable non-reentrant gradient checkpointing and disable KV cache."""
        cfg = getattr(self.hf, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "use_cache"):
            text_cfg.use_cache = False
        enable = getattr(self.hf, "gradient_checkpointing_enable", None)
        if enable is None:
            raise AttributeError(
                "HF model lacks gradient_checkpointing_enable; upgrade transformers "
                "or set llm.grad_checkpoint to false")
        try:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            raise RuntimeError(
                "transformers gradient_checkpointing_enable lacks use_reentrant=False; "
                "upgrade transformers or disable llm.grad_checkpoint")

    def _embed_tokens(self):
        return self.hf.get_input_embeddings()

    def _text_model(self):
        base = getattr(self.hf, "model", self.hf)
        return getattr(base, "language_model", base)

    def _lm_head(self):
        return self.hf.get_output_embeddings()

    @staticmethod
    def _scatter_visual(inputs_embeds: Tensor, placeholder_mask: Tensor,
                        visual_states: Tensor, visual_mask: Tensor | None) -> Tensor:
        """Write visual states into placeholder positions in row-major order."""
        B, T, d = inputs_embeds.shape
        if visual_mask is not None:
            vis = visual_states[visual_mask]
            n_vis_per = visual_mask.sum(dim=1)
        else:
            vis = visual_states.reshape(-1, d)
            n_vis_per = torch.full((B,), visual_states.shape[1],
                                   device=inputs_embeds.device, dtype=torch.long)
        n_slot_per = placeholder_mask.sum(dim=1)
        n_slots = int(n_slot_per.sum().item())
        if n_slots == 0:
            return inputs_embeds
        if not bool(torch.equal(n_vis_per, n_slot_per.to(n_vis_per.dtype))):
            bad = (n_vis_per != n_slot_per).nonzero(as_tuple=True)[0].tolist()
            ns, nv = n_slot_per.cpu().tolist(), n_vis_per.cpu().tolist()
            detail = ", ".join(f"sample{i}: slots={ns[i]} vs states={nv[i]}" for i in bad[:8])
            raise ValueError(
                "[Qwen3VL] per-sample video placeholder count != valid visual states "
                f"({detail}). Check frame counts and max_video_tokens truncation.")
        if vis.shape[0] < n_slots:
            vis = torch.cat([vis, vis.new_zeros(n_slots - vis.shape[0], d)], 0)
        vis = vis[:n_slots].to(inputs_embeds.dtype)
        return inputs_embeds.masked_scatter(placeholder_mask.unsqueeze(-1), vis)

    def forward(self, input_ids: Tensor, visual_states: Tensor,
                attention_mask: Tensor | None = None,
                visual_mask: Tensor | None = None,
                labels: Tensor | None = None,
                need_logits: bool = True) -> dict:
        """need_logits False keeps validation on the chunked path without logits."""
        embed = self._embed_tokens()
        inputs_embeds = embed(input_ids)
        mask = (input_ids == self.video_token_id)
        inputs_embeds = self._scatter_visual(inputs_embeds, mask, visual_states, visual_mask)

        text_model = self._text_model()
        hidden = text_model(inputs_embeds=inputs_embeds,
                            attention_mask=attention_mask).last_hidden_state
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


class Qwen3VLVisionFeatureExtractor:
    """Offline frame-feature extraction with the Qwen3-VL visual tower.

    Each sampled frame is preprocessed as an independent image so the visual
    tower's temporal patching never merges non-uniformly sampled frames.
    """

    DEFAULT_MAX_FRAME_TOKENS = 256

    def __init__(self, hf_model, processor):
        self.hf = hf_model
        self.processor = processor
        self.visual = getattr(getattr(hf_model, "model", hf_model), "visual", None)
        if self.visual is None:
            raise AttributeError("Qwen3-VL visual tower (model.visual) not found.")
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
        """Inject a visual tower and processor directly (testing/advanced use)."""
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

    def _device(self):
        try:
            return next(self.visual.parameters()).device
        except Exception:
            return torch.device("cpu")

    def _preprocess_frames(self, frames, max_tokens: int | None = None):
        """Preprocess frames as an image batch; returns (pixel_values, image_grid_thw)."""
        ip = getattr(self.processor, "image_processor", self.processor)
        kw = {}
        if max_tokens and int(max_tokens) > 0:
            unit = (self._merge * self._ps) ** 2
            mx = max(int(max_tokens), 4) * unit
            cur_min = getattr(ip, "min_pixels", None) or unit
            kw = {"min_pixels": min(int(cur_min), mx), "max_pixels": mx}
        try:
            enc = ip(images=list(frames), return_tensors="pt", **kw)
        except TypeError:
            enc = ip(images=list(frames), return_tensors="pt")
        pv = enc.get("pixel_values", None) if hasattr(enc, "get") else enc["pixel_values"]
        grid = enc.get("image_grid_thw", None) if hasattr(enc, "get") else enc["image_grid_thw"]
        if pv is None or grid is None:
            raise KeyError("image_processor did not return pixel_values/image_grid_thw")
        return pv, grid

    @staticmethod
    def _factor_grid(P: int) -> tuple[int, int]:
        """Factor target patch count P into a near-square (gh, gw) grid."""
        gh = 1
        for k in range(1, int(P ** 0.5) + 1):
            if P % k == 0:
                gh = k
        return gh, P // gh

    @classmethod
    def _pool_tokens(cls, tok: Tensor, target_P: int, hw: tuple[int, int] | None = None) -> Tensor:
        """Pool one frame's (n,d) tokens to (target_P,d) with 2D pooling when possible."""
        n, d = tok.shape
        if target_P == 1:
            return tok.mean(0, keepdim=True)
        if hw is not None and hw[0] * hw[1] == n:
            gh, gw = cls._factor_grid(target_P)
            x = tok.view(1, hw[0], hw[1], d).permute(0, 3, 1, 2).float()
            y = F.adaptive_avg_pool2d(x, (gh, gw))
            return y.permute(0, 2, 3, 1).reshape(gh * gw, d).to(tok.dtype)
        if n == target_P:
            return tok
        if n < target_P:
            reps = (target_P + n - 1) // n
            return tok.repeat(reps, 1)[:target_P]
        groups = torch.tensor_split(tok, target_P, dim=0)
        return torch.stack([g.mean(0) for g in groups], 0)

    def _align_hidden(self, hidden: Tensor, grid_thw: Tensor):
        """Align visual outputs to post-merge token counts; runs merger when needed."""
        merge = self._merge
        merge2 = merge * merge
        g = [(int(t), int(h), int(w)) for (t, h, w) in grid_thw.tolist()]
        prods = [t * h * w for (t, h, w) in g]
        total, n = sum(prods), int(hidden.shape[0])
        if n == total // merge2:
            pass
        elif n == total:
            merger = getattr(self.visual, "merger", None)
            if merger is None:
                raise RuntimeError(
                    f"visual output has {n} tokens = grid patch total; merger unavailable")
            hidden = merger(hidden)
            if int(hidden.shape[0]) != total // merge2:
                raise RuntimeError(
                    f"post-merger token count {int(hidden.shape[0])} != expected "
                    f"{total // merge2} (spatial_merge_size={merge})")
        else:
            raise ValueError(
                f"token total {n} matches neither merged ({total // merge2}) nor unmerged "
                f"({total}) counts; spatial_merge_size={merge}")
        sizes = [p // merge2 for p in prods]
        hw = [(h // merge, w // merge) if t == 1 else None for (t, h, w) in g]
        return hidden, sizes, hw

    def _pool_all(self, hidden: Tensor, sizes, hw, target_P: int) -> Tensor:
        """Split merged tokens per frame and pool; returns (L,target_P,d)."""
        out, off = [], 0
        for s, g in zip(sizes, hw):
            out.append(self._pool_tokens(hidden[off:off + s], target_P, g))
            off += s
        return torch.stack(out, 0)

    def _segment_and_pool(self, hidden: Tensor, grid_thw: Tensor, target_P: int) -> Tensor:
        hidden, sizes, hw = self._align_hidden(hidden, grid_thw)
        return self._pool_all(hidden, sizes, hw, target_P)

    def _run_visual(self, pv: Tensor, grid: Tensor) -> Tensor:
        dev = self._device()
        hidden = self.visual(pv.to(dev), grid_thw=grid.to(dev))
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]
        return getattr(hidden, "last_hidden_state", hidden)

    @torch.no_grad()
    def encode_frames(self, frames_rgb, timestamps, target_patches: int | None = None,
                      max_tokens_per_frame: int | None = None):
        """Frames plus timestamps to ([L,P,d] features, ts[L])."""
        import numpy as np
        frames = [np.asarray(f) for f in frames_rgb]
        ts = np.asarray(timestamps, dtype=np.float32)
        if len(frames) != len(ts):
            raise ValueError(f"frame count {len(frames)} != timestamp count {len(ts)}")
        budget = self.max_frame_tokens if max_tokens_per_frame is None else max_tokens_per_frame
        pv, grid = self._preprocess_frames(frames, budget)
        hidden = self._run_visual(pv, grid)
        hidden, sizes, hw = self._align_hidden(hidden, grid)
        P = int(target_patches) if target_patches else int(min(sizes))
        feats = self._pool_all(hidden, sizes, hw, P)
        return feats.cpu().float().numpy(), ts

    @staticmethod
    def make_reader(video_path: str):
        """Reusable decord reader: reader(video_path, frame_idx) -> RGB frames."""
        import numpy as np  # noqa: F401
        try:
            import decord  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("decord is required for frame-exact reads: pip install decord. " + str(e))
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)

        def _read(_path, frame_idx):
            batch = vr.get_batch([int(i) for i in frame_idx]).asnumpy()
            return [batch[i] for i in range(batch.shape[0])]
        return _read

    @classmethod
    def _decode_frames(cls, video_path: str, frame_idx):
        return cls.make_reader(video_path)(video_path, frame_idx)

    @torch.no_grad()
    def encode_video_at(self, video_path: str, frame_idx, timestamps,
                        target_patches: int | None = None, reader=None,
                        max_tokens_per_frame: int | None = None):
        """Extract features at sampled frame indices; returns feats[L,P,d], ts[L]."""
        frames = (reader(video_path, frame_idx) if reader is not None
                  else self._decode_frames(video_path, frame_idx))
        return self.encode_frames(frames, timestamps, target_patches, max_tokens_per_frame)

    @torch.no_grad()
    def encode_video(self, video_path: str, fps: float = 2.0):
        """Uniform-fps extraction placeholder; prefer encode_video_at for real timestamps."""
        import numpy as np
        messages = [{"role": "user", "content": [{"type": "video", "video": video_path}]}]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", video_fps=fps)
        pv = inputs["pixel_values_videos"]
        grid = inputs["video_grid_thw"]
        hidden = self._run_visual(pv, grid)
        hidden, _, _ = self._align_hidden(hidden, grid)
        T, H, W = [int(x) for x in grid[0]]
        P = (H // self._merge) * (W // self._merge)
        d = hidden.shape[-1]
        feats = hidden[: T * P].reshape(T, P, d).cpu().float().numpy()
        ts = (np.arange(T) * self._tp / float(fps)).astype("float32")
        return feats, ts


def _release_visual_tower(hf) -> tuple[int, int]:
    """Drop the unused visual tower after loading; returns (n_params, n_bytes)."""
    base = getattr(hf, "model", hf)
    visual = getattr(base, "visual", None)
    if visual is None:
        return 0, 0
    n = sum(p.numel() for p in visual.parameters())
    nbytes = sum(p.numel() * p.element_size() for p in visual.parameters())
    try:
        delattr(base, "visual")
    except AttributeError:
        base.visual = None
    return n, nbytes


def build_cstssm_qwen3vl(model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
                         d_model: int = 768,
                         branches=None,
                         gate_init_eps: float = 0.1,
                         eacs_chunk: int = 32,
                         lora: bool = True,
                         lora_r: int = 16, lora_alpha: int = 32,
                         dtype=None, base_cfg=None, grad_checkpoint: bool | None = None,
                         device_map=None,
                         keep_visual: bool = False,
                         **hf_kwargs):
    """Build the end-to-end model: Qwen3-VL visual features, CST-SSM temporal, Qwen3 LLM.

    Args:
        base_cfg: optional full CSTSSMConfig; only input_mode/feat_dim/llm are overridden.
        device_map: optional HF device map for direct-to-GPU loading.
        keep_visual: keep model.visual when True (feature mode never calls it).
    """
    from dataclasses import replace
    from ..ops.spectral_init import DEFAULT_BRANCHES
    from ..modules.llm_interface import LLMConfig
    from ..models.cst_ssm_model import CSTSSMModel, CSTSSMConfig
    from ..utils.lora import apply_lora, mark_only_lora_trainable

    Qwen3VLForConditionalGeneration, _ = _import_hf()
    _load_kw = {"low_cpu_mem_usage": True}
    _load_kw.update({"torch_dtype": dtype} if dtype else {})
    if device_map is not None:
        _load_kw["device_map"] = device_map
    _load_kw.update(hf_kwargs)
    hf = _from_pretrained_resilient(Qwen3VLForConditionalGeneration, model_name, _load_kw)
    hidden = hf.config.text_config.hidden_size
    feat_dim = getattr(hf.config.vision_config, "out_hidden_size", None)
    if not feat_dim:
        feat_dim = QWEN3VL_OUT_HIDDEN
        print(f"[qwen3vl] {model_name} vision_config lacks out_hidden_size; "
              f"falling back to {QWEN3VL_OUT_HIDDEN}. Verify feature dims match.")
    _base_llm = getattr(base_cfg, "llm", None)
    _gc = bool(getattr(_base_llm, "grad_checkpoint", False))
    _lc = int(getattr(_base_llm, "loss_chunk", LLMConfig.loss_chunk))
    if grad_checkpoint is not None:
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
    model = CSTSSMModel(cfg, llm=qwen_llm)

    if not keep_visual:
        n_rel, b_rel = _release_visual_tower(hf)
        if n_rel:
            print(f"[qwen3vl] released unused visual tower: {n_rel} params, "
                  f"~{b_rel / 1024 ** 2:.0f}MB per rank")

    if lora:
        apply_lora(model.llm, targets=QWEN3_LORA_TARGETS, r=lora_r, alpha=lora_alpha)
        mark_only_lora_trainable(model)
    return model
