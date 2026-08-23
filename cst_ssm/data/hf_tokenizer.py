"""HuggingFace 分词器适配器：把 Qwen3-VL 的 processor 包装成与 ByteTokenizer 同接口的分词器。

设计要点（对齐 cst_ssm/integrations/qwen3_vl.py 的注入约定）：
  - Qwen3VLLanguageModel._scatter_visual 要求 prompt 内 video 占位符（video_token_id=151656）
    的数量 **== 该样本有效视觉状态数（帧数 Lv）**。本适配器把 prompt 里的单个 `<video>` 标记
    展开成 Lv 个 video_token_id，由 VideoTemporalDataset 在 __getitem__ 里通过 set_video_tokens(Lv)
    告知当前样本帧数（子采样后），从而保证二者严格对齐。
  - build_lm_example 与 ByteTokenizer 完全同签名：返回 (input_ids, labels)，labels 对 prompt 部分
    打 -100（只学答案），供 VisualConditionedLM / Qwen3VLLanguageModel 的 cross-entropy 使用。

依赖：transformers（含 Qwen3-VL）。lazy import，未安装时给出清晰报错，不影响其余模块（ByteTokenizer 兜底）。
"""
from __future__ import annotations

# 与 integrations/qwen3_vl.py 保持一致的关键常量（不同 Qwen3-VL 变体可从 config 覆盖）
VIDEO_PLACEHOLDER = "<video>"
DEFAULT_VIDEO_TOKEN_ID = 151656
DEFAULT_PAD_TOKEN_ID = 151643


def _import_processor():
    try:
        from transformers import AutoProcessor
        return AutoProcessor
    except Exception as e:  # pragma: no cover - 取决于运行环境
        raise ImportError(
            "接入 Qwen3-VL 分词器需要 transformers（含 Qwen3-VL）：pip install -U transformers。"
            f"\n原始错误: {e}")


class HFTokenizer:
    """包装 Qwen3-VL processor 的文本分词器，接口对齐 ByteTokenizer。

    属性 vocab_size / pad_id 供 collate 与 LLMConfig 使用；video_token_id 用于展开 `<video>` 占位符。
    """

    def __init__(self, model_name: str,
                 video_token_id: int | None = None,
                 max_video_tokens: int = 8192):
        AutoProcessor = _import_processor()
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.tok = getattr(self.proc, "tokenizer", self.proc)
        # video_token_id：优先从 processor/tokenizer 的 special tokens 读，回退常量
        self.video_token_id = video_token_id or self._resolve_video_token_id()
        self.max_video_tokens = max_video_tokens
        self._n_video = 1                       # 当前样本帧数（默认1，由 dataset 逐样本设置）
        self._video_placeholder = VIDEO_PLACEHOLDER

    def _resolve_video_token_id(self) -> int:
        """跨版本稳健地取 video 占位符 token id。"""
        # 1) tokenizer 特殊词表
        try:
            vid = self.tok.convert_tokens_to_ids("<|video_pad|>")
            if isinstance(vid, int) and vid >= 0:
                return vid
        except Exception:
            pass
        # 2) 附加特殊 token 映射
        added = getattr(self.tok, "added_tokens_encoder", {}) or {}
        for key in ("<|video_pad|>", "<video>"):
            if key in added:
                return int(added[key])
        return DEFAULT_VIDEO_TOKEN_ID

    @property
    def vocab_size(self) -> int:
        try:
            return int(self.tok.vocab_size) + len(getattr(self.tok, "added_tokens_encoder", {}) or {})
        except Exception:
            return len(self.tok)

    @property
    def pad_id(self) -> int:
        pid = getattr(self.tok, "pad_token_id", None)
        return int(pid) if pid is not None else DEFAULT_PAD_TOKEN_ID

    # ---- 与 ByteTokenizer 对齐的接口 ----
    def set_video_tokens(self, n: int) -> None:
        """告知当前样本的视觉帧数 Lv，使 `<video>` 展开成 Lv 个 video 占位符。"""
        self._n_video = max(1, int(n))

    def encode(self, s: str) -> list:
        return self.tok.encode(s, add_special_tokens=False)

    def build_lm_example(self, prompt: str, answer: str, max_len: int):
        """拼 prompt+answer，labels 对 prompt 部分打 -100（只学答案）。

        prompt 内的单个 `<video>` 会被展开成 self._n_video 个 video_token_id（与帧数对齐）。
        """
        prompt_ids = self._encode_with_video(prompt)
        answer_ids = self.tok.encode(answer, add_special_tokens=False) if answer else []
        # 追加 EOS（若 tokenizer 有）
        eos = getattr(self.tok, "eos_token_id", None)
        if eos is not None:
            answer_ids = answer_ids + [int(eos)]

        ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        return ids, labels

    def _encode_with_video(self, prompt: str) -> list:
        """把 prompt 里的 `<video>` 展开成 _n_video 个 video 占位符 token。"""
        if self._video_placeholder in prompt:
            head, tail = prompt.split(self._video_placeholder, 1)
            head_ids = self.tok.encode(head, add_special_tokens=False) if head.strip() else []
            tail_ids = self.tok.encode(tail, add_special_tokens=False) if tail.strip() else []
            n = min(self._n_video, self.max_video_tokens)
            if n < self._n_video and not getattr(self, "_warned_truncate", False):
                # 占位符数量必须 == 有效视觉状态数（_scatter_visual 约定）；截断后尾部帧
                # 特征将被零填充丢弃——不报错保运行，但必须显式告警，避免静默丢帧
                print(f"[HFTokenizer] 警告: 帧数 {self._n_video} > max_video_tokens="
                      f"{self.max_video_tokens}，video 占位符被截断，尾部 {self._n_video - n} "
                      f"帧视觉特征将被丢弃。请调大 max_video_tokens 或降低采样帧数。")
                self._warned_truncate = True
            return head_ids + [self.video_token_id] * n + tail_ids
        # 无占位符：原样编码（纯文本样本）
        return self.tok.encode(prompt, add_special_tokens=False)


def build_hf_tokenizer(model_name: str, **kw) -> HFTokenizer:
    """工厂：构造 Qwen3-VL 分词器（供 scripts/train_qwen3vl.py 使用）。"""
    return HFTokenizer(model_name, **kw)
