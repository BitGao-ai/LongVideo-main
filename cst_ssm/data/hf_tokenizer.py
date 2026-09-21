"""HuggingFace tokenizer adapter with the ByteTokenizer interface."""
from __future__ import annotations

VIDEO_PLACEHOLDER = "<video>"
DEFAULT_VIDEO_TOKEN_ID = 151656
DEFAULT_PAD_TOKEN_ID = 151643


def _import_processor():
    try:
        from transformers import AutoProcessor
        return AutoProcessor
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Qwen3-VL tokenizers require transformers: pip install -U transformers."
            f"\nOriginal error: {e}")


class HFTokenizer:
    """Wraps a Qwen3-VL processor; expands <video> into per-frame placeholder tokens."""

    def __init__(self, model_name: str,
                 video_token_id: int | None = None,
                 max_video_tokens: int = 8192):
        AutoProcessor = _import_processor()
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.tok = getattr(self.proc, "tokenizer", self.proc)
        self.video_token_id = video_token_id or self._resolve_video_token_id()
        self.max_video_tokens = max_video_tokens
        self._n_video = 1
        self._video_placeholder = VIDEO_PLACEHOLDER

    def _resolve_video_token_id(self) -> int:
        try:
            vid = self.tok.convert_tokens_to_ids("<|video_pad|>")
            if isinstance(vid, int) and vid >= 0:
                return vid
        except Exception:
            pass
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

    def set_video_tokens(self, n: int) -> None:
        self._n_video = max(1, int(n))

    def encode(self, s: str) -> list:
        return self.tok.encode(s, add_special_tokens=False)

    def build_lm_example(self, prompt: str, answer: str, max_len: int):
        """Concatenate prompt and answer; labels mask the prompt with -100."""
        prompt_ids = self._encode_with_video(prompt)
        answer_ids = self.tok.encode(answer, add_special_tokens=False) if answer else []
        eos = getattr(self.tok, "eos_token_id", None)
        if eos is not None:
            answer_ids = answer_ids + [int(eos)]

        ids = (prompt_ids + answer_ids)[:max_len]
        labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
        return ids, labels

    def _encode_with_video(self, prompt: str) -> list:
        if self._video_placeholder in prompt:
            head, tail = prompt.split(self._video_placeholder, 1)
            head_ids = self.tok.encode(head, add_special_tokens=False) if head.strip() else []
            tail_ids = self.tok.encode(tail, add_special_tokens=False) if tail.strip() else []
            n = min(self._n_video, self.max_video_tokens)
            if n < self._n_video and not getattr(self, "_warned_truncate", False):
                print(f"[HFTokenizer] frames {self._n_video} > max_video_tokens="
                      f"{self.max_video_tokens}; trailing frames are dropped.")
                self._warned_truncate = True
            return head_ids + [self.video_token_id] * n + tail_ids
        return self.tok.encode(prompt, add_special_tokens=False)


def build_hf_tokenizer(model_name: str, **kw) -> HFTokenizer:
    return HFTokenizer(model_name, **kw)
