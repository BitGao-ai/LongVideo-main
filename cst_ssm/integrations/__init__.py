"""External model integrations."""
from __future__ import annotations

from .qwen3_vl import (
    Qwen3VLLanguageModel, Qwen3VLVisionFeatureExtractor, build_cstssm_qwen3vl,
    QWEN3_LORA_TARGETS, VIDEO_TOKEN_ID, QWEN3VL_OUT_HIDDEN,
)

__all__ = [
    "Qwen3VLLanguageModel", "Qwen3VLVisionFeatureExtractor", "build_cstssm_qwen3vl",
    "QWEN3_LORA_TARGETS", "VIDEO_TOKEN_ID", "QWEN3VL_OUT_HIDDEN",
]
