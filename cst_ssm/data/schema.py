"""Data schemas: per-sample fields and dataset config."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class VideoSample:
    video_id: str
    prompt: str
    answer: str
    task_type: str = "qa"                # qa | grounding | caption | causal
    feature_ref: Optional[str] = None    # Pre-extracted .npz path
    frame_dir: Optional[str] = None      # Frame directory (pixel mode)
    n_frames: Optional[int] = None       # Frame count metadata for bucketing
    timestamps: Optional[list] = None
    duration: float = 0.0
    gt_start: Optional[float] = None
    gt_end: Optional[float] = None
    split: str = "train"

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DataConfig:
    manifest: str
    mode: str = "feature"                # feature | pixel
    max_frames: int = 8192
    max_text_len: int = 8704             # Must cover max_frames + prompt text.
    feat_patches: int = 1
    feat_dim: int = 768
    feat_dtype: str = "auto"             # auto | float16 | bfloat16 | float32
    frame_size: int = 224
    pad_frame_value: float = 0.0
