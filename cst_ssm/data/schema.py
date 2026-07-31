"""数据 schema：一条样本的字段定义（设计方案 §4，数据架构见 README）。

核心：视频被建模为**带时间戳的帧/特征序列**（真实 Δt 是 CST-SSM 的一等公民）。
两种视觉来源二选一：
  - feature_ref : 预抽取特征文件（.npz，含 features[L,P,d] 与 timestamps[L]）——内存/算力友好，推荐
  - frame_dir   : 帧图目录（pixel 模式，端到端从像素）
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class VideoSample:
    video_id: str
    prompt: str                          # 指令/问题（含 <video> 占位）
    answer: str                          # 目标回答（训练标签）
    task_type: str = "qa"                # qa | grounding | caption | causal
    # 视觉来源（二选一）
    feature_ref: Optional[str] = None    # .npz 路径
    frame_dir: Optional[str] = None      # 帧图目录
    # 时间轴（feature 模式下也可由 .npz 内 timestamps 提供，这里可留空）
    timestamps: Optional[list] = None    # 每帧秒级时间戳
    duration: float = 0.0
    # 定位任务标注（可选，毫秒级）
    gt_start: Optional[float] = None
    gt_end: Optional[float] = None
    split: str = "train"

    def to_json(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DataConfig:
    manifest: str                        # jsonl，每行一个 VideoSample
    mode: str = "feature"                # feature | pixel
    max_frames: int = 512                # 单样本最大帧数（超出按均匀/事件抽样）
    max_text_len: int = 512
    feat_patches: int = 1                # 每帧 patch 数 P
    feat_dim: int = 768
    frame_size: int = 224                # pixel 模式帧分辨率
    pad_frame_value: float = 0.0
