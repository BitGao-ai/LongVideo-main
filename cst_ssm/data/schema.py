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
    # 帧数（元数据）：长度分桶采样器用它避免逐个探测特征文件头。缺省时会回退到探测。
    n_frames: Optional[int] = None
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
    max_frames: int = 8192               # 单样本最大帧数（超出按均匀/事件抽样）
    # 必须 ≥ max_frames + prompt 文本长度：prompt 里每帧展开一个 video 占位符，
    # 不够会在 build_lm_example 里被 [:max_len] 截掉尾部帧（静默丢视觉信息）。
    # 8704 = 8192 帧 + 512 文本余量。
    max_text_len: int = 8704
    feat_patches: int = 1                # 每帧 patch 数 P
    feat_dim: int = 768
    # 特征张量的 dtype。"auto"（默认）= **沿用特征文件自身的 dtype**。
    # 特征在磁盘上是 float16（extract_features.py 的 astype(np.float16)），此前这里取出后
    # 立刻 .float() 升到 float32——而可训练路径的第一个算子是 autocast(bf16) 下的 Linear，
    # 无论如何都会把它降到 bf16，那份 float32 全程没有消费者，只是让 worker RSS / pinned
    # buffer / H2D 带宽 / GPU 常驻 batch 全部翻倍（实测占整步留存 21%，B=2/L=8192 约 755MB）。
    # 数值零代价：float16→float32 无损，故 fp32→bf16 与 fp16→bf16 逐位相同（已实测）。
    # 不带 autocast 的路径（评测/推理脚本）由 FeatureAdapter 入口处升精度兜底，结果与旧行为
    # 逐位相同。显式写 "float32" 可恢复旧的"取出即升精度"行为。
    feat_dtype: str = "auto"             # auto | float16 | bfloat16 | float32
    frame_size: int = 224                # pixel 模式帧分辨率
    pad_frame_value: float = 0.0
