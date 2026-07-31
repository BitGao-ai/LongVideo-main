"""数据层：schema、数据集、collate、加载工厂。"""
from __future__ import annotations

from .schema import VideoSample, DataConfig
from .dataset import (ByteTokenizer, VideoTemporalDataset, SyntheticVideoDataset,
                      SyntheticGroundingDataset)
from .collate import collate_fn, make_loader
from .loaders import LoaderConfig, build_dataset, build_dataloader, cycle, infer_feat_dim

__all__ = [
    "VideoSample", "DataConfig",
    "ByteTokenizer", "VideoTemporalDataset", "SyntheticVideoDataset", "SyntheticGroundingDataset",
    "collate_fn", "make_loader",
    "LoaderConfig", "build_dataset", "build_dataloader", "cycle", "infer_feat_dim",
]
