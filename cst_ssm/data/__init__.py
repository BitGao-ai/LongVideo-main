"""数据层：schema、数据集、collate、加载工厂、分词器。"""
from __future__ import annotations

from .schema import VideoSample, DataConfig
from .dataset import (ByteTokenizer, VideoTemporalDataset, SyntheticVideoDataset,
                      SyntheticGroundingDataset)
from .collate import collate_fn, make_loader
from .loaders import (LoaderConfig, build_dataset, build_dataloader, cycle,
                      infer_feat_dim, align_feat_dim)
from .hf_tokenizer import HFTokenizer, build_hf_tokenizer

__all__ = [
    "VideoSample", "DataConfig",
    "ByteTokenizer", "VideoTemporalDataset", "SyntheticVideoDataset", "SyntheticGroundingDataset",
    "collate_fn", "make_loader",
    "LoaderConfig", "build_dataset", "build_dataloader", "cycle", "infer_feat_dim",
    "align_feat_dim",
    "HFTokenizer", "build_hf_tokenizer",
]
