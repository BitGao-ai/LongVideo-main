"""统一数据加载工厂：一处构造 loader，训练/评测/合成三态同口径。

替代 train_stage1/2 里各自内联的 `SyntheticVideoDataset + make_loader`，解决三件事：
  1) **真实 manifest 接入**：有 manifest → VideoTemporalDataset（feature/pixel），无 → SyntheticVideoDataset。
  2) **无限取批 `cycle`**：Trainer 按 max_steps 停，但真实 loader 有限；cycle 让 step 训练不会因
     max_steps>len(loader) 提前结束（合成路径原本靠 n=steps*2+4 侥幸够用，脆弱）。
  3) **吞吐/一致性**：num_workers/pin_memory/persistent_workers/drop_last + 特征维自动推断（防 feat_dim 错配）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from .schema import DataConfig
from .dataset import (VideoTemporalDataset, SyntheticVideoDataset,
                      SyntheticGroundingDataset, ByteTokenizer)
from .collate import collate_fn


@dataclass
class LoaderConfig:
    # 真实数据（manifest 非空即启用真实路径）
    manifest: str | None = None
    data_root: str = ""
    mode: str = "feature"                 # feature | pixel
    max_frames: int = 512
    max_text_len: int = 512
    # 批与吞吐
    batch_size: int = 2
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    persistent_workers: bool = False
    pad_id: int = 256
    # 特征维（合成用；真实可 auto-infer 覆盖）
    feat_dim: int = 64
    feat_patches: int = 4
    # 合成兜底
    synth_n: int = 128
    synth_frames: int = 32
    synth_grounding: bool = False         # True：合成定位数据（带 gt_start/gt_end），供 grounding 训练


def infer_feat_dim(manifest: str, data_root: str = "") -> int | None:
    """从 manifest 首个可读特征推断 d（防止 cfg.feat_dim 与真实特征错配）。失败返回 None。"""
    try:
        with open(manifest) as f:
            for line in f:
                if not line.strip():
                    continue
                ref = json.loads(line).get("feature_ref")
                if not ref:
                    continue
                path = os.path.join(data_root, ref)
                if ref.endswith(".npy"):
                    return int(np.load(path, mmap_mode="r").shape[-1])
                if ref.endswith(".npz"):
                    z = np.load(path)
                    try:
                        return int(z["features"].shape[-1])
                    finally:
                        z.close()
    except Exception:
        return None
    return None


def build_dataset(lc: LoaderConfig, tokenizer=None):
    """按配置构造 dataset（真实 manifest 或合成）。"""
    if lc.manifest:
        dcfg = DataConfig(manifest=lc.manifest, mode=lc.mode, max_frames=lc.max_frames,
                          max_text_len=lc.max_text_len, feat_dim=lc.feat_dim,
                          feat_patches=lc.feat_patches)
        return VideoTemporalDataset(dcfg, tokenizer=tokenizer or ByteTokenizer(),
                                    data_root=lc.data_root)
    if lc.synth_grounding:
        return SyntheticGroundingDataset(n=lc.synth_n, L=lc.synth_frames, P=lc.feat_patches,
                                         d=lc.feat_dim)
    return SyntheticVideoDataset(n=lc.synth_n, L=lc.synth_frames, P=lc.feat_patches,
                                 d=lc.feat_dim)


def build_dataloader(lc: LoaderConfig, tokenizer=None):
    """构造 DataLoader（完整吞吐参数）。返回 (loader, dataset)。"""
    ds = build_dataset(lc, tokenizer)
    pixel = (lc.mode == "pixel")
    loader = DataLoader(
        ds, batch_size=lc.batch_size, shuffle=lc.shuffle, num_workers=lc.num_workers,
        pin_memory=lc.pin_memory, drop_last=lc.drop_last,
        persistent_workers=(lc.persistent_workers and lc.num_workers > 0),
        collate_fn=lambda b: collate_fn(b, pad_id=lc.pad_id, pixel=pixel),
    )
    return loader, ds


def cycle(loader):
    """无限迭代 loader（每轮重新洗牌）；供按 step 计数的 Trainer.fit 使用，天然处理 max_steps>一个 epoch。"""
    while True:
        for batch in loader:
            yield batch
