"""数据集：带时间戳的视频序列 + 文本，支持特征缓存与像素两种模式。

- ByteTokenizer        : 零依赖字节级分词器（stand-in，便于 smoke test；生产替换为 LLM 自带分词器）
- VideoTemporalDataset : 读 jsonl manifest + 每视频 .npz 特征缓存（内存映射，省内存）
- SyntheticVideoDataset: 合成数据，用于无数据时跑通全流程
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import DataConfig


class ByteTokenizer:
    """字节级分词器（vocab=256 + 特殊符）。仅用于 stand-in/smoke test。"""
    PAD, BOS, EOS = 256, 257, 258
    vocab_size = 259

    def encode(self, s: str) -> list:
        return [self.BOS] + list(s.encode("utf-8")) + [self.EOS]

    def build_lm_example(self, prompt: str, answer: str, max_len: int):
        """拼 prompt+answer，labels 对 prompt 部分打 -100（只学答案）。"""
        p = [self.BOS] + list(prompt.encode("utf-8"))
        a = list(answer.encode("utf-8")) + [self.EOS]
        ids = (p + a)[:max_len]
        labels = ([-100] * len(p) + a)[:max_len]
        return ids, labels


class VideoTemporalDataset(Dataset):
    def __init__(self, cfg: DataConfig, tokenizer=None, data_root: str = ""):
        self.cfg = cfg
        self.tok = tokenizer or ByteTokenizer()
        self.root = data_root
        with open(cfg.manifest) as f:
            self.rows = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.rows)

    def _load_feature(self, ref: str):
        """读特征与时间戳。
        - ref 以 .npy 结尾：真内存映射（惰性按索引读盘），时间戳取同名 <stem>.ts.npy；
        - ref 以 .npz 结尾：注意 mmap 对 zip 归档无效，会整载解压该成员（大特征建议改用分离 .npy）。
        保留 float16 存储，索引后再由调用方转 float32（省一半内存）。
        """
        path = os.path.join(self.root, ref)
        if ref.endswith(".npy"):
            feats = np.load(path, mmap_mode="r")               # 惰性 mmap
            ts = np.load(path[:-4] + ".ts.npy")
            return feats, np.asarray(ts, dtype=np.float32)
        z = np.load(path)                                      # .npz 整载解压
        return z["features"], np.asarray(z["timestamps"], dtype=np.float32)

    def _subsample(self, L: int) -> np.ndarray:
        """超过 max_frames 时均匀抽样索引（保持时间戳真实性）。"""
        if L <= self.cfg.max_frames:
            return np.arange(L)
        return np.linspace(0, L - 1, self.cfg.max_frames).round().astype(int)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        feats, ts = self._load_feature(r["feature_ref"])
        idx = self._subsample(len(ts))
        feats = torch.from_numpy(feats[idx]).float()          # (L,P,d)
        ts = torch.from_numpy(ts[idx]).float()                # (L,)
        ids, labels = self.tok.build_lm_example(r["prompt"], r.get("answer", ""), self.cfg.max_text_len)
        sample = dict(
            features=feats, timestamps=ts,
            input_ids=torch.tensor(ids, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
        )
        for k in ("gt_start", "gt_end", "duration", "task_type", "video_id"):
            if k in r:
                sample[k] = r[k]
        return sample


class SyntheticVideoDataset(Dataset):
    """合成数据：随机特征 + 可变时间戳 + 随机文本。用于 smoke test / 无数据跑通。"""

    def __init__(self, n: int = 64, L: int = 32, P: int = 4, d: int = 64,
                 text_len: int = 24, vocab: int = 259, seed: int = 0):
        self.n, self.L, self.P, self.d, self.text_len, self.vocab = n, L, P, d, text_len, vocab
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)   # 逐样本确定性，num_workers 下不重复
        L = self.L
        gaps = torch.rand(L, generator=g) * 0.4 + 0.1
        return dict(
            features=torch.randn(L, self.P, self.d, generator=g),
            timestamps=torch.cumsum(gaps, 0),
            input_ids=torch.randint(0, 256, (self.text_len,), generator=g),
            labels=torch.randint(0, 256, (self.text_len,), generator=g),
        )


class SyntheticGroundingDataset(Dataset):
    """合成定位数据：随机特征 + 可变时间戳 + 随机 query + 随机 [gt_start,gt_end] 区间。

    用于 grounding_head 无数据训练/自测（含 gt_start/gt_end，collate 会转发）。
    """

    def __init__(self, n: int = 64, L: int = 24, P: int = 4, d: int = 64,
                 text_len: int = 16, seed: int = 0):
        self.n, self.L, self.P, self.d, self.text_len, self.seed = n, L, P, d, text_len, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        L = self.L
        ts = torch.cumsum(torch.rand(L, generator=g) * 0.4 + 0.1, 0)
        lo = int(torch.randint(0, L - 2, (1,), generator=g))
        hi = int(torch.randint(lo + 1, L, (1,), generator=g))
        return dict(
            features=torch.randn(L, self.P, self.d, generator=g),
            timestamps=ts,
            input_ids=torch.randint(0, 256, (self.text_len,), generator=g),
            labels=torch.randint(0, 256, (self.text_len,), generator=g),
            gt_start=float(ts[lo]), gt_end=float(ts[hi]),
            task_type="grounding",
        )
