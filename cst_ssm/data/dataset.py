"""Datasets: timestamped video sequences with text, feature-cache or pixel input."""
from __future__ import annotations

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import DataConfig
from ..dtypes import resolve_feat_dtype


class ByteTokenizer:
    """Byte-level tokenizer for stand-in models and smoke tests."""

    PAD, BOS, EOS = 256, 257, 258
    vocab_size = 259

    def encode(self, s: str) -> list:
        return [self.BOS] + list(s.encode("utf-8")) + [self.EOS]

    def build_lm_example(self, prompt: str, answer: str, max_len: int):
        """Concatenate prompt and answer; labels mask the prompt with -100."""
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
        self._pixel = (cfg.mode == "pixel")
        self._n_fail = 0
        self._n_fallback = 0
        self._feat_dtype = resolve_feat_dtype(getattr(cfg, "feat_dtype", "auto"))
        self._warned_npz = False

    def _cast_feat(self, t: torch.Tensor) -> torch.Tensor:
        """Cast features per cfg.feat_dtype; unknown dtypes fall back to float32."""
        if self._feat_dtype is not None:
            return t if t.dtype == self._feat_dtype else t.to(self._feat_dtype)
        if t.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return t
        return t.float()

    def __len__(self):
        return len(self.rows)

    def _load_feature(self, ref: str):
        """Load features and timestamps; .npy uses mmap, .npz loads the member."""
        path = os.path.join(self.root, ref)
        if ref.endswith(".npy"):
            feats = np.load(path, mmap_mode="r")
            ts = np.load(path[:-4] + ".ts.npy")
            return feats, np.asarray(ts, dtype=np.float32)
        self._warn_npz(ref)
        z = np.load(path)
        return z["features"], np.asarray(z["timestamps"], dtype=np.float32)

    def _warn_npz(self, ref: str) -> None:
        if self._warned_npz:
            return
        self._warned_npz = True
        print(f"[dataset] features are .npz ({ref}); mmap does not apply to zip archives. "
              f"Convert to .npy before large-scale training:\n"
              f"[dataset]   python -m data_pipeline.src.npz_to_npy --in <npz dir> --out <npy dir>")

    def _load_pixel_frames(self, frame_dir: str):
        """Load sorted frames from a directory; returns (frames[L,C,H,W], ts[L])."""
        from torchvision import transforms
        from PIL import Image

        fdir = os.path.join(self.root, frame_dir)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        fnames = sorted(f for f in os.listdir(fdir) if f.lower().endswith(exts))
        if not fnames:
            raise FileNotFoundError(f"no frames in frame_dir: {fdir}")

        transform = transforms.Compose([
            transforms.Resize((self.cfg.frame_size, self.cfg.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        frames = []
        for fn in fnames:
            img = Image.open(os.path.join(fdir, fn)).convert("RGB")
            frames.append(transform(img))
        frames = torch.stack(frames)

        ts_path = os.path.join(fdir, "timestamps.npy")
        if os.path.exists(ts_path):
            ts = np.load(ts_path).astype(np.float32)
        else:
            ts = np.arange(len(fnames), dtype=np.float32) / 30.0
        return frames, ts

    def _subsample(self, L: int) -> np.ndarray:
        """Uniform index subsample beyond max_frames, preserving timestamps."""
        if L <= self.cfg.max_frames:
            return np.arange(L)
        return np.linspace(0, L - 1, self.cfg.max_frames).round().astype(int)

    def __getitem__(self, i: int) -> dict:
        """Load one sample; failed rows are replaced by a random row (counted)."""
        try:
            return self._load_sample(i)
        except Exception as e:
            self._n_fail += 1
            self._report_fail(i, e)
            for _ in range(5):
                j = random.randint(0, len(self.rows) - 1)
                try:
                    return self._load_sample(j)
                except Exception:
                    continue
            self._n_fallback += 1
            if self._n_fallback == 1:
                print(f"[dataset] 6 consecutive sample failures; returning a zero-feature "
                      f"placeholder. Metrics are unreliable until data is fixed. First error: "
                      f"{type(e).__name__}: {e}")
            return self._fallback_sample()

    def _report_fail(self, i: int, e: Exception) -> None:
        n = self._n_fail
        if n & (n - 1) == 0:
            ratio = n / max(len(self.rows), 1)
            print(f"[dataset] sample failure #{n} ({ratio:.1%} of manifest), "
                  f"row {i} video_id={self.rows[i].get('video_id')!r}: {type(e).__name__}: {e}")

    @property
    def failure_stats(self) -> dict:
        return {"failures": self._n_fail, "fallbacks": self._n_fallback,
                "total_rows": len(self.rows)}

    def _load_sample(self, i: int) -> dict:
        r = self.rows[i]
        if self._pixel and r.get("frame_dir"):
            frames, ts = self._load_pixel_frames(r["frame_dir"])
            idx = self._subsample(len(ts))
            vis = frames[idx]
            ts = torch.from_numpy(ts[idx]).float()
        else:
            feats, ts = self._load_feature(r["feature_ref"])
            idx = self._subsample(len(ts))
            vis = torch.from_numpy(np.ascontiguousarray(feats[idx]))
            vis = self._cast_feat(vis)
            ts = torch.from_numpy(ts[idx]).float()

        if hasattr(self.tok, "set_video_tokens"):
            self.tok.set_video_tokens(len(idx))
        ids, labels = self.tok.build_lm_example(r["prompt"], r.get("answer", ""), self.cfg.max_text_len)
        vkey = "frames" if self._pixel else "features"
        sample = {
            vkey: vis, "timestamps": ts,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "row_index": i,
        }
        for k in ("gt_start", "gt_end", "duration", "task_type", "video_id"):
            if k in r:
                sample[k] = r[k]
        return sample

    def _fallback_sample(self) -> dict:
        """Zero-feature placeholder when no real sample is loadable."""
        L, P, d = 4, self.cfg.feat_patches, self.cfg.feat_dim
        vkey = "frames" if self._pixel else "features"
        if self._pixel:
            vis = torch.zeros(L, 3, self.cfg.frame_size, self.cfg.frame_size)
        else:
            vis = torch.zeros(L, P, d)
        return {
            vkey: vis,
            "timestamps": torch.arange(L, dtype=torch.float32),
            "input_ids": torch.tensor([self.tok.BOS, self.tok.EOS], dtype=torch.long),
            "labels": torch.tensor([-100, self.tok.EOS], dtype=torch.long),
            "row_index": -1,
        }


class SyntheticVideoDataset(Dataset):
    """Synthetic features, timestamps, and text for smoke tests."""

    def __init__(self, n: int = 64, L: int = 32, P: int = 4, d: int = 64,
                 text_len: int = 24, vocab: int = 259, seed: int = 0):
        self.n, self.L, self.P, self.d, self.text_len, self.vocab = n, L, P, d, text_len, vocab
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i: int) -> dict:
        g = torch.Generator().manual_seed(self.seed * 100003 + i)
        L = self.L
        gaps = torch.rand(L, generator=g) * 0.4 + 0.1
        return dict(
            features=torch.randn(L, self.P, self.d, generator=g),
            timestamps=torch.cumsum(gaps, 0),
            input_ids=torch.randint(0, 256, (self.text_len,), generator=g),
            labels=torch.randint(0, 256, (self.text_len,), generator=g),
        )


class SyntheticGroundingDataset(Dataset):
    """Synthetic localization data with random queries and [gt_start, gt_end] spans."""

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
