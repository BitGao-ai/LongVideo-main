"""数据集：带时间戳的视频序列 + 文本，支持特征缓存与像素两种模式。

- ByteTokenizer        : 零依赖字节级分词器（stand-in，便于 smoke test；生产替换为 LLM 自带分词器）
- VideoTemporalDataset : 读 jsonl manifest + 每视频 .npz 特征缓存（内存映射，省内存）
                         或 frame_dir 像素模式（端到端从帧图）
- SyntheticVideoDataset: 合成数据，用于无数据时跑通全流程

错误隔离：__getitem__ 内单样本失败时随机替换另一条，不中断训练。
"""
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
        self._pixel = (cfg.mode == "pixel")
        self._n_fail = 0        # 取样失败次数（含被成功替换的）
        self._n_fallback = 0    # 返回零特征占位样本的次数
        # None = 沿用特征文件自身 dtype（默认，见 DataConfig.feat_dtype）
        self._feat_dtype = resolve_feat_dtype(getattr(cfg, "feat_dtype", "auto"))
        self._warned_npz = False

    def _cast_feat(self, t: torch.Tensor) -> torch.Tensor:
        """按 cfg.feat_dtype 决定特征 dtype。

        "auto"（默认）只对**下游确实吃得下的低精度浮点**放行：float16 / bfloat16 / float32
        原样返回（零拷贝）。其余一律回落到 float32，也就是旧的 `.float()` 行为——
        第三方特征若存成 float64 或 uint8，autocast 与 nn.Linear 的处理各不相同，
        "沿用文件 dtype" 在那里不是省显存而是埋雷。
        """
        if self._feat_dtype is not None:
            return t if t.dtype == self._feat_dtype else t.to(self._feat_dtype)
        if t.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return t
        return t.float()

    def __len__(self):
        return len(self.rows)

    def _load_feature(self, ref: str):
        """读特征与时间戳。
        - ref 以 .npy 结尾：真内存映射（惰性按索引读盘），时间戳取同名 <stem>.ts.npy；
        - ref 以 .npz 结尾：注意 mmap 对 zip 归档无效，会整载解压该成员（大特征建议改用分离 .npy）。
        特征 dtype 由 cfg.feat_dtype 决定，默认沿用文件自身（fp16 存就 fp16 用）。
        """
        path = os.path.join(self.root, ref)
        if ref.endswith(".npy"):
            feats = np.load(path, mmap_mode="r")               # 惰性 mmap
            ts = np.load(path[:-4] + ".ts.npy")
            return feats, np.asarray(ts, dtype=np.float32)
        self._warn_npz(ref)
        z = np.load(path)                                      # .npz 整载解压
        return z["features"], np.asarray(z["timestamps"], dtype=np.float32)

    def _warn_npz(self, ref: str) -> None:
        """.npz 无法 mmap —— 规模化训练前必须转 .npy，这里出声提醒一次。

        每个 worker 各喊一次（每个 worker 是独立进程，各有一份实例）就够了：多卡训练下
        8 rank × 8 worker = 64 个进程**各自**整载解压每个样本的全部特征，单样本
        L=4096/P=9/d=2560 的 fp16 就是 189 MB，几十个在世样本足以吃掉几十 GB 主机内存。
        转成 .npy + .ts.npy 后 mmap_mode='r' 生效，只读被 _subsample 选中的那些帧。
        """
        if self._warned_npz:
            return
        self._warned_npz = True
        print(f"[dataset] 提示: 特征是 .npz（{ref}），np.load 的 mmap 对 zip 归档无效，"
              f"每次取样都要整载解压该视频的全部特征。多卡/长视频训练前请先转格式：\n"
              f"[dataset]   python -m data_pipeline.src.npz_to_npy --in <npz目录> "
              f"--out <npy目录> [--shard-dirs]\n"
              f"[dataset]   再用 build_manifest.py --prefer-npy 让 feature_ref 指向 .npy。")

    def _load_pixel_frames(self, frame_dir: str):
        """像素模式：从帧图目录加载帧（按文件名排序），返回 (frames[L,C,H,W], ts[L])。

        帧图目录约定：{frame_dir}/000000.jpg, 000001.jpg, ... 或 .png。
        时间戳：若存在 {frame_dir}/timestamps.npy 则读取，否则按等间隔 1/fps 生成。
        """
        from torchvision import transforms
        from PIL import Image

        fdir = os.path.join(self.root, frame_dir)
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        fnames = sorted(f for f in os.listdir(fdir) if f.lower().endswith(exts))
        if not fnames:
            raise FileNotFoundError(f"frame_dir 无帧图: {fdir}")

        transform = transforms.Compose([
            transforms.Resize((self.cfg.frame_size, self.cfg.frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        frames = []
        for fn in fnames:
            img = Image.open(os.path.join(fdir, fn)).convert("RGB")
            frames.append(transform(img))
        frames = torch.stack(frames)  # (L, C, H, W)

        # 时间戳
        ts_path = os.path.join(fdir, "timestamps.npy")
        if os.path.exists(ts_path):
            ts = np.load(ts_path).astype(np.float32)
        else:
            ts = np.arange(len(fnames), dtype=np.float32) / 30.0  # 默认 30fps
        return frames, ts

    def _subsample(self, L: int) -> np.ndarray:
        """超过 max_frames 时均匀抽样索引（保持时间戳真实性）。"""
        if L <= self.cfg.max_frames:
            return np.arange(L)
        return np.linspace(0, L - 1, self.cfg.max_frames).round().astype(int)

    def __getitem__(self, i: int) -> dict:
        """加载单样本。单样本失败时随机替换另一条（错误隔离，不中断训练）。

        **失败是可见的**：每次替换/兜底都计数，首次出现时告警，之后按指数间隔提示总量。
        静默吞掉异常会让"特征路径写错 / npz 损坏 / 维度不匹配"这类问题在训练与评测里
        完全不留痕迹——loss 曲线照样漂亮、准确率照样有数，但跑的其实是零特征。
        """
        try:
            return self._load_sample(i)
        except Exception as e:
            self._n_fail += 1
            self._report_fail(i, e)
            # 错误隔离：随机选另一条（避免递归死循环，最多重试 5 次）
            for _ in range(5):
                j = random.randint(0, len(self.rows) - 1)
                try:
                    return self._load_sample(j)
                except Exception:
                    continue
            # 全部失败：返回合成占位样本（保证训练不崩）
            self._n_fallback += 1
            if self._n_fallback == 1:
                print(f"[dataset] 严重告警: 连续 6 次取样失败，返回**零特征占位样本**。"
                      f"此后的指标不可信，请先修数据。首个错误: {type(e).__name__}: {e}")
            return self._fallback_sample()

    def _report_fail(self, i: int, e: Exception) -> None:
        """按 1,2,4,8,… 的间隔告警，既不刷屏也不静默。"""
        n = self._n_fail
        if n & (n - 1) == 0:      # n 是 2 的幂
            ratio = n / max(len(self.rows), 1)
            print(f"[dataset] 警告: 第 {n} 次取样失败（占清单 {ratio:.1%}），"
                  f"行 {i} video_id={self.rows[i].get('video_id')!r}: {type(e).__name__}: {e}")

    @property
    def failure_stats(self) -> dict:
        """取样失败统计。评测脚本应在收尾时检查，failures>0 意味着指标含替换样本。"""
        return {"failures": self._n_fail, "fallbacks": self._n_fallback,
                "total_rows": len(self.rows)}

    def _load_sample(self, i: int) -> dict:
        r = self.rows[i]
        if self._pixel and r.get("frame_dir"):
            frames, ts = self._load_pixel_frames(r["frame_dir"])
            idx = self._subsample(len(ts))
            vis = frames[idx]                              # (L,C,H,W)
            ts = torch.from_numpy(ts[idx]).float()         # (L,)
        else:
            feats, ts = self._load_feature(r["feature_ref"])
            idx = self._subsample(len(ts))
            # 不再无条件 .float()：磁盘上是 fp16，而下游第一个算子（FeatureAdapter 的
            # Linear）在 autocast(bf16) 下无论如何都会降到 bf16，那份 fp32 全程没有消费者。
            # dtype 由 cfg.feat_dtype 决定（默认 auto=沿用文件自身），不带 autocast 的路径
            # 由 FeatureAdapter 入口的 align_to_param 升精度兜底，结果与旧行为逐位相同。
            # np.ascontiguousarray：mmap 下 feats[idx] 已是新数组，但 .npz 路径下可能是
            # 非连续视图，torch.from_numpy 要求连续。
            vis = torch.from_numpy(np.ascontiguousarray(feats[idx]))   # (L,P,d)
            vis = self._cast_feat(vis)
            ts = torch.from_numpy(ts[idx]).float()         # (L,)

        # 若分词器支持（HFTokenizer），告知当前帧数 Lv 以展开 <video> 占位符（对齐 _scatter_visual）
        if hasattr(self.tok, "set_video_tokens"):
            self.tok.set_video_tokens(len(idx))
        ids, labels = self.tok.build_lm_example(r["prompt"], r.get("answer", ""), self.cfg.max_text_len)
        vkey = "frames" if self._pixel else "features"
        sample = {
            vkey: vis, "timestamps": ts,
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            # 样本真实对应的 manifest 行号。__getitem__ 失败时会替换成别的行，
            # 下游评测若按 batch 顺序推算行号就会静默错位（拿 A 的预测配 B 的答案），
            # 因此把行号跟着样本一起传出去，由 collate 透传给评测脚本。
            "row_index": i,
        }
        for k in ("gt_start", "gt_end", "duration", "task_type", "video_id"):
            if k in r:
                sample[k] = r[k]
        return sample

    def _fallback_sample(self) -> dict:
        """合成占位样本（所有真实样本均不可用时的最后兜底）。"""
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
            "row_index": -1,          # -1 = 占位样本，不对应任何 manifest 行
        }


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
