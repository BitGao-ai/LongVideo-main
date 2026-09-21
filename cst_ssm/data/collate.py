"""Variable-length collate: pads frames and text, emits validity masks."""
from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader


def collate_fn(batch: list, pad_id: int = 256, pixel: bool = False) -> dict:
    B = len(batch)
    vkey = "frames" if pixel else "features"
    Lmax = max(b[vkey].shape[0] for b in batch)
    Tmax = max(b["input_ids"].shape[0] for b in batch)

    v0 = batch[0][vkey]
    for i, b in enumerate(batch):
        if b[vkey].shape[1:] != v0.shape[1:]:
            hint = ""
            if not pixel and b[vkey].dim() >= 2 and b[vkey].shape[1] != v0.shape[1]:
                hint = (f" Patch count P differs ({b[vkey].shape[1]} vs {v0.shape[1]}): "
                        "features were extracted with different --patches; re-extract uniformly.")
            raise ValueError(
                f"collate_fn: sample {i} {vkey} shape {tuple(b[vkey].shape[1:])} "
                f"differs from sample 0 {tuple(v0.shape[1:])} outside the time dim.{hint}")

    vis = torch.zeros(B, Lmax, *v0.shape[1:], dtype=v0.dtype)
    ts = torch.zeros(B, Lmax)
    frame_mask = torch.zeros(B, Lmax, dtype=torch.bool)
    input_ids = torch.full((B, Tmax), pad_id, dtype=torch.long)
    labels = torch.full((B, Tmax), -100, dtype=torch.long)
    attn = torch.zeros(B, Tmax, dtype=torch.bool)

    extras: dict = {}
    for i, b in enumerate(batch):
        L = b[vkey].shape[0]
        T = b["input_ids"].shape[0]
        vis[i, :L] = b[vkey]
        ts[i, :L] = b["timestamps"]
        if L < Lmax:
            ts[i, L:] = b["timestamps"][-1]
        frame_mask[i, :L] = True
        input_ids[i, :T] = b["input_ids"]
        labels[i, :T] = b["labels"]
        attn[i, :T] = True
        for k in ("gt_start", "gt_end", "duration", "task_type", "video_id", "row_index"):
            if k in b:
                extras.setdefault(k, []).append(b[k])

    out = {
        vkey: vis, "timestamps": ts, "frame_mask": frame_mask,
        "input_ids": input_ids, "labels": labels, "attention_mask": attn,
    }
    out.update(extras)
    return out


def make_loader(dataset, batch_size: int = 2, shuffle: bool = True,
                pixel: bool = False, num_workers: int = 0, pad_id: int = 256) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=partial(collate_fn, pad_id=pad_id, pixel=pixel))
