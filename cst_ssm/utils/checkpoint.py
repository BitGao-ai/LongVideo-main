"""分片检查点：把模型权重按 safetensors 分片保存，**保证单文件 ≤ 4GB**（需求 2）。

- save_sharded : 贪心装箱把张量分到多个 .safetensors 分片（默认上限 3.8GB<4GB），写 index.json；
                 自动处理**权重绑定/共享存储**（如 lm_head 与 embed 绑定）——只存一份，别名记录在 index。
- load_sharded : 按 index 还原完整 state_dict（含别名重建）。
仅用 safetensors（安全、零拷贝、跨框架），不用 torch.save（pickle 不安全且难分片）。
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import torch
from safetensors.torch import save_file, load_file

_DEFAULT_MAX_BYTES = int(3.8 * 1024 ** 3)   # 3.8 GiB < 4GB 硬约束
_INDEX_NAME = "model.safetensors.index.json"
_HEADER_MARGIN = 0.98                         # 预留 2% 给 safetensors 头，保证**文件**（非仅载荷）≤上限


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.2f}{unit}"
        n /= 1024


def _dtype_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _canonicalize(state_dict: dict) -> tuple[dict, dict]:
    """检测共享存储（权重绑定），返回 (canonical_tensors, aliases)。

    aliases: {别名 -> 规范名}；canonical_tensors 只含每组共享的第一个。
    """
    seen: dict[int, str] = {}
    canonical: dict = {}
    aliases: dict = {}
    for name, t in state_dict.items():
        # 在**原始设备**上取 storage 指针：GPU 上 tied 权重 .cpu() 会各自拷贝成独立存储，
        # 若在 CPU 副本上取指针则识别不到共享，故必须先在原张量上判定。
        ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else id(t)
        key = (ptr, tuple(t.shape))
        if key in seen:
            aliases[name] = seen[key]        # 与已存张量共享 → 记别名
        else:
            seen[key] = name
            canonical[name] = t.detach().cpu().contiguous()   # 存时才转 CPU
    return canonical, aliases


def _pack_shards(tensors: dict, max_bytes: int) -> list[list[str]]:
    """贪心装箱：按大小降序放入分片，每片累计 ≤ max_bytes。单张量超限则独占一片（告警）。"""
    items = sorted(tensors.items(), key=lambda kv: _dtype_bytes(kv[1]), reverse=True)
    shards: list[list[str]] = []
    sizes: list[int] = []
    for name, t in items:
        sz = _dtype_bytes(t)
        if sz > max_bytes:
            print(f"[checkpoint] 警告：张量 {name} 单体 {sz/1e9:.2f}GB 超过分片上限，独占一片。")
            shards.append([name]); sizes.append(sz); continue
        placed = False
        for i in range(len(shards)):
            if sizes[i] + sz <= max_bytes:
                shards[i].append(name); sizes[i] += sz; placed = True; break
        if not placed:
            shards.append([name]); sizes.append(sz)
    return shards


def save_sharded(state_dict: dict, out_dir: str,
                 max_shard_bytes: int = _DEFAULT_MAX_BYTES) -> dict:
    """保存分片检查点到 out_dir。返回 index 字典。"""
    os.makedirs(out_dir, exist_ok=True)
    canonical, aliases = _canonicalize(state_dict)
    effective = max(1, int(max_shard_bytes * _HEADER_MARGIN))   # 预留头部空间
    shards = _pack_shards(canonical, effective)
    n = len(shards)
    weight_map: dict = {}
    total = 0
    for i, names in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        part = {name: canonical[name] for name in names}
        save_file(part, os.path.join(out_dir, fname))
        for name in names:
            weight_map[name] = fname
            total += _dtype_bytes(canonical[name])
    index = {
        "metadata": {"total_size": total, "n_shards": n, "max_shard_bytes": max_shard_bytes},
        "weight_map": weight_map,
        "aliases": aliases,
    }
    with open(os.path.join(out_dir, _INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[checkpoint] 已保存 {len(canonical)} 张量(+{len(aliases)} 别名) 到 {n} 个分片，"
          f"总计 {_human(total)}，单片≤{_human(max_shard_bytes)} → {out_dir}")
    return index


def load_sharded(out_dir: str, map_location: str = "cpu") -> dict:
    """从 out_dir 还原完整 state_dict。"""
    with open(os.path.join(out_dir, _INDEX_NAME)) as f:
        index = json.load(f)
    state: dict = {}
    cache: dict = {}
    for name, fname in index["weight_map"].items():
        if fname not in cache:
            cache[fname] = load_file(os.path.join(out_dir, fname), device=map_location)
        state[name] = cache[fname][name]
    for alias, canonical in index.get("aliases", {}).items():   # 重建绑定权重
        state[alias] = state[canonical]
    return state


def verify_shards(out_dir: str, hard_limit_bytes: int = 4 * 1024 ** 3) -> bool:
    """校验目录内每个分片文件 ≤ 硬上限（默认 4GB）。"""
    ok = True
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".safetensors"):
            sz = os.path.getsize(os.path.join(out_dir, fn))
            flag = "OK" if sz <= hard_limit_bytes else "超限!"
            if sz > hard_limit_bytes:
                ok = False
            print(f"  {fn}: {_human(sz)} {flag}")
    return ok
