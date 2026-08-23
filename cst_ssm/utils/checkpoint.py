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
    seen: dict[tuple, str] = {}
    canonical: dict = {}
    aliases: dict = {}
    for name, t in state_dict.items():
        if not torch.is_tensor(t):
            # 非张量值（如 int/float 缓冲标量）不参与共享判定，直接独立保存
            canonical[name] = t
            continue
        # 在**原始设备**上取 storage 指针：GPU 上 tied 权重 .cpu() 会各自拷贝成独立存储，
        # 若在 CPU 副本上取指针则识别不到共享，故必须先在原张量上判定。
        # 键含 storage_offset：共享 storage 的**偏移视图**（如 a=big[0:5], b=big[5:10]）
        # 同指针同形状但数据不同，缺 offset 会把 b 误判为 tied 而丢弃 → 加载后数据损坏。
        # 键还须含 stride：同指针 + 同 offset + 同形状但 stride 不同的两个视图（方阵与其
        # 转置最典型）内容完全不同，缺 stride 同样会误判为绑定权重并静默丢掉一份。
        ptr = t.untyped_storage().data_ptr() if t.numel() > 0 else id(t)
        key = (ptr, t.storage_offset(), tuple(t.shape), tuple(t.stride()))
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
    """保存分片检查点到 out_dir。返回 index 字典。

    分片文件名带总片数（model-00001-of-00006），同一目录二次保存若片数变少（6 → 1），
    旧的 model-*-of-00006 会原样留下。它们不在新 index 的 weight_map 里，但 verify_shards
    按目录枚举文件，会把这些孤儿分片一并计入，得到一个「校验通过」却混了两代权重的目录。

    清理放在**全部写完之后**：先删后写的话，save_file 中途失败（磁盘满、被中断）会连
    上一代那些本次用不到的分片一起毁掉。写完再删只清理确实不属于本次的文件。

    注意这**不是原子保存**：分片名只由序号与总片数决定，同名分片仍会被就地覆盖，所以
    中途失败时被覆盖过的那几片已经是新内容、目录处于混合状态（index 尚未更新，load_sharded
    会按旧 index 找不到键而报 KeyError，不会静默读出错值）。要真正原子请写进临时目录再
    整体 rename——不在本次修复范围内。
    """
    os.makedirs(out_dir, exist_ok=True)
    canonical, aliases = _canonicalize(state_dict)
    effective = max(1, int(max_shard_bytes * _HEADER_MARGIN))   # 预留头部空间
    shards = _pack_shards(canonical, effective)
    n = len(shards)
    weight_map: dict = {}
    total = 0
    written: set = set()
    for i, names in enumerate(shards):
        fname = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
        part = {name: canonical[name] for name in names}
        save_file(part, os.path.join(out_dir, fname))
        written.add(fname)
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
    # 新分片与 index 都已落盘，此刻再清掉上一代的孤儿分片
    stale = [f for f in os.listdir(out_dir)
             if f.startswith("model-") and f.endswith(".safetensors") and f not in written]
    for f in stale:
        os.remove(os.path.join(out_dir, f))
    if stale:
        print(f"[checkpoint] 已清理 {len(stale)} 个上一代残留分片（本次 {n} 片）")
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


def load_checkpoint(model, path: str, strict: bool = False,
                    max_missing_ratio: float = 0.5, tag: str = "ckpt"):
    """把权重加载进 model，并**报告** missing / unexpected。

    直接写 `model.load_state_dict(load_sharded(p), strict=False)` 并丢弃返回值，
    在架构与检查点不匹配时（feat_dim / d_model 不同、grounding_head 开关不同）会静默
    加载成一个几乎全随机的模型，然后正常产出"评测结果"。这里把缺失比例打出来，
    超过 max_missing_ratio 直接 raise——宁可崩，也别报一个假数。

    path 既可以是 save_sharded 产出的目录，也可以是 torch.save 的 .pt 文件。

    strict=False **不**容忍形状不匹配（PyTorch 只对 missing/unexpected 宽容，尺寸对不上
    照样 raise）。而跨阶段热启必然撞上这一条：projector 的输出维 = LLM 的 hidden，
    stage-1 用 stand-in（512）、stage-2 接 Qwen3-VL-4B（2560），同名张量形状不同。
    这是**合法**的——stage-1 根本没训 projector（它只在 encode_visual 里用到），
    那份权重本就是随机初始值，丢掉零损失。所以这里按名逐个比形状，不一致的剔除并
    计入缺失（照常打印、照常参与阈值判断），而不是让整次加载崩掉。
    """
    import torch as _torch
    if os.path.isdir(path):
        state = load_sharded(path)
    else:
        sd = _torch.load(path, map_location="cpu")
        state = sd.get("model", sd) if isinstance(sd, dict) else sd

    model_sd = model.state_dict()
    shape_bad = [(k, tuple(v.shape), tuple(model_sd[k].shape)) for k, v in state.items()
                 if k in model_sd and hasattr(v, "shape") and v.shape != model_sd[k].shape]
    if shape_bad:
        state = {k: v for k, v in state.items() if k not in {b[0] for b in shape_bad}}

    missing, unexpected = model.load_state_dict(state, strict=strict)
    n_total = len(model_sd)
    ratio = len(missing) / max(n_total, 1)
    print(f"[{tag}] 加载 {path}：命中 {n_total - len(missing)}/{n_total} 个张量"
          f"（缺失 {len(missing)}，多余 {len(unexpected)}，形状不匹配 {len(shape_bad)}）")
    if shape_bad:
        _s = ", ".join(f"{k}: {a}→{b}" for k, a, b in shape_bad[:4])
        print(f"[{tag}]   形状不匹配(已跳过，用初始值): {_s}"
              f"{' ...' if len(shape_bad) > 4 else ''}")
    if missing:
        print(f"[{tag}]   缺失(用初始值): {list(missing)[:6]}{' ...' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"[{tag}]   多余(被忽略): {list(unexpected)[:6]}{' ...' if len(unexpected) > 6 else ''}")
    if ratio > max_missing_ratio:
        raise RuntimeError(
            f"[{tag}] 检查点与模型架构严重不匹配：{ratio:.0%} 的参数在检查点中缺失"
            f"（阈值 {max_missing_ratio:.0%}）。多半是 --config 与训练时不一致；"
            f"若确认是有意为之（如只热启部分权重），请显式放宽 max_missing_ratio。")
    return missing, unexpected
