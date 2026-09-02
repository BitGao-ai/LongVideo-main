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


def _dtype_bytes(t) -> int:
    return t.numel() * t.element_size() if torch.is_tensor(t) else 0


def _to_cpu(t):
    """落盘用的 CPU 连续副本。在**写每个分片时**才调用，见 save_sharded 的说明。"""
    return t.detach().cpu().contiguous() if torch.is_tensor(t) else t


def _canonicalize(state_dict: dict) -> tuple[dict, dict]:
    """检测共享存储（权重绑定），返回 (canonical_tensors, aliases)。

    aliases: {别名 -> 规范名}；canonical_tensors 只含每组共享的第一个，且**保留原张量引用**
    （不在这里 .cpu()）。此前这里就地把每个张量拷成 CPU 副本，于是整个模型的 CPU 副本会
    在开始写盘之前全部同时在世——Qwen3-VL-4B 就是 rank0 上一个约 8.8GB 的主机内存尖峰。
    改成把 .cpu() 推迟到 save_sharded 的分片写循环里，峰值降到"一个分片"（≤3.8GB，
    只存可训练权重时通常只有几百 MB）。装箱只需要尺寸，而尺寸在原设备上就能算。
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
        # 若在 CPU 副本上取指针则识别不到共享，故必须先在原张量上判定（这也是不能提前
        # .cpu() 的第二个原因）。
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
            canonical[name] = t
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
                 max_shard_bytes: int = _DEFAULT_MAX_BYTES,
                 extra_metadata: dict | None = None) -> dict:
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

    主机内存：`.cpu()` 只在写每个分片时对该分片的张量做，写完即释放（`del part`），
    所以峰值是"一个分片"而不是"整个模型"。extra_metadata 会并进 index 的 metadata，
    供调用方标注检查点性质（如 trainable_only）。
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
        # 只把本片的张量搬到 CPU；写完立刻释放，主机内存峰值 = 单片大小
        part = {name: _to_cpu(canonical[name]) for name in names}
        save_file(part, os.path.join(out_dir, fname))
        for name in names:
            weight_map[name] = fname
            total += _dtype_bytes(part[name])
        del part
        written.add(fname)
    meta = {"total_size": total, "n_shards": n, "max_shard_bytes": max_shard_bytes}
    meta.update(extra_metadata or {})
    index = {
        "metadata": meta,
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


def frozen_param_names(model) -> set:
    """requires_grad=False 的参数名集合（与 state_dict 的键同名空间）。"""
    return {n for n, p in model.named_parameters() if not p.requires_grad}


def base_weights_recoverable(model) -> bool:
    """模型的冻结权重能否从**外部来源**重新取回（而不是只存在于这份检查点里）？

    唯一算"能"的情形：LLM 段是 HF 底座的包装（Qwen3VLLanguageModel 持有 `.hf`）。那时
    被冻的都是 Qwen3-VL 的权重，下次 `from_pretrained` 原样拿回来，检查点里存它纯属浪费
    （4B bf16 ≈ 8.8GB/次，ckpt_every=500 跑 5000 步就是 ≈88GB 磁盘）。

    反过来必须判"不能"的情形是 **stand-in + LoRA**：那时被冻的是
    VisualConditionedLM 的**随机初始化**权重，任何外部来源都复现不了，丢掉检查点就废了。
    未知的注入式 LLM 一律按"不能"处理——保守的一侧是多存，不是少存。
    """
    llm = getattr(model, "llm", None)
    return llm is not None and hasattr(llm, "hf")


def trainable_state_dict(model) -> dict:
    """state_dict 去掉冻结参数，**保留全部 buffer**。

    buffer 必须留：EventGate 的 RunningStandardizer 统计（running_mean/var）是训练过程中
    积累出来的，丢了它门控残差的尺度就变了；而它们本来就只有几 KB。
    """
    frozen = frozen_param_names(model)
    return {k: v for k, v in model.state_dict().items() if k not in frozen}


def checkpoint_metadata(path: str) -> dict:
    """读取分片检查点的 index.metadata（含 trainable_only 标记）。读不到返回 {}。

    给加载侧一个判据：大量键缺失时，究竟是"保存时省掉了可从 HF 取回的冻结底座"
    （trainable_only=True，正常），还是"架构真的不匹配"（该报错）。
    path 可以是目录，也可以是 .pt 文件（后者无 index，恒返回 {}）。
    """
    if not os.path.isdir(path):
        return {}
    try:
        with open(os.path.join(path, _INDEX_NAME)) as f:
            return json.load(f).get("metadata", {}) or {}
    except (OSError, ValueError):
        return {}


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


CRITICAL_SHAPE_PREFIXES = ("vision.",)
"""形状**不允许**静默不匹配的参数前缀。

`vision.` 是 feature 模式下的 FeatureAdapter（d_in=cfg.feat_dim, d_out=cfg.d_model）。
它的形状不匹配意味着这份检查点与当前配置不是一套，属于数据/配置错误，不是合法的跨阶段
差异（对照：projector 形状变化是合法的，见 load_checkpoint 的 docstring）。
"""


def _vision_mismatch_hint(critical: list, tag: str) -> str:
    """按 `vision.proj.weight` 的两个维度分辨根因，给出**对应**的补救路径。

    vision.proj = nn.Linear(feat_dim, d_model)，权重形状 (d_model, feat_dim)：
      · 列（in_features）不同 → 特征是另一个视觉塔抽的，必须重抽特征 + 重训 stage-1；
      · 行（out_features）不同 → d_model 被改过，两个阶段用的不是同一份 --config，
        重训 stage-1 即可，**特征可原样复用**。
    必须分开说：此前这里无条件按"另一个视觉塔"叙述，而 d_model 错配（最常见的成因是
    stage-1 漏传 --config，退到脚本内置兜底的 d_model=96）会被这句话导向"重抽特征"——
    那是几十 GPU 小时，且根本修不掉问题。两条补救路径的代价差着数量级，不能混为一谈。
    """
    generic = (f"[{tag}] vision.* 是 feature 模式下的 FeatureAdapter"
               f"(d_in=feat_dim, d_out=d_model)：形状对不上说明这份检查点与当前配置不是"
               f"一套。请核对两次训练的 --config（d_model）与特征来源（feat_dim）。")
    proj = next((b for b in critical if str(b[0]).endswith("vision.proj.weight")), None)
    if proj is None or len(proj[1]) != 2 or len(proj[2]) != 2:
        return generic
    (ck_d_model, ck_feat), (m_d_model, m_feat) = proj[1], proj[2]
    lines = []
    if ck_feat != m_feat:
        lines.append(
            f"[{tag}] · feat_dim 不同（检查点 {ck_feat} vs 当前 {m_feat}）：特征由**另一个"
            f"视觉塔**抽出。feat_dim 由抽特征的模型唯一决定，改配置没用——请确认 "
            f"extract_features 的 --model、build_manifest 的 --feature-dir 与本次训练的 "
            f"--base-model 三者同源，用同一底座**重抽特征**并重训 stage-1。")
    if ck_d_model != m_d_model:
        lines.append(
            f"[{tag}] · d_model 不同（检查点 {ck_d_model} vs 当前 {m_d_model}）：两个阶段"
            f"用的不是同一份配置——最常见的是 stage-1 漏传 --config，退到了脚本内置兜底"
            f"（d_model=96、stand-in LLM dim=128）。**特征无需重抽**，用与本次相同的 "
            f"--config 重训 stage-1 即可。")
    return "\n".join(lines) if lines else generic


def load_checkpoint(model, path: str, strict: bool = False,
                    max_missing_ratio: float = 0.5, tag: str = "ckpt",
                    critical_prefixes: tuple = CRITICAL_SHAPE_PREFIXES):
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

    但"剔除并计入缺失"对 `critical_prefixes` 下的参数是**危险**的：stage-2 用
    max_missing_ratio=0.99（4B 底座下 LLM 权重天然全缺），这个阈值宽到足以吞掉任何东西，
    于是一个 feat_dim 错配的 FeatureAdapter（vision.*）会被静默丢弃、退回随机初始值——
    stage-1 学到的视觉对齐全部作废，训练照常进行、loss 照常下降，没有任何一行报错。
    这比直接崩掉坏得多。故对这些前缀：形状不匹配即 raise。

    两道守卫的**次序**：缺失比例的判定在前。两者会同时触发（如 d_model 改了，vision 与
    其余各段一起变形），此时根因是"架构与检查点不是一套"，缺失比例那条信息量更大；
    把关键前缀的判定放在其后，才不会把那句话挤掉。想显式强跑请传 critical_prefixes=()。
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
    # 关键前缀的形状守卫放在缺失比例之后：见 docstring 的"两道守卫的次序"。
    critical = [b for b in shape_bad if str(b[0]).startswith(tuple(critical_prefixes or ()))]
    if critical:
        _d = "; ".join(f"{k}: 检查点{a} vs 模型{b}" for k, a, b in critical[:4])
        raise RuntimeError(
            f"[{tag}] 检查点与模型在关键参数上形状不匹配，拒绝静默丢弃：{_d}"
            f"{' ...' if len(critical) > 4 else ''}\n"
            f"{_vision_mismatch_hint(critical, tag)}\n"
            f"[{tag}] 若确认要丢弃这些权重（从随机初始值重训该段），显式传 critical_prefixes=()。")
    return missing, unexpected
