# CST-SSM 训练与推理运行指令

> 覆盖三类形态：单卡 40G、单卡 80G、单机 8 卡（每卡 40G）。
> 所有脚本 `--device` 默认为 `cpu`，GPU 训练必须显式传 `--device cuda`（bf16 会随之自动开启）。
> 统一使用 `python3` 执行。

## ⚠️ 三条铁律（不遵守会静默训出废模型，日志上完全看不出来）

1. **所有阶段必须传同一份 `--config`。** stage-1 少传一次 `--config`，它会落到脚本内置的
   CPU smoke 配置（`d_model=96`），stage-2 再热启时 65 个张量因形状不符被静默跳过，
   `--load` 等于没写。判据见下方 [§2 热启自检](#2-热启自检必看)。
2. **所有阶段必须显式传 `--ckpt`。** `configs/default.yaml` 里 `train.ckpt_dir: checkpoints`，
   而三个训练脚本在「给了 `--config` 且没给 `--ckpt`」时会直接采用它——**stage1/stage2/grounding
   会全部写到 `checkpoints/final` 互相覆盖**。详见 [§8 已知问题](#8-已知问题)。
3. **stage-2 / grounding 的生产训练必须传 `--base-model`。** 不传时 LLM 段是随机初始化的
   字节级 stand-in 小模型（vocab=259），管线照常跑通、loss 照常下降、检查点照常产出，
   但训不出任何有语言能力的模型。

---

## 0. 前置准备（所有形态通用）

```bash
cd /Users/bityangzi/python/LongVideo-main
python3 tests/smoke_test.py                      # 环境自检
```

真实数据五步链路（已有 manifest 可跳过）：

```
quality_filter → extract_features(Qwen3-VL 抽 .npz) → npz_to_npy(可选，启用 mmap)
→ build_manifest 生成 pretrain.jsonl / lvb_train.jsonl → validate_dataset 校验
```

### 0.1 开训前的数据体检（30 秒，能省掉一整轮无效训练）

```bash
# ① 样本数：必须远大于 batch_size。样本数 < batch_size×8 时基本等于在过拟合单个 batch
for f in data/manifests/*.jsonl; do echo "$(wc -l < $f)  $f"; done

# ② 特征维：必须与 --base-model 的视觉塔一致（Qwen3-VL-4B = 2560）
python3 -c "
import json,numpy as np,os
r=json.loads(open('data/manifests/pretrain.jsonl').readline())
p=os.path.join('data', r['feature_ref']); a=np.load(p) if p.endswith('.npz') else None
print('feat_dim =', (a['features'] if a is not None else np.load(p)).shape[-1])"
```

`configs/default.yaml` 里的 `model.feat_dim: 768` 只是 smoke 兜底值；给了 `--manifest` 时
`align_feat_dim` 会按真实特征自动校正并打印一行 `[cfg] feat_dim 与真实特征不符：768 → 2560`。
**看到这行是正常的**；没看到说明特征文件读不到，后面会在第一个 Linear 崩。

---

## 1. 训练脚本总览

| 脚本 | 作用 | LLM 段 | 关键显存手段 |
|---|---|---|---|
| `scripts/train_stage1.py` | 时序预测自监督预训练 | **不使用 LLM**（损失只有 pred/recon/spectral） | bf16 + `eacs_chunk=32` 分块梯度检查点 |
| `scripts/train_stage2.py` | 端到端任务微调 | `--base-model` → Qwen3-VL；不给 → stand-in | 同上 + `--lora` + `--load` 热启 stage-1 |
| `scripts/train_qwen3vl.py` | Qwen3-VL 底座生产入口（不吃 `--config`） | 始终 Qwen3-VL | LoRA（默认开）+ `--grad-accum` + LLM 逐层 `grad_checkpoint` |
| `scripts/train_grounding.py` | 定位打分头 | 同 stage-2 | `--lora` + `--load` 热启 stage-2 |

显存主开关在 `configs/default.yaml`：

- `model.eacs_chunk: 32` — EACS 分块梯度检查点
- `model.llm.grad_checkpoint: true` — **40G 卡必须开**（不开时 40G 上 B=1 只能吃约 4200 帧）；80G 可改 `false` 换约 30% 速度
- `train.grad_accum: 4`、`train.bf16`
- `data.length_bucketing: true` — 长度分桶减少 padding 浪费，自带 DDP 分片逻辑

---

## 2. 热启自检（必看）

stage-2 / grounding 传了 `--load` 之后，**第一件事是读这行日志**：

```
[stage2-load] 加载 checkpoints/stage1/final：命中 81/1305 个张量（缺失 1224，多余 110，形状不匹配 6）
[stage2-load]   形状不匹配(已跳过，用初始值): projector.net.0.weight: (512,384)→(2560,384), ...
```

| 字段 | 健康值 | 异常含义 |
|---|---|---|
| **命中** | **≈ 81**（CST-SSM 段 87 个张量 − 6 个 projector） | 只有二十几个 → 配置漂移，`--load` 失效 |
| **形状不匹配** | **≈ 6，且全部是 `projector.*`** | 出现 `vision.*` / `temporal.*` → **stage-1 与 stage-2 的 `--config` 不一致** |
| **多余** | 全部是 `llm.*`（stage-1 存下的 stand-in LLM，无用） | 出现非 `llm.*` → 检查点不是本项目产出的 |
| **缺失** | 大部分是 `llm.*`（由 HF `from_pretrained` 提供，**不是随机初始化**） | — |

> `projector.*` 形状不匹配是**设计使然、可以忽略**：stage-1 没有底座（投影目标维 = YAML 的
> `llm.dim: 512`），stage-2 接 Qwen3-VL（2560），且 stage-1 的损失里根本不含 LLM 项、
> 从未训练过 projector。丢掉零损失。
>
> `vision.*` / `temporal.*` 形状不匹配则是**致命的**：那是 CST-SSM 主干，意味着热启白做。

反例（本项目实际踩过的）：

```
[stage2-load] 加载 checkpoints/stage1/final：命中 22/1305（缺失 1283，多余 75，形状不匹配 65）
[stage2-load]   形状不匹配: vision.proj.weight: (96,2560)→(384,2560), ...
                                              ^^^ stage-1 的 d_model=96，说明它没吃到 --config
```

命中的那 22 个全是标量和偏置（门控阈值、频段边界、`proj_B/proj_C.bias`、`fusion.bias`），
**没有一个权重矩阵**——stage-1 学到的表征一个字节都没传过来。

---

## 3. 单卡 40G（保守配置：batch=1 + 梯度累积 + LoRA）

```bash
# ① Stage-1 自监督预训练（不接底座；损失 = pred + recon + spectral）
python3 scripts/train_stage1.py --device cuda \
    --config configs/default.yaml \
    --manifest data/manifests/pretrain.jsonl --data-root data \
    --ckpt checkpoints/stage1 \
    --batch-size 1 --num-workers 8 --steps 20000

# ② Stage-2 任务微调（40G 必须 --lora；--base-model 不可省）
python3 scripts/train_stage2.py --device cuda \
    --config configs/default.yaml \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --ckpt checkpoints/stage2 \
    --load checkpoints/stage1/final \
    --batch-size 1 --num-workers 8 --steps 5000 --lora

# ③ Qwen3-VL 生产入口（4B + LoRA，grad_accum=8 补等效批；该脚本不吃 --config）
python3 scripts/train_qwen3vl.py --device cuda \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --ckpt checkpoints/qwen3vl \
    --batch-size 1 --num-workers 4 --grad-accum 8 --steps 5000

# ④ 定位打分头（热启 stage-2；--base-model 同样不可省）
python3 scripts/train_grounding.py --device cuda \
    --config configs/default.yaml \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/charades_train.jsonl --data-root data \
    --ckpt checkpoints/grounding \
    --load checkpoints/stage2/final \
    --batch-size 2 --num-workers 8 --steps 20000 --lora
```

## 4. 单卡 80G（放大 batch；可选关 LLM 检查点换速度）

```bash
# ① Stage-1：batch 提到 4~8（可先在 YAML 把 llm.grad_checkpoint 改 false 换 ~30% 速度）
python3 scripts/train_stage1.py --device cuda \
    --config configs/default.yaml \
    --manifest data/manifests/pretrain.jsonl --data-root data \
    --ckpt checkpoints/stage1 \
    --batch-size 4 --num-workers 8 --steps 20000

# ② Stage-2：显存宽裕可加 --no-lora 全参微调；保留 LoRA 则可用更大 batch
python3 scripts/train_stage2.py --device cuda \
    --config configs/default.yaml \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --val-manifest data/manifests/lvb_val.jsonl \
    --ckpt checkpoints/stage2 \
    --load checkpoints/stage1/final \
    --batch-size 4 --num-workers 8 --steps 5000

# ③ Qwen3-VL
python3 scripts/train_qwen3vl.py --device cuda \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --ckpt checkpoints/qwen3vl \
    --batch-size 2 --num-workers 4 --grad-accum 4 --steps 5000
```

> `--base-model` 给了之后 `--lora` 默认就是开的（`train_stage2.py` 里
> `args.lora = bool(args.base_model)`），显式写出来只是为了可读性。关掉要用 `--no-lora`。

## 5. 推理与评测（单卡即可，推理状态为 O(1)，两种显存形态通用）

```bash
# 流式推理效率演示
python3 scripts/infer_stream.py --device cuda --frames 2000

# 连续查询（亚帧定位）演示
python3 scripts/infer_query.py --device cuda --frames 64 --nquery 200

# QA 准确率评测
python3 scripts/eval_benchmark.py --mode qa \
    --ckpt checkpoints/qwen3vl/final \
    --manifest data/benchmarks/mlvu_test.jsonl --data-root data --device cuda

# 效率评测（更新率/延迟/显存峰值）
python3 scripts/eval_benchmark.py --mode efficiency \
    --ckpt checkpoints/qwen3vl/final --device cuda

# 定位推理（消费 train_grounding 产出的权重）
python3 benchmarks/infer_grounding.py \
    --manifest data/benchmarks/charades_infer.jsonl --data-root data \
    --out results/grounding_pred.jsonl \
    --ckpt checkpoints/grounding/final --config configs/default.yaml --device cuda

# 消融实验（单卡即可）
python3 scripts/ablation_study.py --device cuda --bf16 --steps 50
```

---

## 6. 单机 8 卡（每卡 40G）

当前代码启用 DDP 的方式：在 YAML 的 `train:` 段加 `ddp: true`
（`TrainConfig.ddp` 字段会被 `train_config_from_yaml` 消费）。建议复制一份专用配置：

```bash
cp configs/default.yaml configs/ddp.yaml
# 在 ddp.yaml 的 train: 段追加两行：
#   ddp: true
#   ddp_backend: nccl
```

用 `torchrun` 启动（`--device cuda` 时 Trainer 自动按 `LOCAL_RANK` 绑到各卡）：

```bash
# ① Stage-1：8 卡 × batch2 × grad_accum4 = 等效全局批 64
torchrun --standalone --nproc_per_node=8 scripts/train_stage1.py \
    --device cuda --config configs/ddp.yaml \
    --manifest data/manifests/pretrain.jsonl --data-root data \
    --ckpt checkpoints/stage1 \
    --batch-size 2 --num-workers 8 --steps 20000

# ② Stage-2（40G/卡仍建议保留 --lora）
torchrun --standalone --nproc_per_node=8 scripts/train_stage2.py \
    --device cuda --config configs/ddp.yaml \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --ckpt checkpoints/stage2 \
    --load checkpoints/stage1/final \
    --batch-size 2 --num-workers 8 --steps 5000 --lora

# ③ 定位头（同样需要 --base-model 与 --load）
torchrun --standalone --nproc_per_node=8 scripts/train_grounding.py \
    --device cuda --config configs/ddp.yaml \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/charades_train.jsonl --data-root data \
    --ckpt checkpoints/grounding \
    --load checkpoints/stage2/final \
    --batch-size 2 --num-workers 8 --steps 20000 --lora
```

---

## 7. 训练日志怎么读

一个健康的 stage-2 首屏应该依次出现这些行（缺任何一行都说明走错了路径）：

```
[cfg] feat_dim 与真实特征不符：配置 768 → 按特征校正为 2560（来源 .../lvb_train.jsonl）
[model] 接入真实底座: Qwen/Qwen3-VL-4B-Instruct（lora=True, r=64, dtype=bfloat16）
[tokenizer] 加载 Qwen3-VL 分词器: Qwen/Qwen3-VL-4B-Instruct
[stage2-load] 加载 checkpoints/stage1/final：命中 81/1305 ...        ← 见 §2
[cfg] YAML 四段已全部消费：model/train/loss/data (...)
[data] 真实 manifest: ...（N 样本, batch=1, workers=8）              ← N 必须远大于 batch
[data] 自检通过: video 占位符数 == 帧数 == 1789
[train] EACS 分块梯度检查点: 块大小=32（来自 模型配置）
[step     10] task=... pred=... update_rate=... spectral=... total=...
```

step 行各项的判读：

| 指标 | 健康范围 | 异常含义 |
|---|---|---|
| `task` | 真底座首步 **2~4**，随后下降 | **≈ 11.93 = ln(151936)**，即 LM 输出均匀分布、语言先验被摧毁。见 [§8 已知问题 3](#8-已知问题) |
| `pred` | 0.3~1.0，缓慢下降 | 不降 → EACS 时序段没学到东西 |
| `update_rate` | 初始 1.0，应在数百步内降到 0.2~0.5 | **长期贴 1.0 = 事件门控没学起来，压缩率为 0**（这是 EACS 的核心指标，必须盯） |
| `spectral` | **初始恰为 0 是正常的** | 它是 band hinge 正则，特征值初始化时本就落在目标频段内；变正说明特征值漂出频段 |

`total` 可以自己验算：`total = 1.0×task + 0.5×pred + 0.1×update_rate + 0.05×spectral`
（权重来自 YAML 的 `loss:` 段；`[cfg]` 那行只打印了 task/pred 两项，另外两项同样生效）。

---

## 8. 已知问题

1. **检查点目录冲突（stage1/2/grounding）**：`configs/default.yaml` 的
   `train.ckpt_dir: checkpoints` 会在「给了 `--config` 且没给 `--ckpt`」时被采用
   （`train_stage1.py:79`、`train_stage2.py:184`、`train_grounding.py:143`），
   **三个阶段全部写到 `checkpoints/final` 互相覆盖**。
   规避：本文档所有命令都显式传 `--ckpt`。根治：删掉 YAML 的 `train.ckpt_dir`，
   或把脚本的 fallback 改成 `os.path.join(tc.ckpt_dir, "stage1")`。

2. **`--load` 的护栏分母口径不对**：`load_checkpoint` 用
   `len(missing)/len(model_sd)` 判断是否架构不匹配，分母含 1200 多个本该由
   HF `from_pretrained` 提供的 Qwen 参数，真底座路径下这个比例天然逼近 1，
   因此阈值被放宽到 `max_missing_ratio=0.99`（`train_stage2.py:173`）。
   实测配置漂移时该比例是 98.3%，**差 0.7 个点没有触发**。
   在修好之前，**必须人工核对 §2 的那行日志**，不能依赖它自动报错。

3. **首步 `task` 可能等于 `ln(vocab)`**：`CrossModalProjector` 末层是
   `nn.LayerNorm(d_out)`（gain=1，输出 std≈1.0），而 Qwen 的 token embedding
   量级小一到两个数量级；注入的视觉 soft token 占序列 ~99%，会把注意力带偏。
   诊断：同一 batch 把 `visual_states` 置零再跑一次前向，loss 掉到 2~4 即坐实。

4. **数据分片初始化顺序缺陷（stage1/2/grounding）**：训练脚本在 `Trainer` 初始化
   （此时才 `dist.init_process_group`）**之前**调用 `build_dataloader`，导致
   `DistributedSampler` / 长度分桶采样器的分片判断 `dist.is_initialized()` 为 False——
   8 个 rank 会各自迭代全量重复数据。训练不会崩、梯度同步也正确，但吞吐等效于单卡。
   修复方式：在 `build_dataloader` 前调用 `maybe_init_distributed()` 前置初始化进程组。

5. **`train_qwen3vl.py` 目前不支持 DDP**：其 `TrainConfig` 直接由 argparse 构造，
   没有任何 `ddp` 入口；用 `torchrun` 启动会导致 8 个进程全部挤到 `cuda:0` 而 OOM。
   该脚本目前只能单卡运行。

6. **stage-1 检查点携带无用的 stand-in LLM 权重**：stage-1 的损失里不含 LLM 项
   （`losses.py` 的 pretrain 分支只有 pred/recon/spectral），但 `state_dict()` 会把
   从未训练的 stand-in LLM 张量一并存下，于是每次热启固定产生几十条「多余(被忽略)」
   告警。属噪声，不影响正确性。

---

## 9. 参数速查

| 参数 | 适用脚本 | 说明 |
|---|---|---|
| `--device cuda` | 全部 | 缺省为 `cpu`；非 cpu 时自动开 bf16 |
| `--config` | stage1/2/grounding | YAML 四段（model/train/loss/data），命令行覆盖 YAML。**各阶段必须用同一份** |
| `--base-model` | stage2/grounding | **生产必传**；不传则 LLM 段是随机初始化的 stand-in 小模型 |
| `--dtype` | stage2/grounding/qwen3vl | 底座 dtype，生产用 `bfloat16` |
| `--manifest` / `--data-root` | 全部训练 | 真实数据入口；缺省用合成数据 |
| `--val-manifest` | stage1/2/qwen3vl | 验证集；缺省跳过验证（给了则自动 `val_every=500`） |
| `--batch-size` / `--num-workers` | 全部训练 | 40G 建议 1~2 / 8；80G 可 4~8 |
| `--lora` / `--no-lora` | stage2/grounding | 给了 `--base-model` 时默认开；40G 必开 |
| `--load` | stage2/grounding/qwen3vl | 热启上一阶段检查点目录。**传了之后必须核对 §2 的命中数** |
| `--ckpt` | 全部训练 | 检查点输出目录。**必须显式传**，见 §8 已知问题 1 |
| `--grad-accum` | qwen3vl | 梯度累积，补小 batch 的等效批量 |
| `--steps` | 全部训练 | 覆盖 YAML `train.max_steps` |
| `--max-video-tokens` | stage2/grounding | 须 ≥ 单样本帧数（默认 8192，与 `data.max_frames` 对齐），否则尾部帧特征被静默丢弃 |
| `--max-frames` / `--max-text-len` | qwen3vl/eval | 同上口径 |
| `--stand-in` | qwen3vl/eval_benchmark | 无需 GPU/transformers 的管线自测模式 |

## 10. 检查点产出位置

按本文档显式传 `--ckpt` 时：

- Stage-1 → `checkpoints/stage1/final`
- Stage-2 → `checkpoints/stage2/final`
- Qwen3-VL → `checkpoints/qwen3vl/final`（每 `ckpt_every` 步另有 `stepN`）
- 定位头 → `checkpoints/grounding/final`（供 `benchmarks/infer_grounding.py --ckpt` 加载）

DDP 下仅 rank 0 保存；分片保存保证单文件 ≤ 4GB。

**不传 `--ckpt` 时不是这个布局**——见 §8 已知问题 1。确认某个检查点到底是哪个阶段产出的：

```bash
python3 -c "
import json,sys
w=json.load(open(sys.argv[1]+'/model.safetensors.index.json'))['weight_map']
print('张量数:', len(w))
print('含 stand-in LLM:', any(k.startswith('llm.layers') for k in w))" checkpoints/stage1/final
```
