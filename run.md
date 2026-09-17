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
quality_filter → extract_features(Qwen3-VL 抽 .npz) → npz_to_npy(规模化必做，启用 mmap)
→ build_manifest 生成 pretrain.jsonl / lvb_train.jsonl → validate_dataset 校验
```

**逐条命令见下方 [§0.2 数据流水线](#02-数据流水线data_pipeline)。**

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

## 0.2 数据流水线（data_pipeline）

> 原始视频 → 可训练 manifest。**全部命令在仓库根目录执行**（`-m` 导入，别 `cd` 进子目录）。
> 默认值在 `data_pipeline/configs/pipeline.yaml`，命令行参数优先。
> 这一段里**只有 `extract_features` 吃 GPU**（600–1200 GPU·h 量级，常被漏算在训练预算外），其余全是 CPU/IO。

### 0.2.1 脚本总览

| `python3 -m data_pipeline.src.X` | 作用 | 产物 |
|---|---|---|
| `quality_filter` | 时长/黑帧/静止/碎剪过滤 + pHash 去重 + 防评测集泄漏 | `data/filtered.jsonl`、`data/rejected.jsonl` |
| `stats` | 时长与事件密度分布、Δt 变异系数体检 | 打印（`--plot` 存 PNG） |
| `adaptive_sampler` | 单视频变步长采样自查（正式链路由 `extract_features` 内部调用） | 打印 |
| `extract_features` | 变步长采样 + Qwen3-VL 视觉塔离线抽帧级特征 | `data/features/{vid}.npz` |
| `npz_to_npy` | `.npz` → `.npy` + `.ts.npy`（启用真 mmap） | `data/features_npy/` |
| `build_manifest` | `<video>` 占位符对齐 + 标签构建 | `data/manifests/*.jsonl` |
| `validate_dataset` | 放行前四项硬校验 | 退出码（`--strict` 非零即拦） |
| `convert_benchmarks` | 四基准官方标注 → 统一 QA manifest | `data/manifests/{bench}.jsonl` |
| `infer_qa` / `qa_eval` | 流水线自带的逐选项似然推理 / 准确率 | `pred.jsonl` / 打印 |

### 0.2.2 训练数据主链（五步）

#### ① 质量筛选 + 去重

```bash
python3 -m data_pipeline.src.quality_filter \
    --video-dir data/raw_videos \
    --out data/filtered.jsonl --rejected data/rejected.jsonl \
    --min-duration 60 --keep-static-ratio 0.10 \
    --holdout-hashes data/stats/benchmarks_phash.pkl
```

- 输出每行是 `{video_id, video_path, duration, ...}`，`video_path` 被下一步直接消费，**不需要另建 id→路径映射**。
- `--min-duration`：Stage-1 用 60，Stage-2 可放到 15。
- `--keep-static-ratio 0.10` 是**有意保留**的低运动正样本（Theorem 1(B) 的"静止段免更新"要靠它演示），不是漏筛。
- `--holdout-hashes` 是防评测集泄漏的唯一闸门，**出 SOTA 对比前必传**。
- 不传 `--video-dir` 时跑 dry-run（只演示汉明距离与静止判定，不读盘）。

#### ② （可选）分布体检

```bash
python3 -m data_pipeline.src.stats --manifest data/filtered.jsonl     # 时长/事件密度分布
python3 -m data_pipeline.src.stats --feature-dir data/features        # 抽完特征后再跑
```

第二条打印 Δt 变异系数：**CV≈0 说明采样器退化成均匀采样**，EACS 的连续变步长扫描就白做了
（`Ā=exp(λ·Δt)` 退化成固定步长，与原生 Mamba 无差别，Theorem 1 也演示不出来）。
此时回去调 `--theta` / `--dt-max`，别往下走。

#### ③ 抽特征（唯一吃 GPU 的一步）

```bash
python3 -m data_pipeline.src.extract_features \
    --manifest data/filtered.jsonl \
    --model weights/Qwen3-VL-4B-Instruct --device cuda --dtype auto \
    --out data/features \
    --sampler adaptive --theta 0.12 --coarse-fps 4.0 --dt-max 2.0 \
    --max-frames 8192 --patches 9 --max-frame-tokens 256 --chunk-frames 64 \
    --skip-existing

# 多机并行：i 从 0 起，各机同一个 --out，产物互不冲突
python3 -m data_pipeline.src.extract_features ... --shard 0/8
```

| 参数 | 说明 |
|---|---|
| `--skip-existing` | **默认关**（每次全量重抽）。断点续跑 / 增量必须显式加 |
| `--dt-max` | 静止段兜底间隔，**直接决定定位精度地板 δ/4**：2.0→0.5s，4.0→1.0s |
| `--max-frames` | 帧数**上限不是目标**；必须与 `configs/default.yaml` 的 `data.max_frames`（8192）一致，否则训练侧会再抽稀一次 |
| `--patches` | 空间池化 P：1 / 9 / 64。P=9 推荐（d=2560 fp16 时约 45 KB/帧） |
| `--video-root` | 只在 manifest 用 `rel_path` 时需要；有 `video_path` 时留空 |
| `--chunk-frames` | 视觉塔单次前向帧数，MPS/CPU 调小到 16~32 |
| `--dry-run` | 无 GPU/transformers/decord 时用合成变步长特征跑通链路（见 §0.2.4） |

- `--sampler` / `--theta` / `--coarse-fps` / `--dt-max` / `--max-frames` / `--patches` /
  `--max-frame-tokens` / `--saliency-accum` 参与**参数指纹 `sampler_fp`**：任改一个，
  `--skip-existing` 就认定旧缓存失效并重抽。反复试参数会白烧 GPU，先在小子集上定好再全量跑。
- 单视频失败被隔离进 `data/features/extract_failed.jsonl`（**追加模式**，跨多次运行累积），
  不中断整批。前几个连续失败会显式告警——那基本是路径/环境问题，别等跑完几万个再看。
- 抽完复查一次（`.npz` 里存了溯源与抽稀审计字段）：

```bash
python3 -c "
import numpy as np, glob
for p in sorted(glob.glob('data/features/*.npz'))[:3]:
    z = np.load(p)
    print(p, 'd=', int(z['meta_out_hidden']), 'P=', int(z['meta_patches']),
          'model=', str(z['meta_model']), 'L=', int(z['meta_n_sampled']),
          'raw=', int(z['meta_n_selected_raw']), 'decimate=', float(z['meta_decimate_ratio']),
          'dt_p50=', float(z['meta_dt_p50']), 'dt_p99=', float(z['meta_dt_p99']))"
```

`decimate_ratio != 1.0`（即 `raw > L`）说明 `--max-frames` 在这些视频上咬过人、
`--dt-max` 的间隔契约已失效——**定位精度地板不再是 δ/4**，写论文的定位精度声明前必须查这一项。
`d` 与 `model` 是溯源三件套的一部分：`sampler_fp` 只能回答"参数变没变"，回答不了"这批特征是谁抽的"。

#### ④ `.npz` → `.npy`（规模化必做，不是可选）

```bash
python3 -m data_pipeline.src.npz_to_npy \
    --in data/features --out data/features_npy --shard-dirs
```

理由与后果见 §6.0 ①（`.npz` 是 zip 归档，`np.load` 的 mmap 对它无效）。
`--shard-dirs` 按 `vid[:2]` 散列成两级目录（几十万文件别放一个目录）；
`--delete-src` 转完删源 `.npz`；`--shard i/N` 多机分片。

#### ⑤ 构建 manifest

```bash
# Stage-1 自监督：不给 --qa，每个特征出一条无问答样本
python3 -m data_pipeline.src.build_manifest \
    --feature-dir data/features_npy --prefer-npy --data-root data \
    --out data/manifests/pretrain.jsonl --split train \
    --placeholder-mode single --max-frames 8192

# Stage-2 QA：--qa 每行 {video_id, question, answer, [task_type]}
python3 -m data_pipeline.src.build_manifest \
    --feature-dir data/features_npy --prefer-npy --data-root data \
    --qa data/qa_annotations.jsonl --require-qa --task qa \
    --out data/manifests/lvb_train.jsonl --split train \
    --placeholder-mode single --max-frames 8192

# 定位：--qa 每行额外带 gt_start/gt_end（秒，毫秒精度），
#       answer 自动生成 "from X.XXs to Y.YYs"，同时保留结构化字段供 CSG 评测
python3 -m data_pipeline.src.build_manifest \
    --feature-dir data/features_npy --prefer-npy --data-root data \
    --qa data/charades_ann.jsonl --require-qa --task grounding \
    --out data/manifests/charades_train.jsonl --split train
```

- **`--prefer-npy` 必须与 `--feature-dir` 配套**：它只是把扫描后缀从 `.npz` 换成 `.npy`。
  指着 `data/features`（`.npz` 目录）加 `--prefer-npy`，会扫出 0 个文件、写出一个**空 manifest 且不报错**，
  直到训练时才表现为"样本数为 0"。写完立刻用 §0.1 ① 的 `wc -l` 核一遍行数。
- **`--data-root` 决定 `feature_ref` 的相对根，必须与训练脚本的 `--data-root` 一致**
  （本文档全用 `data`），否则训练时拼不出路径。
- `--require-qa`：无标注的特征直接跳过。不加的话它们会被写成空问答样本混进 QA 训练集。
- 一个视频有多条 QA 就出多行样本，共用同一份特征文件。
- `--placeholder-mode single`（推荐）prompt 只放 1 个 `<video>`，运行时按有效帧数展开，与 `max_frames` 解耦；
  `expand` 写死 `min(L, max_frames)` 个，改 `max_frames` 就要重建 manifest。**选定后 ⑥ 与训练侧必须同口径**。
- 自监督 manifest 的 `--task` 只是元数据标签：stage-1 的损失分支由 `train_stage1.py` 固定为
  `pretrain`（`tc.stage = "pretrain"`），不看每条样本的 `task_type`。

#### ⑥ 放行校验（不过就别开训）

```bash
python3 -m data_pipeline.src.validate_dataset \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --placeholder-mode single --max-frames 8192 --strict --verbose
```

四项硬校验：时间戳严格单调 / `<video>` 数对齐 / 跨样本 `feat_dim` 一致 / `feature_ref` 可达
（`.npy` 有配套 `.ts.npy`）。全过打印 `[validate] ✅ 硬校验全通过，可放行训练`；
`--strict` 下有硬错误即非零退出。软告警（Δt 退化为常数、grounding 标注越界）要加 `--verbose` 才逐条打印。

> **别传 `--feat-dim 3584`。** 不传时它从首个可读特征自动推断，这才是推荐用法。
> `3584` 是 `Qwen3VLConfig` 的**占位默认值**，被 `data_pipeline/docs/*.md`、`pipeline.yaml` 的
> `extract.out_hidden` 和 `docs/05` 的示例命令沿用至今；而 **Qwen3-VL-4B 视觉塔的真实
> `out_hidden_size` 是 2560**（与 §0.1 那行 `feat_dim → 2560` 对得上）。
> 硬传 3584 会把一批好特征判成维度不一致、`--strict` 直接拦下。

### 0.2.3 评测集分支（独立于训练）

```bash
# ① 官方标注 → 统一 QA manifest
#    --bench: videomme | mlvu | lvbench | egoschema | longvideobench
python3 -m data_pipeline.src.convert_benchmarks --bench longvideobench \
    --src /data/LongVideoBench/lvb_val.json \
    --out data/manifests/lvb_val.jsonl \
    --feature-subdir features_npy --feature-ext .npy

# ② 抽特征：manifest 已透传 video_path，直接复用主链 ③④
python3 -m data_pipeline.src.extract_features \
    --manifest data/manifests/lvb_val.jsonl --video-root /data/LongVideoBench \
    --model Qwen/Qwen3-VL-4B-Instruct --device cuda \
    --out data/features --patches 9 --sampler adaptive
python3 -m data_pipeline.src.npz_to_npy --in data/features --out data/features_npy --shard-dirs

# ③ 校验（口径同 §0.2.2 ⑥）
python3 -m data_pipeline.src.validate_dataset \
    --manifest data/manifests/lvb_val.jsonl --data-root data --strict

# ④ 逐选项似然推理 → 准确率（按 task_type + duration_bucket 分组报告）
python3 -m data_pipeline.src.infer_qa \
    --manifest data/manifests/lvb_val.jsonl --data-root data \
    --config configs/default.yaml --ckpt checkpoints/stage2/final \
    --out results/lvb_pred.jsonl --device cuda
python3 -m data_pipeline.src.qa_eval \
    --manifest data/manifests/lvb_val.jsonl --pred results/lvb_pred.jsonl

# 评测逻辑自检（用金标合成 perfect/random 两组预测，应分别 ≈1.0 和 ≈1/选项数）
python3 -m data_pipeline.src.qa_eval --manifest data/manifests/lvb_val.jsonl --demo
```

> ⚠️ **`infer_qa` 没有 `--base-model`，接过真底座的检查点不能用它评测。**
> 它只会建 stand-in LLM（`vocab=259` 的字节级小模型）。把接 Qwen3-VL 训出来的检查点喂给它，
> 只是把 CST-SSM 段加载进一个语言能力为零的壳里，照样输出一份**看不出异常的假准确率**——
> §5 里 `eval_benchmark.py` 那道 `trainable_only` 硬退出护栏**在这里不存在**。
>
> **判据**：接过 `--base-model` 训练的检查点走 §5 的
> `eval_benchmark.py --mode qa --base-model ...`；`infer_qa` 只用于 stand-in 下的链路/格式自测，
> 或本来就没接底座的检查点。
> 另外 `--config` 不传时它按 `d_model=96` 建模型——那是 smoke 尺寸，和 §2 那条
> "stage-1 少传 `--config`"是同一个坑。

### 0.2.4 无 GPU 本机自测（验证链路，不验证数值）

```bash
mkdir -p /tmp/dp
printf '{"video_id":"vid_0"}\n{"video_id":"vid_1"}\n' > /tmp/dp/filtered.jsonl

python3 -m data_pipeline.src.quality_filter                       # dry-run：去重/静止判定演示
python3 -m data_pipeline.src.extract_features --dry-run \
    --manifest /tmp/dp/filtered.jsonl --out /tmp/dp/features \
    --patches 9 --max-frames 64
python3 -m data_pipeline.src.npz_to_npy --in /tmp/dp/features --out /tmp/dp/features_npy
python3 -m data_pipeline.src.build_manifest --feature-dir /tmp/dp/features_npy \
    --prefer-npy --data-root /tmp/dp --out /tmp/dp/manifests/smoke.jsonl
python3 -m data_pipeline.src.validate_dataset \
    --manifest /tmp/dp/manifests/smoke.jsonl --data-root /tmp/dp --strict
```

`--dry-run` 合成的特征维取自 `--model` 指向的**本地目录**的 `config.json`
（`vision_config.out_hidden_size`）；给的是 HF 仓库名、读不到本地 config 时**退回 3584 并告警**。
**所以自测链路里的 d=3584 与真实抽取的 2560 无关**，别拿它去对训练配置。
真要用合成特征跑一轮 stage-1 自测，就显式 `--out-hidden 2560`（或让 `--model` 指向本地权重目录）——
否则这份 stage-1 检查点在换成真特征后维度对不上，等于白烧一轮。

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
- `model.llm.loss_chunk: 1024` — **LM 头 logits 分块，40G 卡必须开**。Qwen3-VL 词表 151936，
  一次性算 `(B,T,V)` 的 logits 峰值约 `B·T·V×12` 字节：`B=2/T=8192` 就是 **29.9GB**，
  比 4B 底座权重还大。分块后峰值 `= loss_chunk·V×10`（约 1.5GB）且**与 T 无关**。
  详见 [§6.1 显存预算](#61-显存预算8×40g)
- `train.grad_accum: 4`、`train.bf16`
- `train.ddp: false` — 单机多卡开关，`torchrun` 启动时必须为 `true`（或命令行传 `--ddp`）
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
>
> ⚠ 上面日志里的分母 **1305 是释放视觉塔之前的旧基线**。`build_cstssm_qwen3vl`
> 现在会删掉 `model.visual`（见 [§8.5](#8-已知问题)），stage-2 的 `state_dict` 不再含
> `visual.*`，故**总张量数与「缺失」数会同步下降**（那些视觉塔权重本就由 HF 提供、
> 旧日志里被计作缺失）。**「命中 ≈81」不受影响**——那是 CST-SSM 段，与视觉塔无关。

反例（本项目实际踩过的）：

```
[stage2-load] 加载 checkpoints/stage1/final：命中 22/1305（缺失 1283，多余 75，形状不匹配 65）
[stage2-load]   形状不匹配: vision.proj.weight: (96,2560)→(384,2560), ...
                                              ^^^ stage-1 的 d_model=96，说明它没吃到 --config
```

命中的那 22 个全是标量和偏置（门控阈值、频段边界、`proj_B/proj_C.bias`、`fusion.bias`），
**没有一个权重矩阵**——stage-1 学到的表征一个字节都没传过来。

> **这一条现在会在入口拦住**：三个训练脚本（`train_stage1` / `train_stage2` /
> `train_grounding`）只要**给了 `--manifest` 却拿不到 `model:` 配置段**（没传 `--config`，
> 或传的 YAML 里没有 `model:`），就直接退出，不会再让你训完几万步才发现白跑。
> 只想用冒烟尺寸（`d_model=96`）验证数据管线时，显式加 `--allow-default-config` 放行。
>
> 它与 `align_feat_dim` 是同一类守卫，但**补救方式相反**：`feat_dim` 由特征文件唯一确定，
> 能自动校正；`d_model` 是建模超参、没有唯一正确值，只能要求两个阶段用同一份 `--config`。

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
    --base-model weights/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/pretrain.jsonl --data-root data \
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
# ⚠ 接过真实底座训练的检查点**必须**同时给 --base-model，且与训练时是同一个。
#   不给的话脚本会建一个随机初始化的 stand-in LLM——旧版本会照常算出一个毫无异常迹象的
#   假准确率；现在只存可训练权重的检查点会被识别出来并直接报错退出。
python3 scripts/eval_benchmark.py --mode qa \
    --ckpt checkpoints/qwen3vl/final \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 --lora --lora-r 64 \
    --config configs/default.yaml \
    --manifest data/benchmarks/mlvu_test.jsonl --data-root data --device cuda --bf16

# 效率评测（更新率/延迟/显存峰值）
python3 scripts/eval_benchmark.py --mode efficiency \
    --ckpt checkpoints/qwen3vl/final \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 --device cuda

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

启用方式：**命令行加 `--ddp`**（或在 YAML 的 `train:` 段写 `ddp: true`，命令行优先）。
不需要另建配置文件——`configs/default.yaml` 里已有 `train.ddp: false` 作为单卡默认值。

> **忘记加 `--ddp` 会怎样**：脚本在加载底座之前就检测到 `WORLD_SIZE>1` 而 DDP 未开，
> 直接报错退出并告诉你怎么改。不会再出现"8 个进程挤 `cuda:0` 然后 OOM"那种
> 与根因无关的报错。

> **`--device` 必须写 `cuda`，不带序号**。写成 `--device cuda:0` 时，`resolve_device`
> 会原样返回它（那是为单卡指定卡号服务的），于是多个进程的 `model.to("cuda:0")` 全落到
> 0 号卡。现在这条也被拦下了：序号与本进程 `LOCAL_RANK` 不一致时，脚本在加载底座之前
> 直接退出（自洽的写法如 `LOCAL_RANK=3` 配 `cuda:3` 仍然放行）。

### 6.0 上多卡之前的两步准备（不做会吃掉主机内存）

```bash
# ① 特征转 .npy —— 必做。.npz 是 zip 归档，np.load 的 mmap 对它无效，
#    每次取样都要整载解压该视频的全部特征。8 rank × 8 worker = 64 个进程各自这么干，
#    单样本 L=4096/P=9/d=2560 的 fp16 就是 189MB，几十个在世样本足以吃掉几十 GB。
#    转完之后 mmap_mode='r' 生效，只读被 max_frames 抽稀选中的那些帧。
python3 -m data_pipeline.src.npz_to_npy --in data/features --out data/features_npy --shard-dirs
python3 -m data_pipeline.src.build_manifest \
    --feature-dir data/features_npy --prefer-npy --data-root data \
    --out data/manifests/pretrain.jsonl          # feature_ref 指向 .npy，完整用法见 §0.2.2 ⑤

# ② 确认 data.feat_dtype 保持 auto（默认）。它让特征按磁盘上的 fp16 一路传到 GPU，
#    省掉一份从来没有消费者的 fp32 副本（B=2/L=8192 约 755MB，同时 worker RSS、
#    pinned buffer、H2D 带宽全部减半）。数值零代价，见 configs/default.yaml 的注释。
```

数据侧走对了的话，首次取样**不会**出现这一行：

```
[dataset] 提示: 特征是 .npz（xxx.npz），np.load 的 mmap 对 zip 归档无效 ...
```

```bash
# ① Stage-1：8 卡 × batch2 × grad_accum4 = 等效全局批 64
torchrun --standalone --nproc_per_node=8 scripts/train_stage1.py --ddp \
    --device cuda --config configs/default.yaml \
    --manifest data/manifests/pretrain.jsonl --data-root data \
    --ckpt checkpoints/stage1 \
    --batch-size 2 --num-workers 8 --steps 20000

# ② Stage-2（40G/卡仍建议保留 --lora）
torchrun --standalone --nproc_per_node=8 scripts/train_stage2.py --ddp \
    --device cuda --config configs/default.yaml \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --ckpt checkpoints/stage2 \
    --load checkpoints/stage1/final \
    --batch-size 2 --num-workers 8 --steps 5000 --lora

# ③ 定位头（同样需要 --base-model 与 --load）
torchrun --standalone --nproc_per_node=8 scripts/train_grounding.py --ddp \
    --device cuda --config configs/default.yaml \
    --base-model Qwen/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/charades_train.jsonl --data-root data \
    --ckpt checkpoints/grounding \
    --load checkpoints/stage2/final \
    --batch-size 2 --num-workers 8 --steps 20000 --lora

# ④ Qwen3-VL 生产入口（该脚本不吃 --config，DDP 只有命令行入口）
torchrun --standalone --nproc_per_node=8 scripts/train_qwen3vl.py --ddp \
    --device cuda --model Qwen/Qwen3-VL-4B-Instruct \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --ckpt checkpoints/qwen3vl \
    --batch-size 2 --num-workers 4 --grad-accum 4 --steps 5000
```

**开训第一分钟必须核对这几行**，缺任何一行说明 8 卡没真正生效：

```
[dist] 进程组已建立: backend=nccl, world_size=8, 本 rank 绑定 cuda:0
[loaders] 分布式分片: world_size=8, rank=0
[dist] DDP: find_unused_parameters=True, broadcast_buffers=False, static_graph=False（...）
[train] EACS 分块梯度检查点: 块大小=32（来自 模型配置）
[train] EACS 分支并轴扫描: 未启用 —— 各分支状态维 N 不同（[16, 32, 64]）...
```

`[loaders]` 那行是关键。少了它意味着各 rank 在迭代**全量重复数据**——训练不会崩、
梯度同步也正确，但 8 张卡算的是同一批样本，8 卡等于 1 卡（这正是修复前的行为）。

最后两行是**如实报告，不是警告**：
- 分块梯度检查点是显存主开关，必须看到它真的开着（"关闭"就说明 YAML 没接上）。
- 分支并轴扫描在默认配置下**恒不启用**——它要求三个分支的 `n_state` 相同，而
  `DEFAULT_BRANCHES` 是 16/32/64。这是吞吐优化不是正确性问题，但别把它当成已经生效。
  要启用需统一 `n_state`，那会改变各分支的容量与谱覆盖，属建模决策。

### 6.1 显存预算（8×40G）

单卡峰值的量级构成（Qwen3-VL-4B + LoRA + `grad_checkpoint=true` + `eacs_chunk=32`
+ `loss_chunk=1024`，B=2 / T=8192）：

| 项 | 估算 | 说明 |
|---|---|---|
| 底座权重 bf16 | ~8.5 GB | 已释放未参与训练的视觉塔（特征是离线抽的），见 §8.5；实际常驻略低于此估算 |
| EACS 激活 | ~8.5 GB | 0.52 MB/帧/样本 × 8192 × 2，**修复后的第一大项** |
| LLM 逐层检查点激活 | ~4.5 GB | 36 层 × B × T × 2560 × 2B + 单层重算峰值 |
| LoRA 梯度 + 优化器 | ~2.5 GB | 装了 bitsandbytes 走 8bit Adam 可降到 ~1.3 GB |
| **LM 头 loss（分块后）** | **~1.6 GB** | 分块前是 **29.9 GB**，与 T 成正比；分块后与 T 无关 |
| 输入特征（`feat_dtype: auto`） | ~0.7 GB | fp32 时是 1.5 GB；fp16 直传省掉那份没有消费者的副本 |
| 合计 | **~26 GB** | 40G 卡有余量 |

> 这些是**按张量规模推算的估计值，不是实测**。本机无 CUDA，`tests/profile_memory_prod.py`
> 也接不了真实 4B 底座。上机后请在 stage-2 第一步之后打一次
> `torch.cuda.max_memory_allocated()/1024**3` 核实，再据此定 batch。

**验证步曾是一个独立的峰值**（已修）：`Trainer.validate` 跑在 `no_grad` 下，旧的分块闸门
以"有没有梯度"为判据，于是验证时会走一次性路径物化整块 `(B,T,V)` logits ——
`V=151936`、B=2/T=8192 约 **20 GB**，表现为"训练步好好的，一开 `--val-manifest` 就在
验证步 OOM"。现在闸门改成"调用方要不要 logits"，验证也走分块；`eval_benchmark` 的 MCQ
打分仍按需拿到完整 logits，行为不变。

**检查点也不再是磁盘与主机内存的大头**（已修）：接真实底座 + LoRA 时自动只存可训练权重
（冻结的 Qwen 权重下次由 `from_pretrained` 取回），`ckpt_every=500` 跑 5000 步从
≈88 GB 降到 ≈6 GB；`save_sharded` 的 CPU 副本也改成按分片生成，rank0 的主机内存尖峰
从"整模型 8.8 GB"降到"单个分片"。热启时会看到大量 `missing`，那是预期的——
`index.metadata.trainable_only` 标记了这一点，评测脚本也据此判断。

**关于 `data.max_frames: 8192`**：它是**保险丝不是预算**（`SamplerConfig` 的注释说得很清楚）。
长度分桶按**降序**出批（`bucketing.py:76`），最长的批第一个来——所以语料里只要有一条
命中 8192 帧的样本，显存问题会在第 0 步暴露而不是训到一半才炸。这是有意为之。
若第 0 步就 OOM，按这个顺序调：

1. 先把 `model.llm.loss_chunk` 从 1024 降到 512（峰值再降 0.8 GB，代价是 LM 头 GEMM 更碎）
2. 再把 `--batch-size` 降到 1、`--grad-accum` 提到 8（等效全局批不变，仍是 64）
3. 最后才考虑显式压 `data.max_frames`（这会真的降低时间分辨率，是精度换显存）

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

4. **stage-1 检查点携带无用的 stand-in LLM 权重**：stage-1 的损失里不含 LLM 项
   （`losses.py` 的 pretrain 分支只有 pred/recon/spectral），但 `state_dict()` 会把
   从未训练的 stand-in LLM 张量一并存下，于是每次热启固定产生几十条「多余(被忽略)」
   告警。属噪声，不影响正确性。

5. **视觉塔白占显存**（已修，2026-09-10）：`build_cstssm_qwen3vl` 此前加载完整
   `Qwen3VLForConditionalGeneration`，但训练只用解码器与 LM 头（帧特征是离线抽好的），
   `model.visual` 在 8 张卡上各占一份、DDP 建图时被广播一次。现在工厂在构造完
   `Qwen3VLLanguageModel` **之后**调用 `_release_visual_tower` 删除 `model.visual`
   （放在构造之后是因为其 `__init__` 会调 HF 的 `gradient_checkpointing_enable`，部分
   transformers 版本内部会遍历 `visual`，先删会 `AttributeError`）。启动时会打印一行
   `[qwen3vl] 已释放未使用的视觉塔 model.visual：N 个参数、约 XXX MB/卡`。
   检查点侧此前已被 `trainable_only` 排除，本修复省的是**显存与 DDP 广播**。
   需要在同一进程内在线跑视觉塔时传 `keep_visual=True` 保留（本工厂不支持该路径）。
   回归测试：`tests/regression_release_visual.py`。

   ⚠ **副作用：`state_dict` 少了全部 `visual.*` 键**，故 [§2 热启自检](#2-热启自检必看)
   日志里的**总张量数（分母）会下降**（旧基线 1305 不再适用）。健康判据不变——
   「命中 ≈81、形状不匹配只有 `projector.*`」这两条与视觉塔无关，仍照旧核对；只是
   「多余/缺失」的绝对数会随分母变化。上真底座后按新日志重新标定一次基线即可。
   （原 §8.6 与本条是同一问题的两半，已一并修复，不再单列。）

---

## 8.0 已修复（2026-08-24）—— 显存浪费 + 8 卡可行性专项

回归测试：`tests/regression_waste_20260824.py`（含每一条的逐位/相对差对拍与显存斜率）。

| # | 问题 | 修复 | 收益 |
|---|---|---|---|
| W2 | 特征在 dataset 里被 fp16 → fp32 升精度，而下游第一个算子在 autocast 下必然降回 bf16 | `data.feat_dtype: auto` 沿用磁盘 dtype；`FeatureAdapter` 入口按 autocast 状态对齐，无 autocast 的评测路径升精度兜底 | 整步留存 −21%（B=2/L=8192 约 −755MB），worker RSS / pinned / H2D 同步减半。两条路都**逐位相同** |
| D2 | `.npz` 无法 mmap，64 个 worker 各自整载解压 | 首次取样出声提示转 `.npy`；§6.0 列为必做步骤 | 主机内存与随机 IO |
| W1 | 验证步物化整块 `(B,T,V)` logits（≈20 GB），而它只需要一个标量 | 分块闸门从"有没有梯度"改成"调用方要不要 logits"；`Trainer.validate` 传 `need_logits=False` | 验证步峰值与 T 无关；训练态与 MCQ 打分行为逐字不变 |
| D1 | torchrun 下 `--device cuda:0` 让多进程挤同一张卡，只报无关的 OOM | `check_device_binding`：序号与 `LOCAL_RANK` 不一致时在加载底座前退出 | 消除一类静默失败 |
| W4 | 检查点整存 4B 冻结底座（≈8.8GB/次，5000 步 ≈88GB）；`save_sharded` 先攒满整模型的 CPU 副本 | 冻结权重可从 HF 取回时自动只存可训练部分（`index.metadata.trainable_only`）；CPU 副本按分片生成 | 磁盘 88GB → ≈6GB；rank0 主机内存尖峰 → 单个分片 |
| D3 | 8 进程各自在 CPU 物化一份底座 | 工厂支持 `device_map`，脚本加 `--load-to-device`（默认关，见下方说明）；可选 kwarg 不支持时逐个摘掉而非加载失败 | 可选 |
| D6 | 评测脚本只会建 stand-in LLM，用真实底座训出的检查点会算出无异常迹象的假准确率 | 加 `--base-model` 通路；`trainable_only` 检查点缺 `--base-model` 时**硬退出**；缺失比例阈值收紧到 2% | 消除评测无效 |
| D7 | LoRA 的 A/B 恒 fp32、底座 bf16，无 autocast 的推理路径直接 dtype 崩 | `LoRALinear.forward` 对齐**权重侧**（A/B 比激活小两个数量级）；全 fp32 路径逐位不变 | 推理路径可用 |
| W3 | stand-in 注意力显式物化 `(B,heads,T,T)`，`_safe_softmax` 又连开三份（单层约 14.5GB @ T=8704） | 改走 `scaled_dot_product_attention`，并保住"整行遮蔽 → 输出 0"的语义 | 显存 O(T²) → O(T)（T=512 实测省 35×）；梯度相对差 ≤ 2.2e-7 |
| W6 | 分块结果 `append + torch.cat`，cat 那一瞬间新旧两份同时在世 | 写进预分配缓冲（`_write_chunk`） | 省一次峰值翻倍（定位路径的提交轨迹 B=2/L=4096 是 3GB 量级）；前向与全部梯度**逐位相同** |
| W7 | 分支并轴扫描要求各分支 `n_state` 相同，而默认是 16/32/64 —— 这条吞吐优化默认恒不生效，且看不出来 | `MultiScaleEACS.merge_status()` + Trainer 启动日志如实报告 | 消除一个不成立的假设 |

**两项经评估后决定不做**（结论已写进代码注释，避免重复踩坑）：

- **W5**（融合处的 `torch.stack` 是一份纯复制，约 37–150MB）：改成逐分支加权累加实测
  **不是逐位等价**（max|Δ|=2.4e-7，einsum 走 bmm 的归约顺序与顺序累加不同）。这行在
  QA / 预训练 / 定位三条路的主干上，逐位可复现比这几十 MB 重要，故维持原样。
- **D5**（`find_unused_parameters=True` 恒开的开销）：唯一替代品 `static_graph=True` 与
  **非重入梯度检查点不兼容**，4 进程 gloo 实测直接抛
  `RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED (reducer.cpp:1660)`。
  而 `eacs_chunk` 与 `llm.grad_checkpoint` 生产上恒开，所以这个组合永远不可用。

**`--load-to-device` 为什么默认关**：训练脚本刻意把"fork DataLoader worker"排在
"CUDA 初始化"之前，而 `device_map` 会在加载底座时就建起 CUDA 上下文。本仓库的 worker
只产出 CPU 张量，实践中可用，但那是 PyTorch 警告过的组合，所以交给使用者显式选择
（`num_workers>0` 时会打印告警）。8 卡机的主机内存通常足够，不开也没问题。

**顺带纠正一处文档不准**：`CSTSSMConfig.eacs_chunk` 原注释称分块与不分块"全部参数梯度
实测 Δ=0"。实测 loss / update_rate 确实逐位相同，但**λ 的两个参数**
（`a_log_neg_real` / `a_imag`）在默认的逐分支路径上有 fp32 末位差（相对 1e-7 ~ 2e-5）：
λ 在分块循环外算一次、作为每个 checkpoint 的输入，梯度按块累加而非一次归约。
这是改动前就存在的固有效应（已用 git HEAD 对拍确认），不是缺陷，但别写进逐位断言。

---

## 8.1 已修复（2026-08-23）
以下四项此前列在已知问题里，现已修复，回归测试见
`tests/regression_ddp_loss_chunk.py`：

| 原问题 | 修复 |
|---|---|
| 数据分片初始化顺序缺陷（8 卡退化成 8 份相同的单卡训练） | 分片信息改走 `utils/distributed.dist_info()`：优先取活跃进程组，取不到就读 torchrun 环境变量。loader 因此能在进程组建立前正确分片，同时保留"CUDA 上下文晚于 worker fork"的既有顺序 |
| `train_qwen3vl.py` 不支持 DDP | 四个训练脚本统一加 `--ddp` / `--no-ddp` / `--ddp-backend` |
| 漏配 ddp 时 8 进程挤 `cuda:0` | `resolve_ddp()` 在**加载底座之前**检测 `WORLD_SIZE>1` 而 ddp 未开，直接退出并给出改法 |
| 缺 `torch.cuda.set_device()`（NCCL 通信器可能建错卡） | `maybe_init_distributed()` 保证 `set_device` 先于 `init_process_group` |

顺带修的三项（都是"分片修好之后才浮出水面"的问题）：

- 验证指标现在按 `(Σloss, N)` 跨 rank `all_reduce` 再平均。不聚合的话，分片生效后
  rank0 打印的就只是 1/8 验证集的数。
- `Trainer.save` 后加了栅栏，"检查点已落盘"有了确定时刻。
- `broadcast_buffers=False`：`RunningStandardizer` 的运行统计不再每次前向被 rank0
  覆盖，各 rank 各自累计自己分片的统计量。

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
| `--steps` | 全部训练 | 覆盖 YAML `train.max_steps`。**数的是微批不是优化步**：`grad_accum=4` 时 20000 步只有 5000 次参数更新 |
| `--ddp` / `--no-ddp` | 全部训练 | 单机多卡开关。`torchrun` 启动时必传（或 YAML `train.ddp: true`），漏了会在加载底座前直接报错退出 |
| `--ddp-backend` | 全部训练 | `nccl`(GPU，默认) \| `gloo`(CPU 多进程冒烟测试) |
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
