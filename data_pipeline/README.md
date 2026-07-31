# CST-SSM 数据处理流水线（data_pipeline）

> 配套《CST-SSM-ICCV设计方案》与《实验执行清单》。本目录把"原始视频 → 可训练样本"的**筛选 / 处理 / 存储架构**落成可执行、可复现、可校验的流程。
>
> 一句话定位：**CST-SSM 的命门是"真实可变 Δt"**。本流水线的每一步都围绕"如何造出并保真地传递真实时间戳"设计——这是区别于"把特征喂 Mamba"的物理来源，也是 Theorem 1 与 CSG/VFR 杀手锏基准能否立住的前提。

---

## 0. 全景图

```
原始视频语料
  │
  ├─(A) 筛选  quality_filter.py + stats.py        docs/01_数据筛选.md
  │      来源准入 → 质量过滤 → 去重 → 时长分层 → 事件密度分层
  │
  ├─(B) 处理  adaptive_sampler.py → extract_features.py   docs/02_数据处理.md
  │      内容自适应变步长采样(真Δt) → Qwen3-VL 视觉塔离线抽特征
  │      → build_manifest.py（占位符对齐 + 标签构建）
  │
  ├─(C) 架构  npz_to_npy.py + convert_benchmarks.py       docs/03_存储架构.md
  │      特征缓存(.npy+.ts.npy, mmap友好) + 统一 manifest(jsonl) + 分片
  │
  └─(D) 校验  validate_dataset.py
         占位符数==有效帧数 / 时间戳单调 / 维度一致 / 引用可达 → 放行训练
```

产物直接被 `cst_ssm/data/dataset.py::VideoTemporalDataset` 消费（feature 模式）。

---

## 1. 目录结构

```
data_pipeline/
├── README.md                  # 本文件：全景 + 数据量 + 显卡估算导航
├── configs/
│   └── pipeline.yaml          # 全流程统一配置（抽帧/采样/存储/校验）
├── docs/
│   ├── 01_数据筛选.md          # 来源、准入、质量过滤、去重、分层
│   ├── 02_数据处理.md          # 抽帧→特征→变步长时间戳→占位符→标签
│   ├── 03_存储架构.md          # 目录规范、manifest schema、分片、mmap
│   ├── 04_数据量与算力.md       # 训练需多少数据 + 需多少显卡 + 存储估算
│   └── 05_LongVideoBench示例.md # 端到端示例（convert→抽特征→校验→infer→eval + 训练接入）
└── src/
    ├── __init__.py
    ├── adaptive_sampler.py    # 内容自适应变步长采样 → 真实可变 Δt
    ├── extract_features.py    # Qwen3-VL 视觉塔离线抽帧级特征 → .npz
    ├── quality_filter.py      # 质量筛选 + 感知哈希去重
    ├── build_manifest.py      # 统一 manifest 构建 + <video> 占位符对齐
    ├── convert_benchmarks.py  # MLVU/LVBench/VideoMME/EgoSchema → 统一 QA manifest（真实适配器）
    ├── infer_qa.py            # 四基准 QA 逐选项似然推理 → pred.jsonl（闭合评测环）
    ├── qa_eval.py             # 四基准 QA 准确率评测（按 task_type + VideoMME 分档，GATE-4）
    ├── npz_to_npy.py          # .npz → .npy+.ts.npy（真惰性 mmap，规模化必做）
    ├── validate_dataset.py    # 放行前四项硬校验
    └── stats.py               # 时长/事件密度分布统计 → 指导分层与采样
```

---

## 2. 快速上手（端到端最小闭环）

```bash
# 0) 一切以 configs/pipeline.yaml 为准
cd /Users/bityangzi/python/LongVideo-main

# 1) 质量筛选 + 去重：raw_videos/ → filtered.jsonl（保留通过的视频清单）
python -m data_pipeline.src.quality_filter \
    --video-dir data/raw_videos --out data/filtered.jsonl

# 2) 抽特征（含内容自适应变步长采样）：filtered.jsonl → features/*.npz
python -m data_pipeline.src.extract_features \
    --manifest data/filtered.jsonl --model Qwen/Qwen3-VL-4B-Instruct \
    --out data/features --sampler adaptive --patches 9

# 3) 构建训练 manifest（占位符对齐 + 标签）：+ QA 标注 → train.jsonl
python -m data_pipeline.src.build_manifest \
    --feature-dir data/features --qa data/qa_annotations.jsonl \
    --out data/manifests/train.jsonl --task qa

# 4) 规模化存储转换（真 mmap）：features/*.npz → features_npy/*.npy(+.ts.npy)
python -m data_pipeline.src.npz_to_npy --in data/features --out data/features_npy

# 5) 放行前硬校验（不过则拒绝进训练）
python -m data_pipeline.src.validate_dataset \
    --manifest data/manifests/train.jsonl --data-root data --strict

# 基准评测集（评测用，独立于训练）：标注→manifest → 抽特征 → 模型出 pred → 评准确率
python -m data_pipeline.src.convert_benchmarks --bench videomme \
    --src data/benchmarks/VideoMME.json --out data/manifests/videomme.jsonl
#   （特征同走 extract_features 抽到 features_npy/）
python -m data_pipeline.src.infer_qa --manifest data/manifests/videomme.jsonl \
    --data-root data --ckpt ckpt_final.pt --out cstssm_videomme.jsonl   # 逐选项似然
python -m data_pipeline.src.qa_eval --manifest data/manifests/videomme.jsonl \
    --pred cstssm_videomme.jsonl        # 出 整体/分task/分档(long) 准确率，GATE-4
```

> 无 GPU / 无 transformers 时，`extract_features.py --dry-run` 会退化成合成特征跑通全链路（等价于 `scripts/prepare_features.py`），用于自测流程正确性。

---

## 3. 训练需要多少数据、多少显卡（速览）

> 详细拆解、依据与存储估算见 **`docs/04_数据量与算力.md`**。此处给结论表。

### 3.1 数据量（推荐档 / 最小可投档）

| 阶段 | 用途 | 数据来源（示例） | 推荐档 | 最小可投档 |
|---|---|---|---|---|
| Stage-1 自监督预训练 | 学连续时序动力学 | InternVid / HowTo100M 子集 + 自有长视频 | **30–50 万**长视频（均 3–10 min） | **8–10 万**（或直接热启底座权重，砍此阶段） |
| Stage-2 端到端微调 | 适配下游 + 定位增强 | LLaVA-Video-178K / VideoChat2-IT + 定位集 | **40–80 万** QA（10–20 万视频） | **15–20 万** QA |
| 定位增强（喂 CSG/VFR） | 亚帧定位能力 | Charades-STA / ActivityNet-Cap / QVHighlights | **3–5 万**带毫秒标注段 | **1 万** |
| 四基准评测（仅推理） | SOTA 对比 | MLVU / LVBench / VideoMME / EgoSchema | 各自官方集（合计 ~1.2 万 QA） | 同左 |
| CSG / VFR 杀手锏 | novelty 防线 | 由带精标注视频重采样合成 | 0.5–2 万条合成 | 0.5 万 |

### 3.2 显卡（8×A100/H100-80G 单节点为基准单位）

| 用途 | GPU·h | 8×80G 满负荷 | 备注 |
|---|---|---|---|
| **数据处理·特征抽取**（离线，易并行） | **600–1,200** | ~3–6 天 | ⚠️ 训练预算之外，常被漏算 |
| 训练全量（阶段0–7） | 4,250–7,150 | 22–37 天 | 见实验清单汇总表 |
| **训练最小可投**（砍 Stage-1 + 单底座） | 2,000–2,800 | 10–15 天 | 热启底座权重 |

**最低门槛硬件**：8×A100-80G（或 8×H100-80G）单节点即可全流程跑通；显存靠 LoRA + EACS 分块检查点 + bf16 + 梯度累积压住。**推荐**：2 节点 16 卡把周期压到两周内。**存储**：特征缓存是大头，推荐档约 **8–20 TB**（P=9、float16），详见 04 文档的存储公式。

---

## 4. 与主仓库的接口契约

| 本流水线产物 | 消费方 | 契约 |
|---|---|---|
| `features/{vid}.npy` + `{vid}.ts.npy` | `VideoTemporalDataset._load_feature` | features `[L,P,d]` float16；timestamps `[L]` float32 **严格单调递增（秒）** |
| `manifests/{split}.jsonl` | `VideoTemporalDataset` | 每行一个 `VideoSample`（见 `cst_ssm/data/schema.py`） |
| prompt 内 `<video>` 数量 | `Qwen3VLLanguageModel._scatter_visual` | **== subsample 后有效帧数**（validate 强校验，见 03 文档§4） |
| `feat_dim` | `CSTSSMConfig.feat_dim` | 抽取器 out_hidden（Qwen3-VL=3584）需与配置一致 |

**红线**：占位符对齐（③）与时间戳单调（①）任一违反，训练会静默降质或定位任务错位——`validate_dataset.py --strict` 会拦截。
