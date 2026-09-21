# CST-SSM：事件自适应连续时空状态空间模型 — 端到端长视频理解实现

**简体中文** | [English](README_EN.md)

> 设计方案对应 `CST-SSM-ICCV设计方案.tex/.md`。本仓库把方案中的**每个技术创新点**落成可读、高内聚低耦合、可训练、可推理的工程代码（纯 PyTorch + 变步长/门控 CUDA 内核），CPU 即可 `python tests/smoke_test.py` 跑通全链路。

---

## 1. 这是什么

一个把视频当作**连续时间信号**建模的长视频理解框架。核心算子 **EACS（Event-Adaptive Continuous Scan）** 用真实物理 Δt 参数化状态演化，并用**预测编码式事件驱动稀疏更新**只在"事件"处更新状态——从而算力正比于**事件内容**、显存正比于**状态维**（与视频时长无关），且支持**任意时刻连续查询**。

四段式端到端架构（训练/推理逻辑一致，无外挂采样）：

```
帧/特征 ─▶ 空间编码 ─▶ 连续时序建模(EACS) ─▶ 跨模态投影 ─▶ LLM(交叉注意力)
          spatial_      MultiScaleEACS         projector      VisualConditionedLM
          encoder       (短/中/长三尺度)                       (文本自注意力+视觉交叉注意力)
```

### 创新点 → 代码位置映射

| 设计方案创新点 | 实现位置 |
|---|---|
| ① 连续 ZOH 离散化（可变 Δt，复数对角 SSM） | `cst_ssm/ops/discretization.py` |
| ② 多尺度 HiPPO 谱初始化（τ=-1/Re(λ)） | `cst_ssm/ops/spectral_init.py` |
| ③ 事件驱动稀疏更新（预测编码残差门控） | `cst_ssm/modules/event_gate.py` |
| **C1 EACS 主算子**（①+③融合，保持输入 ZOH 预测） | `cst_ssm/modules/eacs.py` |
| 多尺度三分支 + 输入门控融合 | `cst_ssm/modules/multiscale.py` |
| **C3 任意时刻连续查询**（突破 δ/4 地板） | `cst_ssm/modules/continuous_query.py` |
| 并行 associative scan（Mamba-2 SSD 对应） | `cst_ssm/ops/scan.py` |
| **变步长选择性扫描 CUDA 内核** | `cst_ssm/ops/csrc/eacs_scan_cuda.cu` · `scan_cuda.py` |
| **门控 EACS cell 融合前向内核（推理）** | `cst_ssm/ops/csrc/eacs_cell_cuda.cu` · `eacs_gated_scan.py` |
| **门控 cell 可微融合训练（autograd.Function）** | `cst_ssm/ops/eacs_cell_train.py` |
| 非门控连续 SSM 基线（用 CUDA 内核） | `cst_ssm/modules/continuous_ssm.py` |
| **Qwen3-VL 视觉塔 + LLM 接入** | `cst_ssm/integrations/qwen3_vl.py` |
| 空间窗口编码 / LLM 门控交叉注意力 | `cst_ssm/modules/spatial_encoder.py` / `llm_interface.py` |
| 两阶段训练（未来预测+掩码重构 / 任务+L_pred+更新率+谱正则）| `cst_ssm/models/cst_ssm_model.py` · `train/losses.py` |
| **残差暴涨鲁棒兜底**（§4.4） | `cst_ssm/modules/eacs.py`（`robust_guard`）|
| **消融基线开关 + ε-Pareto**（离散固定/学习步长·随机/稠密门控·随机谱初始化）| `eacs.py`/`event_gate.py` · `scripts/ablation_pareto.py` |
| **QLoRA / 通用矩阵指数（Scale-and-Square）** | `utils/lora.py` · `ops/matrix_exp.py` |
| 显存优化（LoRA/QLoRA/检查点/ε·T 退火）+ 分片检查点(≤4G) | `cst_ssm/utils/{memory,lora,checkpoint}.py` |

---

## 2. 安装

```bash
pip install torch>=2.1 safetensors>=0.4 numpy pyyaml     # 核心（必需）
# 可选加速/生产：transformers bitsandbytes mamba-ssm timm（见 requirements.txt，缺失自动降级）
```

一键自检（零数据，CPU）：
```bash
python tests/smoke_test.py
```

---

## 3. 数据架构（如何存放数据）

CST-SSM 把视频建模为**带真实时间戳的帧序列**，真实 Δt 是一等公民。推荐**特征缓存**模式（视觉骨干离线抽好，训练只读特征，省显存/算力）。

### 3.1 目录布局

```
data_root/
├── features/                      # 每视频一个 .npz（内存映射读取）
│   ├── vid0000.npz                #   features: float16 [L, P, d]  (L帧, P patch/帧, d维)
│   ├── vid0001.npz                #   timestamps: float32 [L]       (每帧真实秒级时间戳, 单调递增)
│   └── ...
├── manifests/                     # 训练/评测清单（jsonl，一行一样本）
│   ├── pretrain.jsonl             #   stage-1 自监督（可只需 feature_ref）
│   ├── finetune.jsonl             #   stage-2 任务微调（含 prompt/answer）
│   ├── csg.jsonl / vfr.jsonl      #   杀手锏基准（见 benchmarks/）
│   └── val.jsonl
└── (可选) raw_videos/, frames/    # 像素模式：原视频或抽帧图目录
```

### 3.2 特征文件格式（`.npz`）

| 键 | dtype | 形状 | 说明 |
|---|---|---|---|
| `features` | float16 | `[L, P, d]` | 帧级 patch 特征（视觉骨干输出）；P=1 表示每帧已池化为单向量 |
| `timestamps` | float32 | `[L]` | 每帧真实时间戳（秒），**必须单调递增**（连续查询依赖） |

> 用 float16 存储减半磁盘/内存。注：`mmap` 对 `.npz` 归档无效（会整载解压该成员）；超大特征建议改用分离 `.npy`（真 mmap 惰性索引，见 `dataset._load_feature`）。

### 3.3 Manifest 记录 schema（`VideoSample`，见 `cst_ssm/data/schema.py`）

```json
{"video_id": "vid0000",
 "feature_ref": "features/vid0000.npz",
 "prompt": "<video> When does the key event happen?",
 "answer": "from 12.30s to 18.90s",
 "task_type": "grounding",              // qa | grounding | caption | causal
 "gt_start": 12.30, "gt_end": 18.90,    // 定位任务标注（毫秒级，可选）
 "duration": 240.0, "split": "train"}
```

像素模式：把 `feature_ref` 换成 `frame_dir`（帧图目录），并设 `DataConfig(mode="pixel")`。

### 3.4 从真实视频构造（一键示例）

```bash
python scripts/prepare_features.py --out data_demo --n 8 --frames 40   # 生成 .npz + manifest 示例
```
生产替换：把 `prepare_features.py` 里的随机特征换成你的视觉骨干（CLIP/SigLIP/InternVideo…）逐帧抽取的特征即可。大规模建议进一步用 WebDataset/tar 分片或 LMDB 打包 `features/`。

> 单机 8 卡生产链路（吃 GPU 的是抽特征）：`torchrun --standalone --nproc_per_node=8 -m
> data_pipeline.src.extract_features --shard auto` 并行抽（自动分片+自动绑卡），
> 抽完必须转 `.npy`（启用真 mmap，否则 8 rank×8 worker 主机内存爆炸），详见 §4。

### 3.5 读取

```python
from cst_ssm.data import VideoTemporalDataset, DataConfig, make_loader
ds = VideoTemporalDataset(DataConfig(manifest="data_root/manifests/finetune.jsonl",
                                     feat_dim=768, feat_patches=1), data_root="data_root")
loader = make_loader(ds, batch_size=4)      # 变长 L/T 自动 padding + 掩码
```

---

## 4. 快速上手（单机 8 卡目标环境）

> 默认按**单机 8 卡（每卡 40G）**给指令，8 卡并行利用率拉满。
> 通用三铁律：① 各阶段用**同一份 `--config`**；② 必须显式传 `--ckpt`（否则多阶段互覆盖）；
> ③ stage-2 / grounding 生产必须传 `--base-model`。完整说明与单卡命令见 `run.md`。

### 4.0 数据准备（8 卡并行，CPU 步骤单进程即可）

```bash
# 质量筛选（CPU，单进程）
python3 -m data_pipeline.src.quality_filter \
    --video-dir data/raw_videos --out data/filtered.jsonl --rejected data/rejected.jsonl \
    --min-duration 60 --keep-static-ratio 0.10

# 抽特征（吃 GPU 的步骤，600–1200 GPU·h）：torchrun 8 进程并行，--shard auto 自动分片
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --standalone --nproc_per_node=8 -m data_pipeline.src.extract_features \
    --manifest data/filtered.jsonl --model weights/Qwen3-VL-4B-Instruct --device cuda \
    --out data/features --sampler adaptive --theta 0.12 --coarse-fps 4.0 --dt-max 2.0 \
    --max-frames 8192 --patches 9 --max-frame-tokens 256 --chunk-frames 64 \
    --shard auto --skip-existing

# .npz → .npy（规模化必做，启用真 mmap；CPU/IO 型，torchrun 只做分片加速不占 GPU）
torchrun --standalone --nproc_per_node=8 -m data_pipeline.src.npz_to_npy \
    --in data/features --out data/features_npy --shard-dirs --shard auto

# 构建 manifest + 放行校验（单进程，秒级）
python3 -m data_pipeline.src.build_manifest \
    --feature-dir data/features_npy --prefer-npy --data-root data \
    --out data/manifests/pretrain.jsonl --split train \
    --placeholder-mode single --max-frames 8192
python3 -m data_pipeline.src.validate_dataset \
    --manifest data/manifests/pretrain.jsonl --data-root data \
    --placeholder-mode single --max-frames 8192 --strict
```

### 4.1 训练 / 推理（8 卡 DDP）

```bash
# stage-1 时序预测自监督预训练：8卡×batch2×grad_accum4 = 等效全局批 64
torchrun --standalone --nproc_per_node=8 scripts/train_stage1.py --ddp \
    --device cuda --config configs/default.yaml \
    --manifest data/manifests/pretrain.jsonl --data-root data \
    --ckpt checkpoints/stage1 --batch-size 2 --num-workers 8 --steps 20000

# stage-2 端到端任务微调（LoRA 省显存，从 stage-1 热启；--base-model 不可省）
torchrun --standalone --nproc_per_node=8 scripts/train_stage2.py --ddp \
    --device cuda --config configs/default.yaml \
    --base-model weights/Qwen3-VL-4B-Instruct --dtype bfloat16 \
    --manifest data/manifests/lvb_train.jsonl --data-root data \
    --ckpt checkpoints/stage2 --load checkpoints/stage1/final \
    --batch-size 2 --num-workers 8 --steps 5000 --lora

# 流式推理 / 连续查询（推理状态 O(1)，单卡即可）
python3 scripts/infer_stream.py --device cuda --frames 2000
python3 scripts/infer_query.py --device cuda --frames 64 --nquery 200
```
用配置文件：`python3 scripts/train_stage2.py --config configs/default.yaml`。
开训首分钟核对 `[dist] world_size=8` + `[loaders] 分布式分片: world_size=8`（缺后者=8 卡算重复数据，详见 `run.md §6`）。

### 4.2 训练目标（设计 §4.4）

- **stage-1 自监督**（`pretrain_forward`）：未来帧特征预测 MSE **+ 掩码帧重构**（随机掩码 `mask_ratio` 帧输入，从时序上下文重构原始特征）；`pretrain_loss = λ_pred·pred + λ_recon·recon + λ_spec·谱`。
- **stage-2 任务微调**（`forward`+`finetune_loss`）：`λ_task·L_task + λ_pred·L_pred + λ_upd·更新率 + λ_spec·谱`。
- **退火**：门控温度 T（前期软→后期硬）与 ε（`eps_anneal_start<1` 前期多更新学动力学→后期压稀疏），由 `TrainConfig.{t_start,t_end,eps_anneal_start,eps_anneal_end}` 控制（默认 ε 不退火）。
- **鲁棒兜底**：`CSTSSMConfig(eacs_robust_guard=True)`——连续多帧残差阶跃暴涨时临时降 ε 进高密度更新。

### 4.3 消融（设计 §6.3，一套代码切开关）

```python
CSTSSMConfig(
    eacs_disc_mode="fixed",       # 组1：离散固定步长（忽略真实 Δt）
    # eacs_disc_mode="learned",   # 组2：输入依赖学习步长（Mamba 风格，非物理时间）
    eacs_gate_kind="random",      # 组4：随机跳更新（对照残差驱动 "event"）；"always"=稠密无门控
    eacs_use_spectral_init=False, # 组3：随机初始化（对照谱初始化）
)
```
组5（扫描 ε 的误差-更新率 Pareto，验证 Theorem 1）：`python scripts/ablation_pareto.py`。
消融/兜底模式自动回退纯序列 cell（融合/CUDA 快路仅支持 continuous+event）。

---

## 5. 训练期显存优化（模型单文件 ≤ 4GB）

| 手段 | 位置 | 效果 |
|---|---|---|
| **LoRA 冻结基座** | `utils/lora.py` | 可训练参数 <10%（smoke test 实测 7.5%） |
| **QLoRA**（可选，4bit 量化基座） | `utils/lora.py::apply_qlora`（无 bnb 自动回退 LoRA） | 基座显存 ↓↓ |
| **EACS 分块梯度检查点** | `modules/eacs.py`（`chunk_size`） | 序列扫描按块重算，激活显存 ↓ |
| **bf16 混合精度** | `utils/memory.py`（SSM 核心仍 float32 保稳） | 显存/带宽 ↓ |
| **梯度累积 + set_to_none** | `train/trainer.py` | 等效大 batch，峰值显存 ↓ |
| **8bit AdamW**（可选） | `utils/memory.py`（bitsandbytes） | 优化器状态显存 ↓ |
| **门控温度 T + ε 退火** | `utils/memory.py` | 前期学动力学、后期压稀疏率 → 算力 ↓ |
| **分片 safetensors** | `utils/checkpoint.py` | **保证单文件 ≤ 4GB**（默认 3.8GB/片，自动处理权重绑定） |

分片检查点（需求 2）用法：
```python
from cst_ssm.utils import save_sharded, load_sharded, verify_shards
save_sharded(model.state_dict(), "ckpt/", max_shard_bytes=int(3.8*1024**3))  # 每片≤3.8GB<4GB
verify_shards("ckpt/", hard_limit_bytes=4*1024**3)   # 校验每个文件≤4GB
model.load_state_dict(load_sharded("ckpt/"), strict=False)
```
自动处理**权重绑定/共享存储**（如 `lm_head` 与 `embed` 绑定）：只存一份 + 别名索引，加载时重建。

---

## 6. 代码结构（高内聚低耦合）

```
cst_ssm/
├── ops/          数值算子：discretization / spectral_init / scan / matrix_exp
│                   + CUDA：csrc/{eacs_scan,eacs_cell}_cuda.cu · scan_cuda · eacs_gated_scan · eacs_cell_train
├── modules/      event_gate / eacs / continuous_ssm / multiscale / continuous_query
│                   spatial_encoder / projector / llm_interface
├── models/       cst_ssm_model（四段式端到端 + 消融/兜底开关）
├── train/        losses / trainer（两阶段 + T/ε 退火）
├── data/         schema / dataset / collate
├── utils/        checkpoint(分片≤4G) / lora(+QLoRA) / memory / config
└── integrations/ qwen3_vl（视觉塔 + LLM 接入）
scripts/    train_stage1/2 · infer_stream/query · prepare_features · ablation_pareto
tests/      smoke_test · regression_fixes · regression_features
configs/ default.yaml    docs/ DEPLOY.md    benchmarks/ CSG·VFR
```

**低耦合设计**：
- 各段只经张量接口通信；`VisionBackbone` / `LLMBackbone` 为结构化 `Protocol`（`modules/llm_interface.py`），可换任意骨干。
- 数值算子层（`ops/`）不依赖任何网络模块，可独立单测。
- 扫描后端可插拔：默认纯 PyTorch `associative_scan_diag`，生产可替换为 Mamba-2 可变 Δt CUDA 内核，接口不变。

---

## 7. 接入 Qwen3-VL + CUDA 变步长内核（生产）

### 7.1 接入 Qwen3-VL 系列（`cst_ssm/integrations/qwen3_vl.py`）

**当前默认外接模型**：自包含 stand-in（`VisualConditionedLM` + `ByteTokenizer`，非预训练，仅供 smoke test）。生产用 Qwen3-VL 替换视觉塔 + LLM：

```python
from cst_ssm.integrations import build_cstssm_qwen3vl, Qwen3VLVisionFeatureExtractor
# 端到端模型：Qwen3-VL 视觉特征 → EACS 连续时序压缩 → Qwen3-VL LLM（LoRA 冻结基座）
model = build_cstssm_qwen3vl("Qwen/Qwen3-VL-4B-Instruct", d_model=768, lora=True)
```

集成方式（与 CST-SSM"先时序压缩再喂 LLM"一致）：
- **视觉端**：`Qwen3VLVisionFeatureExtractor` 用 Qwen3-VL 视觉塔离线抽帧级特征（`feat_dim = 视觉塔 out_hidden_size`，默认 3584，从 `hf.config` 动态读取）→ 存 `.npz` 特征缓存 → EACS 做**时长无关**的连续时序压缩。
- **语言端**：`Qwen3VLLanguageModel` 把 EACS 压缩后的视觉状态（每帧 1 向量，远少于 Qwen 原生 帧×patch 视觉 token）经 **`<video>` 占位符位置注入 `inputs_embeds`**（`video_token_id=151656`），再走 Qwen3 解码器 + LM 头。相比一次性喂全部帧 patch，KV 规模大幅下降 —— 与 EACS 的效率目标一致。
- **LoRA**：`QWEN3_LORA_TARGETS`（`q/k/v/o/gate/up/down_proj`）只训低秩增量 + CST-SSM 时序/投影小参数，基座冻结。
- 数据侧：prompt 内放 `Lv` 个 `<video>` 占位符（`Lv`=帧数）；`.npz` 特征由视觉塔离线产出。

> 依赖 `transformers>=4.57`（含 Qwen3-VL）。语言端注入逻辑已用 HF 接口 stub 在 CPU 验证（占位符散射 + loss + 反向）。M-RoPE 对注入状态的完整时间对齐为可选精修项（见文件内注释）。

#### 不同大小 / 变体能否任意切换？—— 架构层面可，权重复用有边界

`build_cstssm_qwen3vl(model_name=...)` **换个模型名即可**：`hidden_size`、`out_hidden_size`、`vocab_size`、`video_token_id` 全部从 `hf.config` **动态读取**，CST-SSM 的 `projector`/`EACS` 按维度自动重建，无需改代码。三类切换情形：

| 切换情形 | 可复用 | 需重建/重训 |
|---|---|---|
| 同模型不同精度、Instruct↔Thinking（**hidden 相同**） | 全部（CST-SSM 权重 + 特征缓存） | 仅换 LLM 权重 |
| **不同大小**（hidden 不同，如 4B↔8B↔32B） | EACS 时序核心（在 `d_model` 空间，与 LLM 解耦）；视觉特征缓存（若 `out_hidden_size` 相同） | `projector`(d_model→hidden) + LoRA（hidden 变→形状变；projector 仅两层 MLP，重训代价小） |
| **Dense↔MoE**（如 4B↔30B-A3B） | 同上；注入走 `get_input_embeddings`/文本 forward/`lm_head` 通用接口，MoE 亦支持 | projector + LoRA；大模型需多卡 |

**要点**：
- **新建即插即用**：`build_cstssm_qwen3vl` 换 name 就换模型，维度自适应。
- **权重复用边界**：EACS/视觉端在 `d_model` 空间与 LLM 解耦 → 可跨模型复用；`projector` 与 LLM `hidden_size` 强绑定 → 换 hidden 需重训（小代价）。LoRA 随 LLM 重训。
- **特征缓存**：`feat_dim` 绑定视觉塔 `out_hidden_size`——换视觉塔维度需重抽特征（同代系列视觉塔常一致，切换前先核对 `vision_config.out_hidden_size`）。
- **占位符/常量**：`video_token_id` 已从 `hf.config` 读取，天然适配变体差异。

### 7.2 CUDA 变步长内核（`cst_ssm/ops/csrc/`）

两个内核（详见 `csrc/README.md`）：
1. **变步长选择性扫描** `eacs_scan_cuda.cu`（fwd+bwd）——非门控连续 SSM 线性扫描，加速 `ContinuousSSMLayer`。
```python
from cst_ssm.ops import selective_scan     # GPU→CUDA内核；CPU→纯PyTorch回退（数值等价）
h = selective_scan(dA, dBu)                 # (B,L,H,N) complex，一阶线性递推
```
2. **门控 EACS cell 融合前向** `eacs_cell_cuda.cu`（仅 fwd/推理）——事件门控稀疏更新的**流式推理**，
   自动接入 `EACSLayer.forward` 推理快路（`not training and x.is_cuda and 内核可用`），显存 O(B·H·N) 与时长无关。
3. **门控 cell 可微融合训练** `ops/eacs_cell_train.py`（`GatedEACSCellFunction`）——训练用 autograd.Function：
   forward 无梯度顺序扫描（可用融合 CUDA 前向内核），backward **逆时重算 + 逐步 autograd 求 VJP** 的反向递推，
   只携带状态伴随（**图显存 O(1)、与 L 无关**，正确构造无需手推公式）。开启：`CSTSSMConfig(eacs_fused_train=True)`。

- 反向伴随 conj 约定与 `torch.autograd` 对拍（<1e-6）；门控 cell 前向用纯 PyTorch 镜像 vs `EACSLayer` eval 对拍（<1e-6，含混合跳过）；可微训练 cell 用 `gradcheck`(double) + 与朴素全图 autograd 逐元素对拍（~1e-15）。均**无需 GPU 验证**。
- 训练：`eacs_fused_train=True` 走可微融合 cell（常数图显存，等价序列 cell）；否则默认序列 cell（+可选分块检查点）。
- 构建：JIT（首次调用自动编译）或 `FORCE_CUDA=1 pip install -e .`。无 GPU/编译失败自动回退，功能不受影响。

### 7.3 其它可替换项
- 预训练视觉骨干：`FeatureAdapter` 直接吃离线特征；或把 `WindowedSpatialEncoder` 换成 timm/CLIP/SigLIP。
- 其它 HF 因果 LM：`VisionBackbone`/`LLMBackbone` Protocol + `CSTSSMModel(cfg, vision=..., llm=...)` 覆盖注入。
- **部署**（vLLM / TensorRT-LLM）：接入点与流程见 `docs/DEPLOY.md`（LLM 段交推理框架，视觉+EACS 作轻量前处理）。

---

## 8. 关键实现说明与边界

- **保持输入的 ZOH 预测**（`eacs.py`）：跳过时 `x̂=Āx_π+B̄u_π`（保持上次输入），静态场景收敛稳态、残差恒 0、无虚假刷新——这是设计方案 Theorem 1(B) 干净成立的前提，也是相对 V1.0"自由演化"的正确性修正。
- **事件门控产生稀疏的前提**：需要 SSM 先学会预测（stage-1）。**未训练模型在随机数据上更新率≈1 是正确的**（无法预测 → 都是事件）；训练后/真实冗余场景更新率显著下降（见 `infer_stream.py`）。门控机制本身正确性由 `EventGate` 单测保证（预测好→跳过、ε 单调控制稀疏率）。
- **复数对角 SSM**：状态复数、SSM 核心在 float32 计算（S4D 惯例），参数均为实数（谱分量 `a_log_neg_real/a_imag`、`log_dt_scale`），故可被 safetensors 保存。
- **连续查询 δ/4 地板**：`continuous_query.py` 在帧间任意 t* 求值，配合 `benchmarks/` 的 CSG/VFR 验证突破离散分辨率下界。
- 纯 PyTorch 序列扫描用于参考实现与门控路径；大 L 生产建议接 CUDA 内核。

---

## 9. 自检与验证

```bash
python tests/smoke_test.py          # 核心ops→模块→模型→训练→分片→连续查询 全链路
python tests/regression_fixes.py    # code-review 修复回归（LoRA 冻结 / masked 更新率 / 检查点单更新）
python tests/regression_features.py # 补全功能回归（掩码重构 / stage2 pred / ε退火 / 6消融 / matrix_exp / QLoRA / EACS反向VJP）
```
覆盖：并行/序列扫描一致、ZOH 组合性、谱频段、端到端前反向、LoRA 占比、分片≤上限+round-trip、亚帧连续查询、可微融合训练等价、**EACS 反向 VJP 数值正确（双精度 gradcheck + 全图 autograd 逐元素对拍）**、全部补全功能。部署接入见 `docs/DEPLOY.md`。
