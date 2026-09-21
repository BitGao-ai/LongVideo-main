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
  └─(D) 校验  check_data_ready.py（门禁出口，内含 validate_dataset.py 四项）
         L0 manifest基础 / L1行级深查(P/n_frames/ts有限) / L2配置对齐 / L3 loader烟雾 → 放行训练
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
    ├── dist_utils.py          # torchrun 自动分片（RANK/WORLD_SIZE→rank::world；裸cuda→cuda:LOCAL_RANK）
    ├── extract_features.py    # Qwen3-VL 视觉塔离线抽帧级特征 → .npz（--shard auto 支持 torchrun）
    ├── quality_filter.py      # 质量筛选 + 感知哈希去重
    ├── build_manifest.py      # 统一 manifest 构建 + <video> 占位符对齐
    ├── convert_benchmarks.py  # MLVU/LVBench/VideoMME/EgoSchema → 统一 QA manifest（真实适配器）
    ├── infer_qa.py            # 四基准 QA 逐选项似然推理 → pred.jsonl（--shard auto，rank0自动合并）
    ├── qa_eval.py             # 四基准 QA 准确率评测（按 task_type + VideoMME 分档，GATE-4）
    ├── npz_to_npy.py          # .npz → .npy+.ts.npy（真惰性 mmap，规模化必做；--shard auto）
    ├── validate_dataset.py    # 放行前四项硬校验
    ├── check_data_ready.py    # 训练放行门禁（L0/L1/L2/L3，一票否决；validate 的上位出口）
    └── stats.py               # 时长/事件密度分布统计 → 指导分层与采样
```

---

## 2. 快速上手（单机 8 卡目标环境）

> 以下命令**在仓库根目录执行**，并行步骤统一用 `torchrun --standalone --nproc_per_node=8 -m` 起
>（注意 `-m` 是 `torchrun` 自己的参数，含义与 `python3 -m` 相同：按模块名启动）。
> 并行能力速览（见 `src/dist_utils.py` 与 `docs/03 §5`）：
> `extract_features` 吃 GPU、`npz_to_npy` 是 CPU/IO 型、`infer_qa` 吃 GPU/CPU，
> 三者均支持 `--shard auto`（torchrun 下自动按 `RANK/WORLD_SIZE` 切 `items[rank::world]`，
> 裸 `--device cuda` 自动映射到 `cuda:LOCAL_RANK`，无需手写 `CUDA_VISIBLE_DEVICES`）；
> 手工 `--shard i/N` 仍兼容（优先级最高）；单进程跑时行为与原来一致。
> 其余（`quality_filter` 无 `--shard`——去重是全局状态，切分即错、`build_manifest`、`validate`、
> `check_data_ready`、`convert_benchmarks`、`qa_eval`）都是**单进程秒~分钟级**，跑一次即可。
> 完整坑位说明见仓库根 `run.md §0.2`，此处只给可直接跑的指令。

```bash
# 0) 仓库根 + 目录 + 环境自检
cd /Users/bityangzi/python/LongVideo-main
python3 tests/smoke_test.py
mkdir -p logs results data/raw_videos data/features data/features_npy data/manifests

# 0.1) 单机 8 卡 + 关键依赖门禁（起 8 卡前必过；不过就别往下跑，否则会静默产出空数据集）
#   - decord 在 requirements.txt 里标为“可选”，但**真实抽特征链路是硬依赖**：
#     adaptive_sampler.sample_video 与 qwen3_vl.make_reader 没有 decord 直接抛错；
#     quality_filter 更隐蔽——缺 decord 时每个视频都判 decode_error、全量被剔除且不报错。
#   - transformers 缺了 extract_features 加载不了 Qwen3-VL 视觉塔。
#   - 缺什么装什么：pip install decord transformers
python3 - <<'PY'
import importlib, sys, torch
ok = True
n = torch.cuda.device_count()
print(f"[preflight] cuda={torch.cuda.is_available()} 可见GPU={n}")
if not (torch.cuda.is_available() and n >= 8):
    print(f"[preflight] ❌ 需单机 8 卡，实测 {n} 张（改卡数只改 --nproc_per_node）"); ok = False
for m in ("decord", "transformers"):
    try:
        importlib.import_module(m); print(f"[preflight] {m} OK")
    except Exception as e:
        print(f"[preflight] ❌ 缺 {m}：pip install {m}（{e}）"); ok = False
sys.exit(0 if ok else 1)
PY

# 1) 质量筛选 + 去重：raw_videos/ → filtered.jsonl（CPU 单进程，无 --shard，别并行起 8 份）
python3 -m data_pipeline.src.quality_filter \
    --video-dir data/raw_videos --out data/filtered.jsonl --rejected data/rejected.jsonl \
    --min-duration 60 --keep-static-ratio 0.10
#   Stage-2 可把 --min-duration 放到 15；出 SOTA 对比前加 --holdout-hashes 防评测泄漏。

# 2) 抽特征（含内容自适应变步长采样）：filtered.jsonl → features/*.npz
#    torchrun 8 进程并行：--shard auto 自动按 RANK/WORLD_SIZE 切分，裸 --device cuda
#    自动映射到 cuda:LOCAL_RANK（8 进程不再全挤 cuda:0，无需手写 CUDA_VISIBLE_DEVICES）。
#    --skip-existing 断点续跑必加（默认关=全量重抽）。
#    --model 必须是**已下载到本地**的权重目录（下例 weights/Qwen3-VL-4B-Instruct）：8 个进程
#    会同时加载它，若填 HF 仓库名会 8 路并发回源下载、互踩缓存。置 OFFLINE 断网兜底；
#    若确要用 HF 仓库名，先单进程预下载一次，再删掉下面两行 export。
#    改卡数只改 --nproc_per_node（= nvidia-smi -L | wc -l），无需改其他参数。
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
torchrun --standalone --nproc_per_node=8 -m data_pipeline.src.extract_features \
    --manifest data/filtered.jsonl --model weights/Qwen3-VL-4B-Instruct --device cuda --dtype auto \
    --out data/features --sampler adaptive --theta 0.12 --coarse-fps 4.0 --dt-max 2.0 \
    --max-frames 8192 --patches 9 --max-frame-tokens 256 --chunk-frames 64 \
    --shard auto --skip-existing
#   注意：--sampler/--theta/--coarse-fps/--dt-max/--max-frames/--patches/--max-frame-tokens
#   任改一个都会使参数指纹 sampler_fp 变化、旧缓存自动失效重算。先小子集定参，再全量跑。
#   单视频失败隔离进 data/features/extract_failed.part-*-of-*.jsonl（各 rank 独立文件，
#   汇总看 cat data/features/extract_failed.part-*-of-*.jsonl），不中断整批。

# 2.1) 抽完立即核对产出：任一 rank 崩了 torchrun 即非零退出，但仍建议核对，
#      带着残缺特征往下走会让 manifest/训练静默少样本。产出+失败 < 清单即判未完成。
python3 - <<'PY'
import glob, os
man, out = "data/filtered.jsonl", "data/features"
n_man = sum(1 for l in open(man) if l.strip())
n_npz = len([f for f in os.listdir(out) if f.endswith(".npz")])
n_fail = sum(sum(1 for l in open(p) if l.strip())
             for p in glob.glob(os.path.join(out, "extract_failed*.jsonl"))
             if os.path.exists(p))
print(f"[extract-check] 清单={n_man} 产出.npz={n_npz} 失败记录={n_fail}（含历史）")
if n_npz == 0 or n_npz + n_fail < n_man:
    print("[extract-check] ❌ 产出不足：有分片没跑完，查 torchrun 各 rank 日志后再继续")
elif n_fail:
    print("[extract-check] ⚠ 存在失败视频（见 extract_failed.part-*-of-*.jsonl），确认非路径/解码问题再继续")
else:
    print("[extract-check] ✅ 清单内视频均已产出特征")
PY

# 3) 规模化存储转换（真 mmap，8 卡机上训练前必做）：features/*.npz → features_npy/*.npy(+.ts.npy)
#    .npz 是 zip 归档，np.load 的 mmap 对它无效；不转的话 8 rank×8 worker 各自整载解压，主机会被吃爆。
#    纯 CPU/IO 型：torchrun 只做分片加速，不占 GPU。8 个分片互不相交（files[rank::world]），
#    同写一个 --out 安全。--shard-dirs 按 vid[:2] 两级散列，几十万文件必加。
torchrun --standalone --nproc_per_node=8 -m data_pipeline.src.npz_to_npy \
    --in data/features --out data/features_npy --shard-dirs --shard auto
#   转完可加 --delete-src 删源 .npz 省盘（先确认下游 manifest 已指向 .npy）。

# 4) 构建训练 manifest（占位符对齐 + 标签）：等上面 wait 结束后再跑，否则 manifest 不全
#    --prefer-npy 必须与 --feature-dir data/features_npy 配套（指着 .npz 目录加它会扫出 0 文件、
#    写出空 manifest 且不报错）。--data-root 必须与训练脚本的 --data-root 一致（本文全用 data）。
#    Stage-1 自监督：不给 --qa，每个特征出一条无问答样本
python3 -m data_pipeline.src.build_manifest \
    --feature-dir data/features_npy --prefer-npy --data-root data \
    --out data/manifests/pretrain.jsonl --split train \
    --placeholder-mode single --max-frames 8192
#    Stage-2 QA（示例）：--qa 每行 {video_id, question, answer, [task_type]}，多条 QA 共用同一特征
# python3 -m data_pipeline.src.build_manifest \
#     --feature-dir data/features_npy --prefer-npy --data-root data \
#     --qa data/qa_annotations.jsonl --require-qa --task qa \
#     --out data/manifests/lvb_train.jsonl --split train \
#     --placeholder-mode single --max-frames 8192

# 5) 放行门禁（L0 manifest/L1 行级/L2 配置/L3 loader，一票否决；不过则拒绝进训练）
python3 -m data_pipeline.src.check_data_ready \
    --manifest data/manifests/pretrain.jsonl --data-root data \
    --config configs/default.yaml --pipeline-config data_pipeline/configs/pipeline.yaml
#   定位任务加 --strict-grounding（标注越界升级为硬错误）；极大 manifest 先加 --no-smoke 快速扫静态项。
#   底层四项硬校验仍可单独跑：python3 -m data_pipeline.src.validate_dataset \
#       --manifest data/manifests/pretrain.jsonl --data-root data \
#       --placeholder-mode single --max-frames 8192 --strict
#   注意：别传 --feat-dim 3584（那是 Qwen3VLConfig 占位默认值；4B 真实 2560）。
#   不传则从首个可读特征自动推断，这才是推荐用法。

# 基准评测集（评测用，独立于训练）：标注→manifest → 抽特征 → 模型出 pred → 评准确率
python3 -m data_pipeline.src.convert_benchmarks --bench longvideobench \
    --src /data/LongVideoBench/lvb_val.json \
    --out data/manifests/lvb_val.jsonl --feature-subdir features_npy --feature-ext .npy
#   评测视频抽特征：manifest 已透传 video_path，直接复用第 2 步的 torchrun 命令，
#   把 --manifest 换成 data/manifests/lvb_val.jsonl（加 --video-root /data/LongVideoBench 仅当 manifest 用 rel_path 时）。
#   转 .npy 同第 3 步。校验口径同第 5 步。
#   推理（同样支持 torchrun：各 rank 写独立 part，rank0 等齐自动合并成 --out；
#   接过真底座的检查点必须走仓库根 scripts/eval_benchmark.py，
#   infer_qa 只用于 stand-in 链路自测，详见 run.md §0.2.3）：
torchrun --standalone --nproc_per_node=8 -m data_pipeline.src.infer_qa \
    --manifest data/manifests/lvb_val.jsonl --data-root data --config configs/default.yaml \
    --ckpt checkpoints/stage2/final --out results/lvb_pred.jsonl --device cuda --shard auto
python3 -m data_pipeline.src.qa_eval --manifest data/manifests/lvb_val.jsonl \
    --pred results/lvb_pred.jsonl        # 出 整体/分task/分档(long) 准确率，GATE-4
```

> 无 GPU / 无 transformers 时，`extract_features.py --dry-run` 会退化成合成特征跑通全链路（等价于 `scripts/prepare_features.py`），用于自测流程正确性。此时用 `--out-hidden 2560`（或让 `--model` 指向本地权重目录）与真实链路对齐维度，否则合成特征训出的 stage-1 换真特征后不可用。

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
| `feat_dim` | `CSTSSMConfig.feat_dim` | 由真实特征自动推断（Qwen3-VL-4B 视觉塔实测 2560；`3584` 只是 `Qwen3VLConfig` 占位默认值，别硬传） |

**红线**：占位符对齐（③）与时间戳单调（①）任一违反，训练会静默降质或定位任务错位——`validate_dataset.py --strict` 会拦截。
