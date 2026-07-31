# 05 · 端到端示例：LongVideoBench

> 以 **LongVideoBench**（长视频多选 QA，NeurIPS 2024）为例，走通 data_pipeline 全流程，并演示训练侧
> 如何吃这条真实 manifest。命令可直接复制；无 GPU 处用 `--dry-run` 验证链路。
>
> ⚠️ LongVideoBench 官方字段名随版本可能微调，首次落地请用一条真实样本核对 `convert_benchmarks.py`
> 的 `ADAPTERS["longvideobench"]`（见 §5 字段映射）。

---

## 0. 数据准备

从官方（HuggingFace `longvideobench/LongVideoBench`）下载后目录约为：

```
LongVideoBench/
├── lvb_val.json          # 验证集（含 correct_choice 答案）
├── lvb_test.json         # 测试集（无答案，答案需在线提交）
├── videos/{video_id}.mp4
└── subtitles/            # 字幕（本算法只用视觉，可忽略）
```

单条（val）结构（关键字段）：
```json
{
  "id": "q_00042",
  "video_id": "vid_abc",
  "video_path": "videos/vid_abc.mp4",
  "question": "After the person sits down, what do they pick up?",
  "candidates": ["a book", "a cup", "a phone", "keys", "a bag"],
  "correct_choice": 2,               // 0 基索引；test 集无此字段
  "question_category": "S2E",
  "level": "L2-Relation",
  "duration_group": 600,             // 秒档 {15,60,600,3600}
  "duration": 642.5
}
```

---

## 1. 标注 → 统一 QA manifest

```bash
cd /Users/bityangzi/python/LongVideo-main
python -m data_pipeline.src.convert_benchmarks --bench longvideobench \
    --src /data/LongVideoBench/lvb_val.json \
    --out data/manifests/lvb_val.jsonl --feature-subdir features_npy
```
产出每行：
```json
{"video_id":"vid_abc","query_id":"q_00042","feature_ref":"features_npy/vid_abc.npy",
 "prompt":"<video> After the person sits down...\nA. a book\nB. a cup\nC. a phone\nD. keys\nE. a bag",
 "answer":"C","answer_index":2,"options":[...],"task_type":"S2E",
 "duration_bucket":"600","video_path":"videos/vid_abc.mp4","duration":642.5,"bench":"longvideobench"}
```
- `correct_choice=2`（0 基）→ 字母 `C`；5 选项自动编到 `E`。
- `duration_group` 存为 `duration_bucket`（qa_eval 按 15<60<600<3600 数值升序报告）。
- **`video_path` 已透传**——下一步抽特征无需另建 video_id→路径映射。

---

## 2. 抽特征（按变步长采样 + Qwen3-VL 视觉塔）

`extract_features` 直接读上一步的 manifest（其含 `video_path`）：

```bash
python -m data_pipeline.src.extract_features \
    --manifest data/manifests/lvb_val.jsonl --video-root /data/LongVideoBench \
    --model Qwen/Qwen3-VL-4B-Instruct --out data/features --patches 9 --sampler adaptive
# 规模化转真 mmap：
python -m data_pipeline.src.npz_to_npy --in data/features --out data/features_npy
```
- feature_ref 已指向 `features_npy/{video_id}.npy`，与产物对齐。
- 无 GPU/transformers/decord 时加 `--dry-run` 用合成变步长特征跑通链路。

---

## 3. 放行校验

```bash
python -m data_pipeline.src.validate_dataset \
    --manifest data/manifests/lvb_val.jsonl --data-root data \
    --feat-dim 3584 --placeholder-mode single --strict
```
时间戳单调 / `<video>` 对齐 / d==3584 / 引用可达 四项硬过才放行。

---

## 4. 推理 → 评准确率（GATE-4）

```bash
# 模型逐选项似然出预测（真权重用 --ckpt；无则未训练占位，仅验链路）
python -m data_pipeline.src.infer_qa --manifest data/manifests/lvb_val.jsonl \
    --data-root data --ckpt checkpoints/stage2/final --out preds/lvb_cstssm.jsonl

# 按 question_category + duration_group 报告准确率
python -m data_pipeline.src.qa_eval --manifest data/manifests/lvb_val.jsonl \
    --pred preds/lvb_cstssm.jsonl
```
输出含 `by_duration_bucket: {15:.., 60:.., 600:.., 3600:..}`——长视频档（600/3600）是 CST-SSM 时长解耦优势的展示位。

> 完整链路：`convert_benchmarks → extract_features → validate_dataset → infer_qa → qa_eval`。

---

## 5. 字段映射（`ADAPTERS["longvideobench"]`）

| 统一字段 | LVB 源字段（按序取首个命中） | 说明 |
|---|---|---|
| video_id | `video_id` / `video_path` / `video` | 特征缓存键 |
| prompt | `question` / `question_wo_referring_query` + `candidates` | 选项字母化 |
| answer_index | `correct_choice`（**0 基**）/ `answer` | test 集缺 → -1（不计入准确率） |
| options | `candidates` / `options` | 去 `A.`/`(A)` 前缀 |
| task_type | `question_category` / `level` / `topic_category` | 分任务报告 |
| duration_bucket | `duration_group` | 数值秒档 |
| query_id | `id` / `question_id` | 与 pred 关联 |
| video_path | `video_path` | 透传给 extract_features |

---

## 6. 训练侧：让训练吃这条真实 manifest

训练数据加载已重构为统一工厂 `cst_ssm.data.build_dataloader`（合成/真实同口径），训练脚本加了
`--manifest/--data-root/--num-workers/--batch-size`：

```bash
# 真实 manifest 训练（feat_dim 从特征自动推断=3584；大模型维请配 --config）
python scripts/train_stage2.py \
    --manifest data/manifests/lvb_val.jsonl --data-root data \
    --num-workers 8 --batch-size 4 --steps 5000 --lora --config configs/default.yaml
# 无 --manifest 时回退合成数据（CPU smoke 仍可跑）
python scripts/train_stage2.py --device cpu --steps 20 --lora
```

重构要点（相对旧版内联 `SyntheticVideoDataset`）：
- **真实/合成同一入口**：`build_dataloader(LoaderConfig(manifest=...))`，manifest 有无自动切换。
- **`cycle` 无限取批**：旧 `fit` 只遍历 loader 一次，`max_steps` 超一个 epoch 会**提前停**；现用
  `cycle(loader)`，按 step 训练与 epoch 边界解耦。
- **feat_dim 自动推断**：`infer_feat_dim` 从首个特征读 d，防 `cfg.feat_dim` 与真实特征错配。
- **吞吐参数**：`num_workers/pin_memory/persistent_workers/drop_last` 全部打通。

> 诚实提示：LongVideoBench 是**评测**集，正式 Stage-2 训练应把 `--manifest` 指向你的**指令微调**
> 数据（同一 schema，见 docs/02）；这里用 LVB manifest 仅作"真实数据加载"的可跑示例。
