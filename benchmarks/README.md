# CST-SSM 杀手锏基准：CSG & VFR-Stress

> 配套《CST-SSM-ICCV设计方案》§5 与命题 2（离散分辨率下界 δ/4）。
> 这两个基准的唯一使命：**正面击穿"直接把 Δt 喂给 Mamba 不就行了"** —— 证明存在离散范式在\*结构上\*做不好、而连续查询能做的任务。

## 为什么需要它们（理论根基）

**命题 2（Discrete Resolution Floor）**：任何"只能在采样时刻 {t_k} 输出定位结果"的模型（含 Δt-fed Mamba、一切固定/可变步长离散 SSM），期望绝对定位误差 ≥ **δ/4**（δ=局部采样间隔），且该下界**不可**由喂入 Δt 或增大模型容量消除——它源于"输出被限制在栅格上"。连续查询算子输出取值于 ℝ，不受此约束。

→ 于是我们设计让**答案精度细于输入帧距 δ** 的任务：离散基线在指标上撞 δ/4 地板，CST-SSM 穿透。

---

## 一、CSG：亚帧连续时间定位（Continuous Sub-frame Grounding）

**构造** (`csg_build.py`)：从 Charades-STA / ActivityNet-Captions / QVHighlights 的定位标注出发，
为每个样本按给定 δ（或非均匀网格）生成"模型可见的输入帧时间戳"，保留**高精度(ms) GT** 边界。

**指标** (`csg_eval.py`)：
| 指标 | 含义 | 期望（离散 vs 连续）|
|---|---|---|
| **MAE/floor 比值** | 亚帧 MAE ÷ (δ/4) | **离散≈1（触底）；连续≪1（穿透）** ← 头号判据 |
| 亚帧 MAE | 边界端点绝对误差均值(s) | 离散≈δ/4 触底；连续 ≪δ/4 |
| R@0.7 / R@0.9 | 紧 IoU 召回 | 连续更高，尤其 R@0.9 |
| Floor-Break Rate | MAE<δ/4 的样本占比 | 离散≈**0.5**(对称于地板)；连续→1.0 |
| grid-snapped MAE | 预测吸附到网格后的 MAE | 画出 δ/4 地板线 |

> **为何头号判据是 MAE/floor 而非 Floor-Break Rate**：离散模型每个边界的最近栅格距离服从 `U[0,δ/2]`，中位数恰为 δ/4，故 ~50% 样本\*偶然\*低于地板——per-sample Floor-Break 对离散≈0.5，不是 0。真正不可伪造的是**均值**：Prop.2 保证离散 mean MAE ≥ δ/4，故 `MAE/floor≥1`；连续可 <1。

**可证伪预测**：`--snap-to-grid` 把任意预测吸附到网格即模拟离散模型——其 mean MAE 应卡在 δ/4（`MAE/floor≈1`）。**若 CST-SSM 的 `MAE/floor` 不能显著<1，则本文核心主张被证伪。**

**核心图**：`MAE vs δ`（多方法），叠加 `y=δ/4` 地板线 → 离散曲线贴地板、CST-SSM 曲线穿透。

---

## 二、VFR-Stress：极端可变帧率鲁棒性

**构造** (`vfr_build.py`)：同一视频内帧率剧烈波动(0.1~30fps)，用 `level∈[0,1]` 控制不均匀度（CV 随 level 单调增），**总帧数≈恒定（等预算）**。关键：同一 (视频,level) 生成**唯一网格**，所有对照方法看**完全相同的帧**，差异只来自"是否用真实 Δt"。

**指标** (`vfr_eval.py`)：每方法一条 `定位 MAE vs CV` 曲线，最小二乘拟合**斜率 dMAE/dCV**。
- 对照：CST-SSM（真实 Δt）vs Δt-fed Mamba（Δt 当门控标量）vs Uniform-Δ Mamba。
- **预测**：CST-SSM 斜率显著最小（最鲁棒）；Uniform-Δ 最敏感。

---

## 快速自测（合成数据，验证指标逻辑，无需模型/GPU）

```bash
cd benchmarks
# CSG：构造 demo 集 → 对比"连续 vs 离散"两种合成预测
python csg_build.py --demo --delta 1 2 4 --out csg_demo.jsonl
python csg_eval.py  --manifest csg_demo.jsonl --demo

# VFR：构造 demo 集 → 三方法斜率对比
python vfr_build.py --demo --levels 0 0.25 0.5 0.75 1.0 --out vfr_demo.jsonl
python vfr_eval.py  --manifest vfr_demo.jsonl --demo
```
预期：CSG demo 中 Discrete 的 `MAE/floor_ratio≈1`（触底）、`Floor_Break_Rate≈0.5`；Continuous 的 `MAE/floor_ratio≪1`（穿透）、`Floor_Break_Rate→1.0`。VFR demo 中斜率 `CST-SSM < Δt-fed < Uniform-Δ`。

---

## 接真实模型（三步）

> build 只产 `input_grid`（时间戳）；跑真实模型前需先把 grid 在**真实视频上重采样**成特征缓存。

1. **物化特征**（`materialize.py`，桥接 `encode_video_at`）：按 `input_grid` 到真实视频精确取帧、
   过 Qwen3-VL 视觉塔抽特征，时间戳**原样用 grid**（保住 VFR 非均匀 Δt）；同 (视频,网格) 跨 query
   共享，保证对照方法看**完全相同的帧**。产出特征 `.npz` + 推理 manifest `*_infer.jsonl`。
   ```bash
   python materialize.py --manifest csg_delta2.jsonl --video-root /data/charades \
       --model Qwen/Qwen3-VL-4B-Instruct --patches 9 --out csg_feats --out-manifest csg_infer.jsonl
   ```
2. **出预测**：用你的模型读 `*_infer.jsonl`（只喂其 `feature_ref` 的帧），连续查询模式下**在帧间任意
   时刻**求解边界，写出 `{query_id, pred_start, pred_end}` 的 jsonl。参考实现 `infer_grounding.py`
   （复用 `ContinuousQuery`；亚帧边界=相关度阈值穿越的线性插值，穿透 δ/4 的核心逻辑在此、可单测）：
   ```bash
   python infer_grounding.py --manifest csg_infer.jsonl --data-root . --ckpt ckpt_final.pt \
       --query-factor 8 --out cstssm_csg.jsonl        # 缺 --ckpt 则为未训练占位，仅跑通链路
   ```
3. **评测**（仍以原 `csg_*/vfr_*.jsonl` 为准，含 floor/grid，按 query_id 关联 pred）：
   ```bash
   python csg_eval.py --manifest csg_delta2.jsonl --pred cstssm_csg.jsonl
   python csg_eval.py --manifest csg_delta2.jsonl --pred baseline_csg.jsonl --snap-to-grid  # 离散对照
   python vfr_eval.py --manifest vfr.jsonl --pred cstssm.jsonl --name CST-SSM \
                      --pred uni.jsonl --name Uniform-Δ --pred dtf.jsonl --name Δt-fed
   ```

> 无 GPU/decord/transformers 时 `materialize.py --dry-run` 用合成特征跑通链路（时间戳仍=grid），
> 逻辑由 `tests/regression_bench_materialize.py` 全覆盖。

## 标注转换约定

把三个源数据集转成统一 jsonl（每行）：
```json
{"video_id":"v1","query_id":"v1_0","query":"...","gt_start":12.34,"gt_end":18.90,"duration":240.0,"fps":30.0,"split":"test"}
```
- Charades-STA：直接有秒级 start/end。
- ActivityNet-Captions：timestamps 为秒。
- QVHighlights：relevant windows 转为 (start,end)；多窗口取主窗口或拆多条。

## 文件

| 文件 | 作用 |
|---|---|
| `common.py` | 共享：IoU、边界 MAE、栅格吸附、δ/4 floor、CV、JSONL IO |
| `csg_build.py` / `csg_eval.py` | CSG 构造 / 评测（亚帧 MAE、Floor-Break Rate）|
| `vfr_build.py` / `vfr_eval.py` | VFR 构造 / 评测（MAE-CV 斜率）|
| `materialize.py` | grid → 真实视频重采样特征 + 推理 manifest（桥接 `encode_video_at`）|
| `infer_grounding.py` | 连续查询定位推理 → pred.jsonl（亚帧边界=阈值穿越插值，闭合评测环）|

## 局限与诚实边界
- 命题 2 假设事件时刻在栅格内近似均匀；对严重聚集的事件分布，地板估计需按局部密度修正（`Segment.local_delta` 已按 GT 邻近间隔计算）。
- 合成 demo 仅验证**指标与流水线正确性**，不代表真实模型表现。
- VFR 等预算约束下，极端 level 可能出现段内单帧，`bursty_grid` 已做 ≥1 帧/段兜底。
