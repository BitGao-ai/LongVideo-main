# CST-SSM CUDA 内核

本目录含两个 CUDA 内核：

| 内核 | 文件 | 用途 |
|---|---|---|
| 变步长对角复数选择性扫描（fwd+bwd） | `eacs_scan_cuda.cu` | 非门控连续 SSM 的线性扫描（训练+推理），加速 `ContinuousSSMLayer` |
| 门控 EACS cell 融合前向（仅 fwd/推理） | `eacs_cell_cuda.cu` | 事件门控稀疏更新的**流式推理**，加速 `EACSLayer` eval 路径 |

---

## 一、变步长选择性扫描 `eacs_scan_cuda.cu`

实现可变 Δt 的对角复数选择性扫描（设计方案 §6），把 Mamba-2 固定 Δ 扫描改造为按**真实物理 Δt**
逐步演化（`a_k = exp(λ·Δt_k)` 由上层逐步算好传入）。

## 算法
- 前向：`h_k = a_k ⊙ h_{k-1} + b_k`，并行 (B·H·N) 通道、线程内沿 L 顺序。
- 反向（线性递推伴随）：`s_l = grad_h_l + conj(a_{l+1})·s_{l+1}`；`grad_b_l = s_l`；`grad_a_l = s_l·conj(h_{l-1})`。
  该 conj 约定已用纯 PyTorch 镜像在 CPU 与 `torch.autograd` 对拍验证（grad 误差 <1e-6）。

## 构建（三选一）

1. **JIT（零配置，推荐调试）**：首次调用 `selective_scan` 于 GPU 张量时，`ops/scan_cuda.py`
   自动 `torch.utils.cpp_extension.load` 编译并缓存。无需手动构建。

2. **AOT（随包安装）**：
   ```bash
   FORCE_CUDA=1 pip install -e .          # 或本机可见 CUDA 时直接 pip install -e .
   ```

3. **单独编译**（验证工具链）：
   ```bash
   python -c "from torch.utils.cpp_extension import load; \
     load(name='eacs_scan_cuda', sources=['cst_ssm/ops/csrc/eacs_scan_cuda.cu'], verbose=True)"
   ```

## 前置
- CUDA 工具链（nvcc）与 PyTorch CUDA 版本匹配；`nvcc --version` 与 `torch.version.cuda` 一致。
- 无 GPU / 编译失败：`selective_scan` 自动回退纯 PyTorch `associative_scan_diag`（数值等价），功能不受影响。

## 使用点
- `cst_ssm/modules/continuous_ssm.py::ContinuousSSMLayer`（非门控连续 SSM 基线）直接调用 `selective_scan`。
- 也可在任何一阶线性递推处替换 `associative_scan_diag` → `selective_scan`。
- 注意：EACS 的**事件门控稀疏更新**含数据依赖控制流，非纯线性递推，仍走 `eacs.py` 的序列 cell
  （其 CUDA 融合为后续工作；本内核加速的是连续 SSM 的线性扫描部分）。

## 性能取向
当前内核为"通道并行 + 时间顺序"，正确且对大 (B·H·N) 已能饱和 GPU。若需进一步加速，可按 Mamba-2
SSD 做分块并行前缀扫描（chunked scan），接口 `scan_fwd/scan_bwd` 保持不变即可替换。

---

## 二、门控 EACS cell 融合前向 `eacs_cell_cuda.cu`

复刻 `modules/eacs.py::EACSLayer._step` 的 segment 形式逐步扫描（保持输入 ZOH 预测 + 预测编码残差
门控），**一次内核跑完整段 L**：
- 每个 block 处理一个 batch 元素，沿 L 顺序（门控依赖状态，无法跨步并行）；
- 块内 256 线程按通道 H 网格跨步并行；残差范数按 H 跨线程树规约；committed 状态 (h_π,t_π,u_π,B_π,C_π)
  常驻全局，显存 **O(B·H·N)（与视频长度无关）**——适配流式长视频推理。
- 残差里 `obs=(u-μ)/σ, pred=(ŷ-μ)/σ ⇒ obs-pred=(u-ŷ)/σ`（μ 抵消），与 PyTorch 路径一致。

**范围**：仅前向/推理（硬门控）。训练走 `eacs.py` 的 autograd 序列 cell（软门控 STE），因为门控含
数据依赖控制流、可微融合是更大工程。封装 `ops/eacs_gated_scan.py`，自动接入 `EACSLayer.forward`
的推理快路（`not training and x.is_cuda and 内核可用`），否则回退序列 cell。

**验证**：纯 PyTorch 镜像逐字复刻本内核分阶段计算，与 `EACSLayer` eval 序列 cell 在 CPU 对拍，
低/高 eps（全更新 / 70% 更新混合跳过）下 y/gate/resid 误差均 <1e-6（无需 GPU）。

**构建**：JIT（首次推理自动编译）或 AOT（`setup.py` 已含两个 `.cu`）。

## 性能取向（门控 cell）
"batch 并行 + 时间顺序 + 块内通道并行"，对大 B·H 饱和 GPU；单条超长视频（B=1）可进一步做
warp 级流水/多 block 协作，接口 `eacs_cell_fwd` 保持不变。
