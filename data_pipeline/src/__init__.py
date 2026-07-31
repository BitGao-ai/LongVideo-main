"""CST-SSM 数据处理流水线（data_pipeline）。

模块：
  adaptive_sampler   内容自适应变步长采样 → 真实可变 Δt（命门①）
  extract_features   Qwen3-VL 视觉塔离线抽帧级特征 → .npz
  quality_filter     质量筛选 + 感知哈希去重
  build_manifest     统一 manifest 构建 + <video> 占位符对齐（命门②）
  convert_benchmarks 四基准 → 统一 manifest
  npz_to_npy         .npz → .npy+.ts.npy（真惰性 mmap，规模化必做）
  validate_dataset   放行前四项硬校验
  stats              时长/事件密度分布统计

设计原则：重依赖（torch/transformers/decord）一律 lazy import，缺依赖时给清晰报错或
dry-run 退化，保证 `--help` 与流程自测在纯 CPU/无依赖环境可跑。
"""
