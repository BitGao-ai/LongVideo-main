"""CST-SSM: Event-Adaptive Continuous-Time State Space Models for hour-long video.

顶层包。子模块：
    ops/       —— 数值算子：连续ZOH离散化、多尺度谱初始化、并行/序列扫描
    modules/   —— 网络模块：EACS、事件门控、多尺度融合、连续查询、空间编码、投影、LLM接口
    models/    —— 端到端顶层模型
    train/     —— 损失、训练器、两阶段入口
    data/      —— 数据schema、数据集、collate
    utils/     —— 配置、注册表、显存工具、分片检查点(≤4G)

设计对应文档：CST-SSM-ICCV设计方案.tex
"""
from __future__ import annotations

__version__ = "0.1.0"
