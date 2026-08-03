"""网络模块层。"""
from __future__ import annotations

from .event_gate import EventGate, RunningStandardizer
from .eacs import EACSLayer, EACSOutput
from .continuous_ssm import ContinuousSSMLayer
from .multiscale import MultiScaleEACS, MultiScaleOutput
from .continuous_query import ContinuousQuery
from .spatial_encoder import WindowedSpatialEncoder, FeatureAdapter
from .projector import CrossModalProjector
from .cgu import CGU, CausalEMA, TokenDistiller, CPIBDistill
from .diff_kv import (DiffKVConfig, LowRankResidualHead, DifferentialKVCache,
                      CPISparseAttention, diffkv_reconstruction_loss)
from .llm_interface import (
    GatedCrossAttention, VisualConditionedLM, LLMConfig,
    VisionBackbone, LLMBackbone,
)

__all__ = [
    "EventGate", "RunningStandardizer",
    "EACSLayer", "EACSOutput",
    "ContinuousSSMLayer",
    "MultiScaleEACS", "MultiScaleOutput",
    "ContinuousQuery",
    "WindowedSpatialEncoder", "FeatureAdapter",
    "CrossModalProjector",
    "CGU", "CausalEMA", "TokenDistiller", "CPIBDistill",
    "DiffKVConfig", "LowRankResidualHead", "DifferentialKVCache",
    "CPISparseAttention", "diffkv_reconstruction_loss",
    "GatedCrossAttention", "VisualConditionedLM", "LLMConfig",
    "VisionBackbone", "LLMBackbone",
]
