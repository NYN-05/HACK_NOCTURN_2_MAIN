"""
Core forensics fusion engine.
Combines meta-model, scoring, decision logic, and configuration.
"""

from .config import APIConfig, BenchmarkConfig, FusionConfig, LayerConfig, VeriSightConfig
from .fusion import (
    DecisionEngine,
    MetaModel,
    MetaModelPrediction,
    ScoreBreakdown,
    ScoringEngine,
    create_decision_engine,
    create_scoring_engine,
)

__all__ = [
    # Configuration
    "VeriSightConfig",
    "LayerConfig",
    "FusionConfig",
    "APIConfig",
    "BenchmarkConfig",
    # Fusion components
    "MetaModel",
    "MetaModelPrediction",
    "ScoringEngine",
    "ScoreBreakdown",
    "DecisionEngine",
    # Factories
    "create_scoring_engine",
    "create_decision_engine",
]
