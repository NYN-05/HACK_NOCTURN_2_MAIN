"""
VeriSight Forensics Engine - Simplified, Enhanced Architecture
Public API for core fusion, configuration, and preprocessing.
"""

# Core fusion and configuration
from engine.core import APIConfig, BenchmarkConfig, DecisionEngine, FusionConfig, LayerConfig, MetaModel, MetaModelPrediction, ScoreBreakdown, ScoringEngine, VeriSightConfig

# Preprocessing
from engine.preprocessing import load_image, preprocess_all, preprocess_cnn, preprocess_clip, preprocess_vit, preprocess_yolo

# Inference adapters
from engine.inference import CnnInterface, GanInterface, OcrInterface, VitInterface

__all__ = [
    # Configuration classes
    "VeriSightConfig",
    "LayerConfig",
    "FusionConfig",
    "APIConfig",
    "BenchmarkConfig",
    # Core fusion components
    "MetaModel",
    "MetaModelPrediction",
    "ScoringEngine",
    "ScoreBreakdown",
    "DecisionEngine",
    # Preprocessing functions
    "load_image",
    "preprocess_cnn",
    "preprocess_vit",
    "preprocess_clip",
    "preprocess_yolo",
    "preprocess_all",
    # Layer interfaces (optional)
    "CnnInterface",
    "VitInterface",
    "GanInterface",
    "OcrInterface",
]
