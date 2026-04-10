"""
Central configuration for VeriSight forensics engine.
Consolidates all model weights, thresholds, and operational settings.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class LayerConfig:
    """Per-layer configuration."""
    timeout_ms: int = 3500
    weight: float = 0.25
    reliability_floor: float = 0.15


@dataclass
class FusionConfig:
    """Score fusion configuration."""
    enable_meta_model: bool = True
    enable_dynamic_weighting: bool = True
    enable_early_exit: bool = True
    early_exit_threshold: float = 95.0
    early_exit_min_reliability: float = 0.90
    min_layers_required: int = 2


@dataclass
class APIConfig:
    """API server configuration."""
    endpoint: str = "/api/v1/verify"
    max_concurrent_requests: int = 4
    request_timeout_ms: int = 15000
    enable_cache: bool = True
    cache_max_items: int = 256
    response_schema_version: str = "2.0.0"


@dataclass
class BenchmarkConfig:
    """Performance targets."""
    min_accuracy: float = 0.85
    max_p90_latency_ms: int = 5000
    max_p99_latency_ms: int = 8000


class VeriSightConfig:
    """Master configuration class with all settings."""

    # Layer weights and timeouts (retuned Apr 2026 - all 4 layers now operational)
    # CNN-dominant weights: CNN highest, ViT second, GAN and OCR lower and similar
    LAYER_WEIGHTS: Dict[str, float] = {
        "cnn": 0.45,          # Core CNN detector (45%) - HIGHEST
        "vit": 0.30,          # Vision Transformer analysis (30%) - SECOND
        "gan": 0.125,         # GAN forgery detection (12.5%) - LOWER
        "ocr": 0.125,         # Text-based forensics (12.5%) - LOWER & SIMILAR
    }

    LAYER_TIMEOUTS: Dict[str, int] = {
        "cnn": 3500,
        "vit": 3500,
        "gan": 3500,
        "ocr": 4500,
    }

    # Decision thresholds (retuned for balanced 4-layer fusion)
    # Adjusted to work well with new balanced weights
    DECISION_THRESHOLDS: List[Tuple[int, str]] = [
        (85, "AUTO_APPROVE"),    # High confidence (was 88)
        (65, "FAST_TRACK"),      # Good confidence (was 64)
        (45, "SUSPICIOUS"),      # Moderate concern (was 44)
        (0, "REJECT"),           # Low authenticity
    ]

    # Meta Model configuration (retuned for CNN-dominant weights)
    META_MODEL_FEATURES: Tuple[str, ...] = ("cnn", "vit", "gan", "ocr")
    META_MODEL_COEFFICIENTS: Tuple[float, ...] = (4.5, 3.0, 1.25, 1.25)  # Proportional to new layer weights
    META_MODEL_INTERCEPT: float = -3.0
    META_MODEL_INPUT_SCALE: float = 100.0

    # Reliability settings (retuned for better robustness)
    LAYER_RELIABILITY_FLOOR: float = 0.10  # Lower floor - trust scores more (was 0.15)

    # Fusion settings (all 4 layers now operational)
    ENABLE_DYNAMIC_RELIABILITY_WEIGHTING: bool = True
    ENABLE_META_MODEL_FUSION: bool = True

    # Early exit settings (balanced 4-layer approach)
    ENABLE_EARLY_EXIT: bool = True
    EARLY_EXIT_CNN_THRESHOLD: float = 92.0  # Higher threshold for 4-layer consensus (was 95.0)
    EARLY_EXIT_MIN_RELIABILITY: float = 0.85  # More achievable with 4 layers (was 0.90)

    # API Cache settings
    ENABLE_REQUEST_CACHE: bool = True
    REQUEST_CACHE_MAX_ITEMS: int = 256

    # API settings
    VERIFY_ENDPOINT: str = "/api/v1/verify"
    MAX_CONCURRENT_REQUESTS: int = 4
    REQUEST_TIMEOUT_MS: int = 15000

    # Response schema
    RESPONSE_SCHEMA_VERSION: str = "2.0.0"

    # Allowed image types
    ALLOWED_IMAGE_TYPES: frozenset = frozenset({
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
    })

    # Performance targets
    BENCHMARK_MIN_ACCURACY: float = 0.85
    BENCHMARK_MAX_P90_LATENCY: int = 5000
    BENCHMARK_MAX_P99_LATENCY: int = 8000

    @classmethod
    def get_layer_config(cls, layer: str) -> LayerConfig:
        """Get configuration for a specific layer."""
        return LayerConfig(
            timeout_ms=cls.LAYER_TIMEOUTS.get(layer, 3500),
            weight=cls.LAYER_WEIGHTS.get(layer, 0.25),
            reliability_floor=cls.LAYER_RELIABILITY_FLOOR,
        )

    @classmethod
    def get_fusion_config(cls) -> FusionConfig:
        """Get fusion configuration."""
        return FusionConfig(
            enable_meta_model=cls.ENABLE_META_MODEL_FUSION,
            enable_dynamic_weighting=cls.ENABLE_DYNAMIC_RELIABILITY_WEIGHTING,
            enable_early_exit=cls.ENABLE_EARLY_EXIT,
            early_exit_threshold=cls.EARLY_EXIT_CNN_THRESHOLD,
            early_exit_min_reliability=cls.EARLY_EXIT_MIN_RELIABILITY,
            min_layers_required=2,
        )

    @classmethod
    def get_api_config(cls) -> APIConfig:
        """Get API configuration."""
        return APIConfig(
            endpoint=cls.VERIFY_ENDPOINT,
            max_concurrent_requests=cls.MAX_CONCURRENT_REQUESTS,
            request_timeout_ms=cls.REQUEST_TIMEOUT_MS,
            enable_cache=cls.ENABLE_REQUEST_CACHE,
            cache_max_items=cls.REQUEST_CACHE_MAX_ITEMS,
            response_schema_version=cls.RESPONSE_SCHEMA_VERSION,
        )


# Backward compatibility exports
MODEL_WEIGHTS = VeriSightConfig.LAYER_WEIGHTS
LAYER_TIMEOUT_MS = VeriSightConfig.LAYER_TIMEOUTS
CALIBRATED_MODEL_WEIGHTS = VeriSightConfig.LAYER_WEIGHTS
CALIBRATED_DECISION_THRESHOLDS = VeriSightConfig.DECISION_THRESHOLDS
ENABLE_DYNAMIC_RELIABILITY_WEIGHTING = VeriSightConfig.ENABLE_DYNAMIC_RELIABILITY_WEIGHTING
ENABLE_META_MODEL_FUSION = VeriSightConfig.ENABLE_META_MODEL_FUSION
ENABLE_REQUEST_CACHE = VeriSightConfig.ENABLE_REQUEST_CACHE
REQUEST_CACHE_MAX_ITEMS = VeriSightConfig.REQUEST_CACHE_MAX_ITEMS
ENABLE_EARLY_EXIT = VeriSightConfig.ENABLE_EARLY_EXIT
EARLY_EXIT_CNN_SCORE_THRESHOLD = VeriSightConfig.EARLY_EXIT_CNN_THRESHOLD
EARLY_EXIT_MIN_RELIABILITY = VeriSightConfig.EARLY_EXIT_MIN_RELIABILITY
LAYER_RELIABILITY_FLOOR = VeriSightConfig.LAYER_RELIABILITY_FLOOR
RESPONSE_SCHEMA_VERSION = VeriSightConfig.RESPONSE_SCHEMA_VERSION
VERIFY_ENDPOINT = VeriSightConfig.VERIFY_ENDPOINT
