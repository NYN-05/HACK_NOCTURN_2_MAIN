from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from ..config import VeriSightConfig

DEFAULT_FEATURE_ORDER = VeriSightConfig.META_MODEL_FEATURES
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "models" / "fusion" / "meta_model.json"


@dataclass(frozen=True)
class MetaModelPrediction:
    score: int
    probability: float
    confidence: float
    effective_weights: Dict[str, float]
    feature_vector: Dict[str, float]
    linear_score: float
    available: bool
    source: str


class MetaModel:
    def __init__(
        self,
        feature_order: Sequence[str],
        coefficients: Sequence[float],
        intercept: float,
        input_scale: float = 100.0,
        min_layers: int = 2,
        source: str = "builtin",
        available: bool = True,
    ) -> None:
        if len(feature_order) != len(coefficients):
            raise ValueError("feature_order and coefficients must have the same length")

        self.feature_order = tuple(feature_order)
        self.coefficients = tuple(float(value) for value in coefficients)
        self.intercept = float(intercept)
        self.input_scale = float(input_scale) if input_scale else 100.0
        self.min_layers = int(min_layers)
        self.source = source
        self.available = available

    @classmethod
    def load(cls, model_path: str | Path | None = None) -> "MetaModel":
        path = Path(model_path or DEFAULT_MODEL_PATH)
        if not path.exists():
            return cls(DEFAULT_FEATURE_ORDER, VeriSightConfig.META_MODEL_COEFFICIENTS, VeriSightConfig.META_MODEL_INTERCEPT, source=str(path), available=False)

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            feature_order = payload.get("feature_order", DEFAULT_FEATURE_ORDER)
            coefficients = payload["coefficients"]
            intercept = payload["intercept"]
            input_scale = payload.get("input_scale", 100.0)
            min_layers = payload.get("min_layers", 2)
            return cls(
                feature_order=feature_order,
                coefficients=coefficients,
                intercept=intercept,
                input_scale=input_scale,
                min_layers=min_layers,
                source=str(path),
                available=True,
            )
        except Exception:
            return cls(DEFAULT_FEATURE_ORDER, VeriSightConfig.META_MODEL_COEFFICIENTS, VeriSightConfig.META_MODEL_INTERCEPT, source=str(path), available=False)

    def predict(
        self,
        feature_scores: Mapping[str, float],
        available_layers: Sequence[str] | None = None,
    ) -> MetaModelPrediction:
        if not self.available:
            raise RuntimeError("Meta-model is unavailable")

        available_layers = list(available_layers or [name for name in self.feature_order if name in feature_scores])
        if len(available_layers) < self.min_layers:
            raise ValueError("Meta-model requires at least two available layers")

        feature_vector = {name: float(feature_scores.get(name, 50.0)) for name in self.feature_order}
        normalized_vector = np.array([feature_vector[name] / self.input_scale for name in self.feature_order], dtype=np.float32)
        coefficients = np.array(self.coefficients, dtype=np.float32)
        linear_score = float(self.intercept + float(np.dot(coefficients, normalized_vector)))
        probability = float(1.0 / (1.0 + math.exp(-linear_score)))
        score = int(round(max(0.0, min(1.0, probability)) * 100.0))
        confidence = float(abs(probability - 0.5) * 2.0)

        active_coefficients = {
            name: abs(self.coefficients[index])
            for index, name in enumerate(self.feature_order)
            if name in available_layers
        }
        active_total = sum(active_coefficients.values())
        if active_total > 0:
            effective_weights = {name: value / active_total for name, value in active_coefficients.items()}
        else:
            effective_weights = {name: 0.0 for name in available_layers}

        return MetaModelPrediction(
            score=score,
            probability=probability,
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            effective_weights=effective_weights,
            feature_vector=feature_vector,
            linear_score=linear_score,
            available=True,
            source=self.source,
        )