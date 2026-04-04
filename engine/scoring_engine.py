from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

from configs.weights import (
    CALIBRATED_MODEL_WEIGHTS,
    ENABLE_DYNAMIC_RELIABILITY_WEIGHTING,
    LAYER_RELIABILITY_FLOOR,
)
from engine.meta_model import MetaModel

LOGGER = logging.getLogger(__name__)


@dataclass
class ScoreBreakdown:
    layer_scores: Dict[str, int]
    weighted_score: int
    effective_weights: Dict[str, float]
    layer_reliabilities: Dict[str, float]
    confidence: float
    available_layers: list[str]
    abstained: bool
    fusion_strategy: str = "weighted_average"
    meta_model_used: bool = False


class ScoringEngine:
    """Weighted fusion over normalized 0-100 authenticity scores."""

    def __init__(self, weights: Dict[str, float] | None = None, meta_model: MetaModel | None = None) -> None:
        self.weights = weights or CALIBRATED_MODEL_WEIGHTS
        self.meta_model = meta_model or MetaModel.load()

    @staticmethod
    def _clip_score(value: float) -> int:
        return int(round(max(0.0, min(100.0, value))))

    def fuse(
        self,
        normalized_scores: Dict[str, float],
        reliabilities: Dict[str, float] | None = None,
        availability: Dict[str, bool] | None = None,
    ) -> ScoreBreakdown:
        reliabilities = reliabilities or {}
        availability = availability or {}

        available_layers = [
            key
            for key in self.weights
            if availability.get(key, True) and key in normalized_scores
        ]

        if not available_layers:
            return ScoreBreakdown(
                layer_scores={},
                weighted_score=50,
                effective_weights={},
                layer_reliabilities={},
                confidence=0.0,
                available_layers=[],
                abstained=True,
                fusion_strategy="abstain",
                meta_model_used=False,
            )

        layer_scores = {key: self._clip_score(normalized_scores[key]) for key in available_layers}
        layer_reliabilities = {
            key: max(LAYER_RELIABILITY_FLOOR, min(1.0, float(reliabilities.get(key, 1.0))))
            for key in available_layers
        }

        if self.meta_model.available and len(available_layers) >= self.meta_model.min_layers:
            try:
                prediction = self.meta_model.predict(normalized_scores, available_layers=available_layers)
                return ScoreBreakdown(
                    layer_scores=layer_scores,
                    weighted_score=prediction.score,
                    effective_weights=prediction.effective_weights,
                    layer_reliabilities=layer_reliabilities,
                    confidence=prediction.confidence,
                    available_layers=available_layers,
                    abstained=False,
                    fusion_strategy="meta_model",
                    meta_model_used=True,
                )
            except Exception as exc:
                LOGGER.warning("Meta-model fusion unavailable; falling back to weighted average: %s", exc)

        active_weights = {key: self.weights[key] for key in available_layers}

        if ENABLE_DYNAMIC_RELIABILITY_WEIGHTING:
            dynamic = {
                key: active_weights[key] * layer_reliabilities[key]
                for key in available_layers
            }
            dynamic_total = sum(dynamic.values())
            if dynamic_total > 0:
                effective_weights = {key: dynamic[key] / dynamic_total for key in available_layers}
            else:
                active_total = sum(active_weights.values())
                effective_weights = {
                    key: active_weights[key] / active_total if active_total > 0 else 0.0
                    for key in available_layers
                }
        else:
            active_total = sum(active_weights.values())
            effective_weights = {
                key: active_weights[key] / active_total if active_total > 0 else 0.0
                for key in available_layers
            }

        weighted_score = sum(layer_scores[key] * effective_weights[key] for key in available_layers)
        confidence = sum(self.weights[key] * layer_reliabilities[key] for key in available_layers)

        return ScoreBreakdown(
            layer_scores=layer_scores,
            weighted_score=self._clip_score(weighted_score),
            effective_weights=effective_weights,
            layer_reliabilities=layer_reliabilities,
            confidence=round(max(0.0, min(1.0, confidence)), 4),
            available_layers=available_layers,
            abstained=False,
            fusion_strategy="weighted_average",
            meta_model_used=False,
        )
