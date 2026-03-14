from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from configs.weights import MODEL_WEIGHTS


@dataclass
class ScoreBreakdown:
    layer_scores: Dict[str, int]
    weighted_score: int


class ScoringEngine:
    """Weighted fusion over normalized 0-100 authenticity scores."""

    def __init__(self, weights: Dict[str, float] | None = None) -> None:
        self.weights = weights or MODEL_WEIGHTS

    @staticmethod
    def _clip_score(value: float) -> int:
        return int(round(max(0.0, min(100.0, value))))

    def fuse(self, normalized_scores: Dict[str, float]) -> ScoreBreakdown:
        layer_scores = {
            key: self._clip_score(normalized_scores.get(key, 50.0))
            for key in self.weights
        }
        weighted_score = sum(layer_scores[key] * self.weights[key] for key in self.weights)

        return ScoreBreakdown(
            layer_scores=layer_scores,
            weighted_score=self._clip_score(weighted_score),
        )
