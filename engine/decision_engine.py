from __future__ import annotations

from configs.weights import CALIBRATED_DECISION_THRESHOLDS


class DecisionEngine:
    """Maps final authenticity score to a platform decision."""

    def __init__(self, thresholds: list[tuple[int, str]] | None = None) -> None:
        self.thresholds = thresholds or CALIBRATED_DECISION_THRESHOLDS

    def classify(self, authenticity_score: int) -> str:
        for floor, label in self.thresholds:
            if authenticity_score >= floor:
                return label
        return "REJECT"
