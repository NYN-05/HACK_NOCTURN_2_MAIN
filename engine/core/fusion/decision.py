from __future__ import annotations

from ..config import VeriSightConfig


class DecisionEngine:
    """Maps final authenticity score to a platform decision."""

    def __init__(self, thresholds: list[tuple[int, str]] | None = None) -> None:
        self.thresholds = thresholds or VeriSightConfig.DECISION_THRESHOLDS

    def classify(self, authenticity_score: int) -> str:
        for floor, label in self.thresholds:
            if authenticity_score >= floor:
                return label
        return "REJECT"


def create_decision_engine(thresholds: list[tuple[int, str]] | None = None) -> DecisionEngine:
    return DecisionEngine(thresholds=thresholds)