from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

from engine import DecisionEngine, ScoringEngine
from interfaces import CnnInterface, GanInterface, OcrInterface, VitInterface

LOGGER = logging.getLogger(__name__)


class VerificationOrchestrator:
    """Central inference pipeline: model execution, fusion, and final decision."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parent.parent)

        self.cnn = CnnInterface(self.project_root)
        self.vit = VitInterface(self.project_root)
        self.gan = GanInterface(self.project_root)
        self.ocr = OcrInterface(self.project_root)

        self.scoring = ScoringEngine()
        self.decision = DecisionEngine()

    def load_models(self) -> None:
        for key, loader in (
            ("cnn", self.cnn.load),
            ("vit", self.vit.load),
            ("gan", self.gan.load),
            ("ocr", self.ocr.load),
        ):
            try:
                loader()
            except Exception:
                LOGGER.exception("%s model failed to load at startup", key)

    def run(self, image_path: str | Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        started = time.perf_counter()
        image_path = str(image_path)
        metadata = metadata or {}

        layer_outputs: Dict[str, Dict[str, Any]] = {}
        layer_scores: Dict[str, float] = {}

        for key, model_call in (
            ("cnn", lambda: self.cnn.predict(image_path)),
            ("vit", lambda: self.vit.predict(image_path)),
            ("gan", lambda: self.gan.predict(image_path)),
            ("ocr", lambda: self.ocr.predict(image_path, metadata=metadata)),
        ):
            model_started = time.perf_counter()
            try:
                output = model_call()
            except Exception as exc:
                LOGGER.exception("%s model failed", key)
                output = {
                    "score": 50.0,
                    "raw": {"error": str(exc), "fallback": "neutral_score"},
                }

            output["processing_time_ms"] = int((time.perf_counter() - model_started) * 1000)
            layer_outputs[key] = output
            layer_scores[key] = float(output.get("score", 50.0))

        fused = self.scoring.fuse(layer_scores)
        decision = self.decision.classify(fused.weighted_score)

        response = {
            "authenticity_score": fused.weighted_score,
            "decision": decision,
            "layer_scores": fused.layer_scores,
            "layer_outputs": layer_outputs,
            "processing_time_ms": int((time.perf_counter() - started) * 1000),
        }

        LOGGER.info(
            "verify_complete score=%s decision=%s duration_ms=%s",
            response["authenticity_score"],
            response["decision"],
            response["processing_time_ms"],
        )
        return response
