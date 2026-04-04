from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from engine.interfaces import CnnInterface, GanInterface, OcrInterface, VitInterface
from engine.pipeline.orchestrator import VerificationOrchestrator
from layer4.orchestrator import Layer4OcrScorer


class _GoodLayer:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def load(self) -> None:
        pass

    def predict(self, *_args, **_kwargs):
        return {"score": 65.0, "raw": {"source": "test"}}


class _BadLayerMissingRaw(_GoodLayer):
    def predict(self, *_args, **_kwargs):
        return {"score": 65.0}


class _FakeFused:
    def __init__(self, layer_scores, reliabilities) -> None:
        self.weighted_score = 73.5
        self.layer_scores = dict(layer_scores)
        self.layer_reliabilities = dict(reliabilities)
        self.effective_weights = {name: 0.25 for name in layer_scores}
        self.confidence = 0.91
        self.available_layers = list(layer_scores)
        self.abstained = False


class _CountingScoringEngine:
    calls = 0

    def fuse(self, layer_scores, reliabilities, availability=None):
        _CountingScoringEngine.calls += 1
        return _FakeFused(layer_scores, reliabilities)


class _CountingDecisionEngine:
    calls = 0

    def classify(self, _score):
        _CountingDecisionEngine.calls += 1
        return "FAST_TRACK"


class EngineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        _CountingScoringEngine.calls = 0
        _CountingDecisionEngine.calls = 0

    @staticmethod
    def _sample_image() -> Image.Image:
        return Image.new("RGB", (32, 32), color=(128, 128, 128))

    def test_invalid_layer_output_is_degraded_by_contract_guard(self) -> None:
        with patch("engine.pipeline.orchestrator.CnnInterface", _BadLayerMissingRaw), patch(
            "engine.pipeline.orchestrator.VitInterface", _GoodLayer
        ), patch("engine.pipeline.orchestrator.GanInterface", _GoodLayer), patch(
            "engine.pipeline.orchestrator.OcrInterface", _GoodLayer
        ), patch("engine.pipeline.orchestrator.ScoringEngine", _CountingScoringEngine), patch(
            "engine.pipeline.orchestrator.DecisionEngine", _CountingDecisionEngine
        ):
            orchestrator = VerificationOrchestrator(project_root=Path.cwd())
            result = orchestrator.run_sync(self._sample_image(), metadata={})

        self.assertEqual(result["layer_status"]["cnn"], "degraded")
        self.assertEqual(result["layer_status"]["vit"], "ok")
        self.assertIn("fallback", result["layer_outputs"]["cnn"]["raw"])
        self.assertIn("score", result["layer_outputs"]["cnn"])
        self.assertIn("raw", result["layer_outputs"]["cnn"])

    def test_all_layer_outputs_contain_score_and_raw_when_contract_is_met(self) -> None:
        with patch("engine.pipeline.orchestrator.CnnInterface", _GoodLayer), patch(
            "engine.pipeline.orchestrator.VitInterface", _GoodLayer
        ), patch("engine.pipeline.orchestrator.GanInterface", _GoodLayer), patch(
            "engine.pipeline.orchestrator.OcrInterface", _GoodLayer
        ), patch("engine.pipeline.orchestrator.ScoringEngine", _CountingScoringEngine), patch(
            "engine.pipeline.orchestrator.DecisionEngine", _CountingDecisionEngine
        ):
            orchestrator = VerificationOrchestrator(project_root=Path.cwd())
            result = orchestrator.run_sync(self._sample_image(), metadata={})

        for layer_name, payload in result["layer_outputs"].items():
            self.assertIn("score", payload, msg=f"{layer_name} must contain score")
            self.assertIn("raw", payload, msg=f"{layer_name} must contain raw")

    def test_fusion_and_decision_are_centralized_in_engine_only(self) -> None:
        with patch("engine.pipeline.orchestrator.CnnInterface", _GoodLayer), patch(
            "engine.pipeline.orchestrator.VitInterface", _GoodLayer
        ), patch("engine.pipeline.orchestrator.GanInterface", _GoodLayer), patch(
            "engine.pipeline.orchestrator.OcrInterface", _GoodLayer
        ), patch("engine.pipeline.orchestrator.ScoringEngine", _CountingScoringEngine), patch(
            "engine.pipeline.orchestrator.DecisionEngine", _CountingDecisionEngine
        ):
            orchestrator = VerificationOrchestrator(project_root=Path.cwd())
            _ = orchestrator.run_sync(self._sample_image(), metadata={})

        self.assertEqual(_CountingScoringEngine.calls, 1)
        self.assertEqual(_CountingDecisionEngine.calls, 1)

        for cls in (CnnInterface, VitInterface, GanInterface, OcrInterface, Layer4OcrScorer):
            self.assertFalse(hasattr(cls, "fuse"), msg=f"{cls.__name__} must not expose fuse")
            self.assertFalse(hasattr(cls, "classify"), msg=f"{cls.__name__} must not expose classify")


if __name__ == "__main__":
    unittest.main()
