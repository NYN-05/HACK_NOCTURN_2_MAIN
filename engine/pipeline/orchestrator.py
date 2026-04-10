from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict

from ..configs.weights import (
    EARLY_EXIT_CNN_SCORE_THRESHOLD,
    EARLY_EXIT_MIN_RELIABILITY,
    ENABLE_EARLY_EXIT,
    LAYER_TIMEOUT_MS,
)
from engine.core.config import VeriSightConfig
from engine import DecisionEngine, ScoringEngine
from engine.interfaces import CnnInterface, GanInterface, OcrInterface, VitInterface
from engine.preprocessing.shared_pipeline import preprocess_all

LOGGER = logging.getLogger(__name__)


class VerificationOrchestrator:
    """Central inference pipeline: model execution, fusion, and final decision."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        # Default to repository root so layer paths resolve to <repo>/layer*.
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2])

        self.cnn = CnnInterface(self.project_root)
        self.vit = VitInterface(self.project_root)
        self.gan = GanInterface(self.project_root)
        self.ocr = OcrInterface(self.project_root)
        self._layer_scorers: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]] = {
            "cnn": lambda bundle, _metadata: self.cnn.predict(bundle, preprocessed=bundle),
            "vit": lambda bundle, _metadata: self.vit.predict(bundle, preprocessed=bundle),
            "gan": lambda bundle, _metadata: self.gan.predict(bundle, preprocessed=bundle),
            "ocr": lambda bundle, metadata: self.ocr.predict(bundle, metadata=metadata, preprocessed=bundle),
        }

        self.scoring = ScoringEngine()
        self.decision = DecisionEngine()
        self.early_exit_enabled = bool(ENABLE_EARLY_EXIT)
        self.early_exit_score_threshold = float(EARLY_EXIT_CNN_SCORE_THRESHOLD)
        self.early_exit_min_reliability = float(EARLY_EXIT_MIN_RELIABILITY)

    def load_models(self) -> None:
        for key, loader in (
            ("cnn", self.cnn.load),
            ("vit", self.vit.load),
            ("gan", self.gan.load),
            ("ocr", self.ocr.load),
        ):
            try:
                loader()
            except (FileNotFoundError, ModuleNotFoundError) as exc:
                LOGGER.warning("%s model unavailable at startup: %s", key, exc)
            except Exception:
                LOGGER.exception("%s model failed to load at startup", key)

    @staticmethod
    def _compute_reliability(output: Dict[str, Any], layer_name: str = "cnn") -> float:
        """Compute reliability with weight-based scaling for UI display.
        
        Higher weight models display higher reliability, lower weight models display lower reliability.
        This reflects the importance of each model in the final decision.
        """
        if not isinstance(output, dict):
            return 0.0

        if output.get("available") is False:
            return 0.0

        score = 1.0
        raw = output.get("raw", {})
        if not isinstance(raw, dict):
            raw = {}

        if raw.get("available") is False:
            return 0.0

        uncertainty = output.get("uncertainty", raw.get("uncertainty"))
        if isinstance(uncertainty, (int, float)):
            score = max(0.0, 1.0 - max(0.0, min(1.0, float(uncertainty))))

        fallback_flag = raw.get("fallback")
        if fallback_flag is not None:
            score -= 0.45

        flags = raw.get("flags", [])
        if isinstance(flags, list) and any("fail" in str(flag).lower() for flag in flags):
            score -= 0.25

        details = raw.get("details", {})
        if isinstance(details, dict) and details.get("ocr_engine_unavailable"):
            score -= 0.2

        # Base reliability before weight scaling
        base_reliability = max(0.15, min(1.0, score))
        
        # Apply weight-based scaling for UI display
        # This makes high-weight models appear more reliable and low-weight models appear less reliable
        layer_weights = VeriSightConfig.LAYER_WEIGHTS
        layer_weight = layer_weights.get(layer_name, 0.25)
        
        # Normalize weights to 0-1 range (CNN=0.45 -> scale factor ~1.5, OCR=0.125 -> scale factor ~0.4)
        weight_scale = layer_weight / 0.3  # 0.3 is a reference point between max (0.45) and min (0.125)
        weight_scale = max(0.35, min(1.5, weight_scale))  # Clamp to reasonable range
        
        # Apply weight scaling while maintaining reliability floor
        weighted_reliability = base_reliability * weight_scale
        
        return max(0.15, min(1.0, weighted_reliability))

    @staticmethod
    def _validate_layer_output(layer_name: str, output: Any) -> Dict[str, Any]:
        if not isinstance(output, dict):
            raise TypeError(f"{layer_name} output must be a dict")

        missing = [key for key in ("score", "raw") if key not in output]
        if missing:
            raise KeyError(f"{layer_name} output missing required keys: {', '.join(missing)}")

        try:
            float(output["score"])
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{layer_name} output score must be numeric") from exc

        if not isinstance(output["raw"], dict):
            raise TypeError(f"{layer_name} output raw must be a dict")

        return output

    @staticmethod
    def _skipped_layer_output(layer_name: str) -> Dict[str, Any]:
        return {
            "score": 50.0,
            "raw": {
                "fallback": "early_exit_skipped",
                "flags": ["layer_skipped"],
                "layer": layer_name,
            },
            "available": False,
            "uncertainty": 1.0,
            "processing_time_ms": 0,
        }

    @staticmethod
    def _build_fallback_output(layer_name: str, exc: Exception, elapsed_ms: int) -> Dict[str, Any]:
        return {
            "score": 50.0,
            "raw": {
                "error": str(exc),
                "fallback": "uncertain_fallback",
                "flags": ["layer_failed"],
                "layer": layer_name,
            },
            "available": False,
            "uncertainty": 1.0,
            "processing_time_ms": elapsed_ms,
        }

    async def _prepare_inputs(self, image: Any, preprocessed: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if preprocessed is not None:
            return preprocessed
        if isinstance(image, dict) and "normalized" in image:
            return image

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, preprocess_all, image)

    async def _run_layer_async(
        self,
        key: str,
        preprocessed: Dict[str, Any],
        metadata: Dict[str, Any],
        start_event: asyncio.Event | None = None,
    ) -> Dict[str, Any]:
        if key != "cnn" and start_event is not None:
            await start_event.wait()

        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        timeout_s = max(0.1, LAYER_TIMEOUT_MS.get(key, 3500) / 1000.0)
        predict = self._layer_scorers[key]

        try:
            output = await asyncio.wait_for(
                loop.run_in_executor(None, predict, preprocessed, metadata),
                timeout=timeout_s,
            )
            output = self._validate_layer_output(key, output)
            output["processing_time_ms"] = int((time.perf_counter() - started) * 1000)
            return output
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("%s model failed", key)
            return self._build_fallback_output(key, exc, int((time.perf_counter() - started) * 1000))

    def _should_early_exit(self, cnn_output: Dict[str, Any], cnn_reliability: float) -> bool:
        if not self.early_exit_enabled:
            return False

        cnn_score = float(cnn_output.get("score", 50.0))
        return cnn_score >= self.early_exit_score_threshold and cnn_reliability >= self.early_exit_min_reliability

    async def run(
        self,
        image: Any,
        metadata: Dict[str, Any] | None = None,
        preprocessed: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        started = time.perf_counter()
        metadata = metadata or {}
        preprocessed_bundle = await self._prepare_inputs(image, preprocessed=preprocessed)
        layer_start_event = asyncio.Event()

        layer_tasks = {
            key: asyncio.create_task(self._run_layer_async(key, preprocessed_bundle, metadata, layer_start_event))
            for key in self._layer_scorers
        }

        layer_outputs: Dict[str, Dict[str, Any]] = {}
        layer_scores: Dict[str, float] = {}
        layer_reliabilities: Dict[str, float] = {}
        layer_status: Dict[str, str] = {}
        layer_availability: Dict[str, bool] = {}

        cnn_output = await layer_tasks["cnn"]
        layer_outputs["cnn"] = cnn_output
        layer_scores["cnn"] = float(cnn_output.get("score", 50.0))
        layer_reliabilities["cnn"] = self._compute_reliability(cnn_output, "cnn")
        layer_availability["cnn"] = bool(cnn_output.get("available", True))
        layer_status["cnn"] = "ok" if layer_availability["cnn"] else "degraded"

        early_exit_triggered = self._should_early_exit(cnn_output, layer_reliabilities["cnn"])

        if early_exit_triggered:
            for key, task in layer_tasks.items():
                if key != "cnn":
                    task.cancel()

            remaining_tasks = [task for key, task in layer_tasks.items() if key != "cnn"]
            if remaining_tasks:
                await asyncio.gather(*remaining_tasks, return_exceptions=True)

            for key in ("vit", "gan", "ocr"):
                skipped = self._skipped_layer_output(key)
                layer_outputs[key] = skipped
                layer_scores[key] = float(skipped["score"])
                layer_reliabilities[key] = self._compute_reliability(skipped, key)
                layer_status[key] = "skipped"
                layer_availability[key] = False
        else:
            layer_start_event.set()
            other_results = await asyncio.gather(
                layer_tasks["vit"],
                layer_tasks["gan"],
                layer_tasks["ocr"],
                return_exceptions=True,
            )

            for key, result in zip(("vit", "gan", "ocr"), other_results):
                if isinstance(result, Exception):
                    output = self._build_fallback_output(key, result, 0)
                else:
                    output = result if isinstance(result, dict) else self._build_fallback_output(key, TypeError(f"{key} output must be a dict"), 0)

                output = self._validate_layer_output(key, output) if output.get("available", True) else output
                layer_outputs[key] = output
                layer_scores[key] = float(output.get("score", 50.0))
                layer_reliabilities[key] = self._compute_reliability(output, key)
                layer_availability[key] = bool(output.get("available", True))
                layer_status[key] = "ok" if layer_availability[key] else "degraded"

        fused = self.scoring.fuse(
            layer_scores,
            reliabilities=layer_reliabilities,
            availability=layer_availability,
        )
        decision = "ABSTAIN" if fused.abstained else self.decision.classify(fused.weighted_score)

        total_ms = int((time.perf_counter() - started) * 1000)
        response = {
            "authenticity_score": fused.weighted_score,
            "decision": decision,
            "layer_scores": fused.layer_scores,
            "layer_reliabilities": fused.layer_reliabilities,
            "effective_weights": fused.effective_weights,
            "confidence": fused.confidence,
            "available_layers": fused.available_layers,
            "abstained": fused.abstained,
            "fusion_strategy": getattr(fused, "fusion_strategy", "weighted_average"),
            "meta_model_used": getattr(fused, "meta_model_used", False),
            "early_exit_triggered": early_exit_triggered,
            "layer_status": layer_status,
            "layer_outputs": layer_outputs,
            "processing_time_ms": total_ms,
        }

        LOGGER.info(
            "verify_complete telemetry=%s",
            {
                "score": response["authenticity_score"],
                "decision": response["decision"],
                "duration_ms": total_ms,
                "confidence": response["confidence"],
                "layer_status": response["layer_status"],
                "layer_reliabilities": response["layer_reliabilities"],
            },
        )
        return response

    def run_sync(
        self,
        image: Any,
        metadata: Dict[str, Any] | None = None,
        preprocessed: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return asyncio.run(self.run(image, metadata=metadata, preprocessed=preprocessed))
