from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Any, Dict

from ...preprocessing import preprocess_all
from .base import load_module_from_file, prepend_sys_path, to_authenticity_from_fraud_probability


class GanInterface:
    """Pluggable Layer-3 wrapper for external GAN detector repositories."""

    def __init__(self, project_root: Path, module_rel_path: str | None = None) -> None:
        self._project_root = Path(project_root)
        configured_path = module_rel_path or os.getenv("VERISIGHT_LAYER3_MODULE")
        self._module_path = self._resolve_module_path(configured_path)
        self._predict_fn = None

    def _resolve_module_path(self, configured_path: str | None) -> Path | None:
        if configured_path:
            candidate = Path(configured_path)
            if not candidate.is_absolute():
                candidate = self._project_root / candidate
            if candidate.exists():
                return candidate

        candidates = [
            self._project_root / "layer3" / "layer3_gan" / "verisight_layer3_gan.py",
            self._project_root / "GAN" / "layer3_gan" / "verisight_layer3_gan.py",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_trained_detector(self):
        trained_module = self._project_root / "layer3" / "layer3_trained_inference.py"
        if not trained_module.exists():
            return None

        with prepend_sys_path(trained_module.parent):
            module = importlib.import_module("layer3_trained_inference")

        load_fn = getattr(module, "load_trained_layer3", None)
        if not callable(load_fn):
            return None

        detector = load_fn(
            checkpoint_path=str(
                (self._project_root / "layer3" / "checkpoints" / "layer3_best.pth")
                if (self._project_root / "layer3" / "checkpoints" / "layer3_best.pth").exists()
                else (self._project_root / "checkpoints" / "layer3_best.pth")
            ),
            centroid_path=str(
                (self._project_root / "layer3" / "checkpoints" / "clip_real_centroid.pt")
                if (self._project_root / "layer3" / "checkpoints" / "clip_real_centroid.pt").exists()
                else (self._project_root / "checkpoints" / "clip_real_centroid.pt")
            ),
            device=os.getenv("VERISIGHT_LAYER3_DEVICE", "cpu"),
        )
        return detector

    @staticmethod
    def _normalize_detector_output(result: Any) -> Dict[str, Any]:
        if isinstance(result, dict):
            return result

        if hasattr(result, "fraud_probability"):
            sub_scores = getattr(result, "sub_scores", None)
            if hasattr(sub_scores, "__dict__"):
                sub_scores = dict(sub_scores.__dict__)
            elif not isinstance(sub_scores, dict):
                sub_scores = {}

            payload = {
                "fraud_probability": float(getattr(result, "fraud_probability", 0.5)),
                "sub_scores": sub_scores,
                "flags": list(getattr(result, "flags", [])),
            }
            heatmap = getattr(result, "heatmap", None)
            if heatmap is not None:
                payload["heatmap_shape"] = list(getattr(heatmap, "shape", []))
            return payload

        return {
            "fraud_probability": 0.5,
            "sub_scores": {},
            "flags": ["layer3_unrecognized_result"],
        }

    @staticmethod
    def _select_image_input(bundle: Dict[str, Any] | None, image: Any) -> Any:
        if bundle is not None:
            for key in ("bgr", "ocr_input", "rgb_array"):
                value = bundle.get(key)
                if value is not None:
                    return value

        if isinstance(image, dict):
            for key in ("bgr", "ocr_input", "rgb_array"):
                value = image.get(key)
                if value is not None:
                    return value

        return image

    def load(self) -> None:
        if self._predict_fn is not None:
            return

        # First, try to load trained detector (independent of _module_path)
        trained_detector = None
        try:
            trained_detector = self._load_trained_detector()
        except Exception:
            trained_detector = None

        if trained_detector is not None:
            def _trained_predict(image_input: Any):
                return self._normalize_detector_output(trained_detector.analyze(image_input))

            self._predict_fn = _trained_predict
            return

        # Fallback: try to load GAN module
        if self._module_path is None:
            return

        with prepend_sys_path(self._module_path.parent.parent):
            module = load_module_from_file("verisight_layer3_gan_module", self._module_path)

        for name in ("run_inference", "infer", "predict", "analyze"):
            fn = getattr(module, name, None)
            if callable(fn):
                self._predict_fn = fn
                return

        detector_cls = getattr(module, "GANDetector", None)
        if detector_cls is not None:
            detector = detector_cls()

            def _class_predict(image_input: Any):
                return self._normalize_detector_output(detector.analyze(image_input))

            self._predict_fn = _class_predict
            return

        raise AttributeError(f"No callable inference function found in Layer-3 module: {self._module_path}")

    def predict(self, image: Any, preprocessed: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.load()

        if self._predict_fn is None:
            raw = {
                "fraud_probability": 0.5,
                "classification": "UNKNOWN",
                "sub_scores": {},
                "flags": ["layer3_module_not_available"],
            }
            return {"score": 50.0, "available": False, "uncertainty": 1.0, "raw": raw}

        if preprocessed is not None:
            bundle = preprocessed
        elif isinstance(image, dict) and any(key in image for key in ("bgr", "ocr_input", "rgb_array")):
            bundle = image
        else:
            bundle = preprocess_all(image)

        image_input = self._select_image_input(bundle, image)

        raw = self._normalize_detector_output(self._predict_fn(image_input))
        fraud_probability = float(raw.get("fraud_probability", 0.5))
        uncertainty = raw.get("uncertainty")
        if not isinstance(uncertainty, (int, float)):
            uncertainty = max(0.0, min(1.0, 1.0 - abs((1.0 - fraud_probability) - 0.5) * 2.0))

        return {
            "score": to_authenticity_from_fraud_probability(fraud_probability),
            "available": True,
            "uncertainty": float(uncertainty),
            "raw": raw,
        }