from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from typing import Any, Dict

from ...preprocessing import preprocess_all
from .base import prepend_sys_path

LOGGER = logging.getLogger(__name__)


class _FallbackOcrDetector:
    def analyze(self, _image_path: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "score": 50.0,
            "available": False,
            "uncertainty": 1.0,
            "flags": ["ocr_weights_missing"],
            "details": {
                "ocr_engine_unavailable": True,
                "reason": "layer4_weights_missing",
                "metadata": metadata or {},
                "uncertainty": 1.0,
            },
            "fallback": "uncertain_fallback",
        }


class OcrInterface:
    def __init__(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._layer_root = self._project_root / "layer4"
        self._text_detector_model_path = self._resolve_text_detector_model_path()
        self._detector = None
        self._ocr_engine = os.getenv("VERISIGHT_OCR_ENGINE", "auto").strip().lower()
        self._use_gpu = os.getenv("VERISIGHT_OCR_GPU", "false").strip().lower() == "true"

    def _resolve_text_detector_model_path(self) -> Path | None:
        candidates = [
            self._layer_root / "best.pt",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        LOGGER.warning(
            "Layer-4 trained OCR detector weights not found at layer4/best.pt; using OCR fallback"
        )
        return None

    def load(self) -> None:
        if self._detector is not None:
            return

        if self._text_detector_model_path is None:
            LOGGER.error(f"best.pt not found at {self._layer_root / 'best.pt'}")
            self._detector = _FallbackOcrDetector()
            return

        module_path = self._layer_root / "inference" / "ocr_verification.py"
        if not module_path.exists():
            LOGGER.error(f"OCR module not found: {module_path}")
            self._detector = _FallbackOcrDetector()
            return

        try:
            with prepend_sys_path(self._layer_root):
                module = importlib.import_module("layer4.inference.ocr_verification")

            ocr_cls = getattr(module, "OCRVerificationModule", None)
            if ocr_cls is None:
                LOGGER.error("OCRVerificationModule is missing in layer4 inference module")
                self._detector = _FallbackOcrDetector()
                return

            enable_paddle = self._ocr_engine in {"auto", "paddle"}
            enable_easy = self._ocr_engine in {"auto", "easy"}
            enable_yolo = self._ocr_engine in {"auto", "easy", "paddle", "yolo"}

            LOGGER.info(f"Loading OCR model from {self._text_detector_model_path}")
            self._detector = ocr_cls(
                text_detector_model=str(self._text_detector_model_path),
                use_gpu=self._use_gpu,
                enable_yolo=enable_yolo,
                enable_easyocr=enable_easy,
                enable_paddle_fallback=enable_paddle,
                confidence_threshold=float(os.getenv("VERISIGHT_OCR_CONF_THRESHOLD", "0.25")),
            )
            LOGGER.info("OCR detector loaded successfully")
        except Exception as e:
            LOGGER.error(f"Failed to load OCR detector: {e}", exc_info=True)
            self._detector = _FallbackOcrDetector()

        LOGGER.info(
            "ocr_interface_loaded engine=%s gpu=%s yolo=%s easyocr=%s paddle=%s",
            self._ocr_engine,
            self._use_gpu,
            enable_yolo,
            enable_easy,
            enable_paddle,
        )

    def _resolve_image_input(self, image: Any, preprocessed: Dict[str, Any] | None = None):
        if preprocessed is not None:
            for key in ("ocr_input", "bgr", "rgb_array"):
                value = preprocessed.get(key)
                if value is not None:
                    return value
        if isinstance(image, dict):
            for key in ("ocr_input", "bgr", "rgb_array"):
                value = image.get(key)
                if value is not None:
                    return value
        return image

    def predict(self, image: Any, metadata: Dict[str, Any] | None = None, preprocessed: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.load()

        if isinstance(self._detector, _FallbackOcrDetector):
            raw = self._detector.analyze("", metadata=metadata or {})
            return {
                "score": float(raw.get("score", 50.0)),
                "available": bool(raw.get("available", False)),
                "uncertainty": float(raw.get("uncertainty", 1.0)),
                "raw": raw,
            }

        if preprocessed is not None:
            bundle = preprocessed
        elif isinstance(image, dict) and any(key in image for key in ("ocr_input", "bgr", "rgb_array")):
            bundle = image
        else:
            bundle = preprocess_all(image)

        image_input = self._resolve_image_input(bundle, bundle)

        raw = self._detector.analyze(image_input, metadata=metadata or {})

        return {
            "score": float(raw.get("score", 50.0)),
            "available": bool(raw.get("available", True)),
            "uncertainty": float(raw.get("uncertainty", raw.get("details", {}).get("uncertainty", 0.0))),
            "raw": raw,
        }