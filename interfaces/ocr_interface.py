from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import prepend_sys_path


class OcrInterface:
    def __init__(self, project_root: Path) -> None:
        self._project_root = Path(project_root)
        self._layer_root = self._project_root / "layer4"
        self._text_detector_model_path = self._resolve_text_detector_model_path()
        self._detector = None

    def _resolve_text_detector_model_path(self) -> Path:
        candidates = [
            self._layer_root / "models" / "yolo_finetune" / "layer4_expiry_region" / "weights" / "best.pt",
            self._layer_root / "models" / "yolo_finetune" / "layer4_expiry_region_y11s_v1" / "weights" / "best.pt",
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Layer-4 trained OCR detector weights not found under layer4/models/yolo_finetune/**/weights/best.pt"
        )

    def load(self) -> None:
        if self._detector is not None:
            return

        with prepend_sys_path(self._layer_root):
            from inference.ocr_verification import OCRVerificationModule  # type: ignore

            self._detector = OCRVerificationModule(
                text_detector_model=str(self._text_detector_model_path)
            )

    def predict(self, image_path: str | Path, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self.load()
        raw = self._detector.analyze(str(image_path), metadata=metadata or {})

        return {
            "score": float(raw.get("score", 50.0)),
            "raw": raw,
        }
