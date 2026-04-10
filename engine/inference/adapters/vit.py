from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ...preprocessing import preprocess_all
from .base import to_authenticity_from_fraud_probability


class VitInterface:
    def __init__(
        self,
        project_root: Path,
        onnx_model: str = "layer2/models/vit_layer2_detector.onnx",
    ) -> None:
        self._project_root = Path(project_root)
        self._onnx_path = self._project_root / onnx_model
        self._session = None
        self._input_name = None

    def load(self) -> None:
        if self._session is not None:
            return

        try:
            import onnxruntime as ort  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Layer-2 dependency missing: onnxruntime. Install with '.\\.venv\\Scripts\\python.exe -m pip install onnxruntime'."
            ) from exc

        if not self._onnx_path.exists():
            raise FileNotFoundError(f"Layer-2 ONNX model not found: {self._onnx_path}")

        providers = [provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"] if provider in ort.get_available_providers()]
        self._session = ort.InferenceSession(self._onnx_path.as_posix(), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    @staticmethod
    def _softmax(logits):
        try:
            import numpy as np  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Layer-2 dependency missing: numpy. Install with '.\\.venv\\Scripts\\python.exe -m pip install numpy'."
            ) from exc

        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def predict_from_preprocessed(self, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        self.load()
        tensor = preprocessed.get("clip_input")
        if tensor is None:
            raise KeyError("preprocessed bundle must contain 'clip_input'")

        tensor = np.asarray(tensor, dtype=np.float32)
        logits = self._session.run(None, {self._input_name: tensor})[0]
        probs = self._softmax(logits[0])

        fraud_probability = float(probs[1])
        raw = {
            "vit_score": int(round(fraud_probability * 100.0)),
            "label": "AI_GENERATED" if fraud_probability >= 0.5 else "REAL",
            "backend": "onnx",
        }

        return {
            "score": to_authenticity_from_fraud_probability(fraud_probability),
            "raw": raw,
            "available": True,
            "uncertainty": round(max(0.0, min(1.0, 1.0 - abs((1.0 - fraud_probability) - 0.5) * 2.0)), 4),
        }

    def predict(self, image: Any, preprocessed: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bundle = preprocessed or (image if isinstance(image, dict) and "clip_input" in image else None)
        if bundle is None:
            bundle = preprocess_all(image)
        return self.predict_from_preprocessed(bundle)