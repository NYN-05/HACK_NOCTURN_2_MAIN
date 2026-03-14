from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import to_authenticity_from_fraud_probability


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
    def _preprocess_image(image_path: str | Path, image_size: int = 224):
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Layer-2 dependencies missing: numpy and/or pillow. Install with '.\\.venv\\Scripts\\python.exe -m pip install numpy pillow'."
            ) from exc

        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size))
        array = np.asarray(image).astype("float32") / 255.0
        array = (array - 0.5) / 0.5
        array = np.transpose(array, (2, 0, 1))
        return np.expand_dims(array, axis=0)

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

    def predict(self, image_path: str | Path) -> Dict[str, Any]:
        self.load()
        tensor = self._preprocess_image(image_path)
        logits = self._session.run(None, {self._input_name: tensor})[0]
        probs = self._softmax(logits[0])

        fraud_probability = float(probs[1])
        raw = {
            "vit_score": int(round(fraud_probability * 100.0)),
            "label": "AI_GENERATED" if fraud_probability >= 0.5 else "REAL",
        }

        return {
            "score": to_authenticity_from_fraud_probability(fraud_probability),
            "raw": raw,
        }
