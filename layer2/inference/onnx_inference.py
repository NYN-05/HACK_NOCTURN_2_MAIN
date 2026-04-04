import ctypes
from pathlib import Path

import numpy as np
import onnxruntime as ort

from engine.preprocessing.shared_pipeline import preprocess_all
from utils.config import ID_TO_LABEL, ONNX_MODEL_PATH


class ONNXDetector:
    def __init__(self, onnx_path: str | Path = ONNX_MODEL_PATH):
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        available_providers = ort.get_available_providers()
        providers = []

        if "CUDAExecutionProvider" in available_providers and _cuda_runtime_ready():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        providers = [provider for provider in providers if provider in available_providers]
        if not providers:
            raise RuntimeError(f"No compatible ONNX Runtime provider found. Available: {available_providers}")

        self.session = ort.InferenceSession(self.onnx_path.as_posix(), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict_from_pixel_values(self, pixel_values: np.ndarray) -> dict:
        logits = self.session.run(None, {self.input_name: pixel_values})[0]

        probs = _softmax(logits[0])
        fake_prob = float(probs[1])

        return {
            "vit_score": int(round(fake_prob * 100)),
            "label": ID_TO_LABEL[1] if fake_prob >= 0.5 else ID_TO_LABEL[0],
        }

    def predict_from_preprocessed(self, preprocessed: dict) -> dict:
        pixel_values = preprocessed.get("clip_input")
        if pixel_values is None:
            raise KeyError("preprocessed bundle must contain 'clip_input'")
        return self.predict_from_pixel_values(np.asarray(pixel_values, dtype=np.float32))

    def predict(self, image_path: str | Path) -> dict:
        pixel_values = preprocess_all(image_path)["clip_input"]
        return self.predict_from_pixel_values(pixel_values)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def detect_ai_generated(image_path: str | Path) -> dict:
    detector = ONNXDetector()
    return detector.predict(image_path)


def _cuda_runtime_ready() -> bool:
    """Return True when the required CUDA runtime dependencies are available."""
    try:
        ctypes.CDLL("cudnn64_9.dll")
    except OSError:
        return False
    return True
