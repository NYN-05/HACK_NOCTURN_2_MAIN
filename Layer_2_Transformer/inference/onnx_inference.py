from pathlib import Path

import numpy as np
import onnxruntime as ort

from inference.preprocessing import preprocess_image
from utils.config import ID_TO_LABEL, ONNX_MODEL_PATH


class ONNXDetector:
    def __init__(self, onnx_path: str | Path = ONNX_MODEL_PATH):
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path.as_posix(), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, image_path: str | Path) -> dict:
        pixel_values = preprocess_image(image_path)
        logits = self.session.run(None, {self.input_name: pixel_values})[0]

        probs = _softmax(logits[0])
        fake_prob = float(probs[1])

        return {
            "vit_score": int(round(fake_prob * 100)),
            "label": ID_TO_LABEL[1] if fake_prob >= 0.5 else ID_TO_LABEL[0],
        }


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def detect_ai_generated(image_path: str | Path) -> dict:
    detector = ONNXDetector()
    return detector.predict(image_path)
