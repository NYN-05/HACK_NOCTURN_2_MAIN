from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import prepend_sys_path, to_authenticity_from_fraud_probability


class CnnInterface:
    def __init__(
        self,
        project_root: Path,
        checkpoint: str = "layer1/artifacts/best_model.pth",
        onnx_model: str = "layer1/artifacts/verisight_layer1.onnx",
        device: str = "cpu",
    ) -> None:
        self._project_root = Path(project_root)
        self._layer_root = self._project_root / "layer1"
        self._checkpoint = str(self._project_root / checkpoint)
        self._onnx_model = self._project_root / onnx_model
        self._device = device
        self._engine = None
        self._onnx_session = None
        self._onnx_input_name = None
        self._backend = None

    def load(self) -> None:
        if self._engine is not None or self._onnx_session is not None:
            return

        checkpoint_path = Path(self._checkpoint)
        if checkpoint_path.exists():
            try:
                with prepend_sys_path(self._layer_root):
                    from inference import ForensicsInferenceEngine  # type: ignore

                    self._engine = ForensicsInferenceEngine(
                        checkpoint_path=str(checkpoint_path),
                        device=self._device,
                        compile_model=False,
                        channels_last=False,
                    )
                self._backend = "torch"
                return
            except ModuleNotFoundError:
                # Fall back to ONNX path when torch/torchvision are unavailable.
                pass

        if not self._onnx_model.exists():
            raise FileNotFoundError(
                f"Layer-1 model not available. Missing both checkpoint ({checkpoint_path}) and ONNX ({self._onnx_model})."
            )

        try:
            import onnxruntime as ort  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Layer-1 ONNX fallback requires onnxruntime. Install with '.\\.venv\\Scripts\\python.exe -m pip install onnxruntime'."
            ) from exc

        providers = [
            provider
            for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider in ort.get_available_providers()
        ]
        self._onnx_session = ort.InferenceSession(self._onnx_model.as_posix(), providers=providers)
        self._onnx_input_name = self._onnx_session.get_inputs()[0].name
        self._backend = "onnx"

    @staticmethod
    def _onnx_preprocess(image_path: str | Path, image_size: int = 224):
        try:
            import numpy as np  # type: ignore
            from PIL import Image  # type: ignore
            from layer1.preprocessing.ela import ELAGenerator  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Layer-1 ONNX preprocessing requires numpy and pillow. Install with '.\\.venv\\Scripts\\python.exe -m pip install numpy pillow'."
            ) from exc

        image = Image.open(image_path).convert("RGB")
        image = image.resize((image_size, image_size))

        ela_generator = ELAGenerator(jpeg_quality=90, ela_scale=10.0)
        ela = ela_generator.generate(image)

        rgb = np.asarray(image).astype("float32") / 255.0
        ela_rgb = np.asarray(ela).astype("float32") / 255.0

        rgb = (rgb - np.array([0.485, 0.456, 0.406], dtype="float32")) / np.array([0.229, 0.224, 0.225], dtype="float32")
        ela_rgb = (ela_rgb - 0.5) / 0.5

        fused = np.concatenate([rgb, ela_rgb], axis=2)
        fused = np.transpose(fused, (2, 0, 1))
        return np.expand_dims(fused, axis=0).astype("float32")

    @staticmethod
    def _softmax(logits):
        import numpy as np  # type: ignore

        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def predict(self, image_path: str | Path) -> Dict[str, Any]:
        self.load()

        if self._backend == "torch":
            result = self._engine.predict(str(image_path))
            fraud_probability = float(result.get("forgery_probability", 0.5))
        else:
            tensor = self._onnx_preprocess(image_path)
            logits = self._onnx_session.run(None, {self._onnx_input_name: tensor})[0]
            probs = self._softmax(logits[0])
            fraud_probability = float(probs[1])
            result = {
                "cnn_score": round(fraud_probability * 100.0, 2),
                "forgery_probability": fraud_probability,
                "prediction": "manipulated" if fraud_probability >= 0.5 else "authentic",
                "backend": "onnx",
            }

        return {
            "score": to_authenticity_from_fraud_probability(fraud_probability),
            "raw": result,
        }
