from __future__ import annotations

from typing import Any, Dict
from pathlib import Path

import numpy as np

from .common import prepend_sys_path, to_authenticity_from_fraud_probability
from engine.preprocessing.shared_pipeline import preprocess_all


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
    def _softmax(logits):
        import numpy as np  # type: ignore

        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def _resolve_bundle(self, image: Any, preprocessed: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if preprocessed is not None:
            return preprocessed
        if isinstance(image, dict) and "normalized" in image:
            return image
        return preprocess_all(image)

    def predict(self, image: Any, preprocessed: Dict[str, Any] | None = None) -> Dict[str, Any]:
        bundle = self._resolve_bundle(image, preprocessed)
        self.load()

        if self._backend == "torch":
            result = self._engine.predict_from_preprocessed(bundle)
            fraud_probability = float(result.get("forgery_probability", 0.5))
        else:
            tensor = bundle.get("cnn_input_np")
            if tensor is None:
                normalized = bundle.get("normalized")
                if hasattr(normalized, "detach"):
                    tensor = normalized.detach().cpu().numpy().astype(np.float32)
                else:
                    tensor = np.asarray(normalized, dtype=np.float32)

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
            "available": True,
            "uncertainty": round(max(0.0, min(1.0, 1.0 - abs((1.0 - fraud_probability) - 0.5) * 2.0)), 4),
        }
