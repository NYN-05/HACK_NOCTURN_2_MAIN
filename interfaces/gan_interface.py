from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from .common import load_module_from_file, prepend_sys_path, to_authenticity_from_fraud_probability


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

        default_candidate = self._project_root / "GAN" / "layer3_gan" / "verisight_layer3_gan.py"
        return default_candidate if default_candidate.exists() else None

    def load(self) -> None:
        if self._predict_fn is not None:
            return
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

            def _class_predict(image_path: str):
                result = detector.analyze(image_path)
                # Normalize dataclass/class result to dict expected by orchestrator.
                if hasattr(result, "fraud_probability"):
                    return {
                        "fraud_probability": float(getattr(result, "fraud_probability", 0.5)),
                        "sub_scores": getattr(getattr(result, "sub_scores", None), "__dict__", {}),
                        "flags": list(getattr(result, "flags", [])),
                    }
                return result

            self._predict_fn = _class_predict
            return

        raise AttributeError(
            f"No callable inference function found in Layer-3 module: {self._module_path}"
        )

    def predict(self, image_path: str | Path) -> Dict[str, Any]:
        self.load()

        if self._predict_fn is None:
            raw = {
                "fraud_probability": 0.5,
                "classification": "UNKNOWN",
                "sub_scores": {},
                "flags": ["layer3_module_not_available"],
            }
            return {"score": 50.0, "raw": raw}

        raw = self._predict_fn(str(image_path))
        fraud_probability = float(raw.get("fraud_probability", 0.5))

        return {
            "score": to_authenticity_from_fraud_probability(fraud_probability),
            "raw": raw,
        }
