from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

from models.efficientnet_forensics import EfficientNetForensics
from utils.checkpointing import load_checkpoint
from utils.device import resolve_device, use_cuda
from utils.warnings_control import suppress_noisy_warnings

try:
    from engine.preprocessing.shared_pipeline import preprocess_all
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    from engine.preprocessing.shared_pipeline import preprocess_all


class ForensicsInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        image_size: int = 224,
        compile_model: bool = False,
        channels_last: bool = False,
    ) -> None:
        self.device = resolve_device(device)
        self.cuda_enabled = use_cuda(self.device)
        self.channels_last = bool(channels_last and self.cuda_enabled)
        self.image_size = image_size

        if self.cuda_enabled:
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.model = EfficientNetForensics(pretrained=False)
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(self.device))

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        if compile_model:
            print("--compile is currently disabled in this project for stability.")
        self.model.eval()

    def preprocess(self, image: Any) -> torch.Tensor:
        bundle = preprocess_all(image, image_size=self.image_size)
        return bundle["normalized"].to(self.device, non_blocking=True)

    def predict_from_preprocessed(self, preprocessed: Dict[str, Any]) -> Dict[str, object]:
        tensor = preprocessed.get("normalized")
        if tensor is None:
            raise KeyError("preprocessed bundle must contain 'normalized'")

        tensor = tensor.to(self.device, non_blocking=True)
        if self.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.cuda_enabled):
                logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)

        manip_prob = float(probs[0, 1].item())
        pred = "manipulated" if manip_prob >= 0.5 else "authentic"

        return {
            "cnn_score": round(manip_prob * 100.0, 2),
            "forgery_probability": manip_prob,
            "prediction": pred,
            "uncertainty": round(max(0.0, min(1.0, 1.0 - abs((1.0 - manip_prob) - 0.5) * 2.0)), 4),
            "backend": "torch",
            "available": True,
        }

    def _predict_from_tensor(self, tensor: torch.Tensor) -> Dict[str, object]:
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor with shape [B, C, H, W], got {tuple(tensor.shape)}")

        tensor = tensor.to(self.device, non_blocking=True)
        if self.channels_last:
            tensor = tensor.contiguous(memory_format=torch.channels_last)

        with torch.inference_mode():
            with torch.amp.autocast(device_type=self.device.type, enabled=self.cuda_enabled):
                logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)

        manip_prob = float(probs[0, 1].item())
        pred = "manipulated" if manip_prob >= 0.5 else "authentic"

        return {
            "cnn_score": round(manip_prob * 100.0, 2),
            "forgery_probability": manip_prob,
            "prediction": pred,
            "uncertainty": round(max(0.0, min(1.0, 1.0 - abs((1.0 - manip_prob) - 0.5) * 2.0)), 4),
            "backend": "torch",
            "available": True,
        }

    @torch.inference_mode()
    def predict(self, image_path: str) -> Dict[str, object]:
        tensor = self.preprocess(image_path)
        return self._predict_from_tensor(tensor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VeriSight Layer 1 inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on CUDA")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format on CUDA")
    return parser.parse_args()


def main() -> None:
    suppress_noisy_warnings()
    args = parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    engine = ForensicsInferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
        compile_model=False,
        channels_last=args.channels_last,
    )
    result = engine.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
