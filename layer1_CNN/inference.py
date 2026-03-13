from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

from models.efficientnet_forensics import EfficientNetForensics
from preprocessing.ela import ELAGenerator
from utils.checkpointing import load_checkpoint
from utils.device import resolve_device, use_cuda


class ForensicsInferenceEngine:
    def __init__(self, checkpoint_path: str, device: str = "cpu", image_size: int = 224) -> None:
        self.device = resolve_device(device)
        self.cuda_enabled = use_cuda(self.device)
        self.image_size = image_size
        self.ela_generator = ELAGenerator(jpeg_quality=90, ela_scale=10.0)

        self.model = EfficientNetForensics(pretrained=False)
        checkpoint = load_checkpoint(checkpoint_path, map_location=str(self.device))

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()
        self.rgb_normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.ela_normalize = transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

    def preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image = self.resize(image)
        ela = self.ela_generator.generate(image)

        rgb = self.rgb_normalize(self.to_tensor(image))
        ela_tensor = self.ela_normalize(self.to_tensor(ela))
        fusion = torch.cat((rgb, ela_tensor), dim=0).unsqueeze(0)
        return fusion.to(self.device, non_blocking=True)

    @torch.inference_mode()
    def predict(self, image_path: str) -> Dict[str, object]:
        tensor = self.preprocess(image_path)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.cuda_enabled):
            logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)

        manip_prob = float(probs[0, 1].item())
        pred = "manipulated" if manip_prob >= 0.5 else "authentic"

        return {
            "cnn_score": round(manip_prob * 100.0, 2),
            "forgery_probability": manip_prob,
            "prediction": pred,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VeriSight Layer 1 inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    engine = ForensicsInferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        image_size=args.image_size,
    )
    result = engine.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
