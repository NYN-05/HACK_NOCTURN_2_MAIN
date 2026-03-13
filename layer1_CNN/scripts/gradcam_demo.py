from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from evaluation.gradcam import GradCAM, overlay_heatmap
from models.efficientnet_forensics import EfficientNetForensics
from preprocessing.ela import ELAGenerator
from utils.checkpointing import load_checkpoint
from utils.device import resolve_device, use_cuda
from utils.warnings_control import suppress_noisy_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM manipulation heatmap")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="artifacts/gradcam_overlay.png")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    suppress_noisy_warnings()
    args = parse_args()
    device = resolve_device(args.device)
    cuda_enabled = use_cuda(device)

    model = EfficientNetForensics(pretrained=False).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = Image.open(args.image).convert("RGB").resize((args.image_size, args.image_size))
    ela = ELAGenerator().generate(image)

    to_tensor = transforms.ToTensor()
    rgb = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(to_tensor(image))
    ela_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(to_tensor(ela))
    fusion = torch.cat((rgb, ela_tensor), dim=0).unsqueeze(0).to(device, non_blocking=True)

    cam_generator = GradCAM(model)
    with torch.amp.autocast(device_type=device.type, enabled=cuda_enabled):
        cam = cam_generator.generate(fusion)

    rgb_np = np.asarray(image)
    overlay = overlay_heatmap(rgb_np, cam)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output, overlay)
    print(f"Saved Grad-CAM overlay to: {output}")


if __name__ == "__main__":
    main()
