from __future__ import annotations

import argparse

import torch

from models.efficientnet_forensics import EfficientNetForensics
from utils.checkpointing import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VeriSight Layer 1 model to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/verisight_layer1.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = EfficientNetForensics(pretrained=False)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 6, args.image_size, args.image_size)

    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported to: {args.output}")


if __name__ == "__main__":
    main()
