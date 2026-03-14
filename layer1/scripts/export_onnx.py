from __future__ import annotations

import argparse

import torch

from models.efficientnet_forensics import EfficientNetForensics
from utils.checkpointing import load_checkpoint
from utils.warnings_control import suppress_noisy_warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export VeriSight Layer 1 model to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="artifacts/verisight_layer1.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dynamic-batch", action="store_true", help="Export with dynamic batch axis")
    parser.add_argument("--num-cpu-threads", type=int, default=0, help="0 keeps PyTorch default")
    return parser.parse_args()


def main() -> None:
    suppress_noisy_warnings()
    args = parse_args()

    if args.num_cpu_threads > 0:
        torch.set_num_threads(args.num_cpu_threads)
        torch.set_num_interop_threads(max(1, args.num_cpu_threads // 2))

    model = EfficientNetForensics(pretrained=False)

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 6, args.image_size, args.image_size)
    dynamic_axes = {"input": {0: "batch"}, "logits": {0: "batch"}} if args.dynamic_batch else None

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            args.output,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )
    print(f"ONNX model exported to: {args.output}")


if __name__ == "__main__":
    main()
