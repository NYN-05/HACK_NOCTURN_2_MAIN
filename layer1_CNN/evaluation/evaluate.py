from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics
from models.efficientnet_forensics import EfficientNetForensics
from utils.checkpointing import load_checkpoint
from utils.device import resolve_device, use_cuda


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VeriSight Layer 1 model")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--multiprocessing-context", default="spawn", choices=["spawn", "forkserver"])
    parser.add_argument("--num-cpu-threads", type=int, default=0, help="0 keeps PyTorch default")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on CUDA")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format on CUDA")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    cuda_enabled = use_cuda(device)
    channels_last_enabled = bool(args.channels_last and cuda_enabled)

    if args.num_cpu_threads > 0:
        torch.set_num_threads(args.num_cpu_threads)
        torch.set_num_interop_threads(max(1, args.num_cpu_threads // 2))

    if cuda_enabled:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    _, _, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=cuda_enabled,
        prefetch_factor=args.prefetch_factor,
        multiprocessing_context=args.multiprocessing_context,
    )

    model = EfficientNetForensics(pretrained=False).to(device)
    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last)

    if args.compile and cuda_enabled and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            with torch.inference_mode():
                warmup = torch.randn(1, 6, args.image_size, args.image_size, device=device)
                if channels_last_enabled:
                    warmup = warmup.contiguous(memory_format=torch.channels_last)
                _ = model(warmup)
        except Exception as exc:
            print(f"torch.compile unavailable at runtime ({exc}). Continuing without compile.")

    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    all_true, all_pred = [], []

    with torch.inference_mode():
        for batch in tqdm(test_loader):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if channels_last_enabled:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type=device.type, enabled=cuda_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(logits, dim=1)
            all_true.append(labels.cpu().numpy())
            all_pred.append(preds.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)

    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / len(test_loader.dataset)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
