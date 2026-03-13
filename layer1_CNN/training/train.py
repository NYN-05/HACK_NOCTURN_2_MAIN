from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from data.dataset import build_dataloaders
from evaluation.metrics import compute_metrics
from models.efficientnet_forensics import EfficientNetForensics
from utils.checkpointing import save_checkpoint
from utils.device import resolve_device, use_cuda
from utils.reproducibility import seed_everything


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    channels_last: bool = False,
) -> Tuple[float, Dict[str, object]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    y_true, y_pred = [], []

    context = torch.enable_grad if is_train else torch.inference_mode
    with context():
        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            if channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)

            y_true.append(labels.detach().cpu().numpy())
            y_pred.append(preds.detach().cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)

    metrics = compute_metrics(y_true_np, y_pred_np)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VeriSight Layer 1 CNN module")

    parser.add_argument("--dataset-root", required=True, help="Root path containing CASIA/CoMoFoD/CG1050")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--multiprocessing-context", default="spawn", choices=["spawn", "forkserver"])
    parser.add_argument("--num-cpu-threads", type=int, default=0, help="0 keeps PyTorch default")
    parser.add_argument("--data-parallel", action="store_true", help="Use all CUDA GPUs via DataParallel")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for single-GPU CUDA runs")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format on CUDA")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision when CUDA is used")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    cuda_enabled = use_cuda(device)
    amp_enabled = bool(args.amp and cuda_enabled)
    channels_last_enabled = bool(args.channels_last and cuda_enabled)

    if args.compile and args.data_parallel:
        print("Both --compile and --data-parallel were set. Prioritizing --data-parallel and disabling --compile.")
        args.compile = False

    if args.num_cpu_threads > 0:
        torch.set_num_threads(args.num_cpu_threads)
        torch.set_num_interop_threads(max(1, args.num_cpu_threads // 2))

    if cuda_enabled:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        pin_memory=cuda_enabled,
        prefetch_factor=args.prefetch_factor,
        multiprocessing_context=args.multiprocessing_context,
    )

    model = EfficientNetForensics(pretrained=True).to(device)
    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last)

    if args.compile and cuda_enabled and not args.data_parallel and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="max-autotune")
            # Trigger backend creation once so Triton issues are caught early.
            with torch.inference_mode():
                warmup = torch.randn(1, 6, args.image_size, args.image_size, device=device)
                if channels_last_enabled:
                    warmup = warmup.contiguous(memory_format=torch.channels_last)
                _ = model(warmup)
            print("Using torch.compile(max-autotune)")
        except Exception as exc:
            print(f"torch.compile unavailable at runtime ({exc}). Continuing without compile.")
            args.compile = False
            model = EfficientNetForensics(pretrained=True).to(device)
            if channels_last_enabled:
                model = model.to(memory_format=torch.channels_last)

    if cuda_enabled and args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            channels_last=channels_last_enabled,
        )
        val_loss, val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            amp_enabled=amp_enabled,
            channels_last=channels_last_enabled,
        )

        scheduler.step(val_metrics["f1"])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_f1": best_f1,
                    "args": vars(args),
                },
                str(output_dir / "best_model.pth"),
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
        channels_last=channels_last_enabled,
    )
    print(f"Test loss: {test_loss:.4f} | Test metrics: {test_metrics}")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
