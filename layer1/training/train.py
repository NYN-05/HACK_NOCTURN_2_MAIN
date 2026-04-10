from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

try:
    from data.dataset import build_dataloaders
    from evaluation.metrics import compute_metrics
    from models.efficientnet_forensics import EfficientNetForensics
    from utils.checkpointing import save_checkpoint
    from utils.reproducibility import seed_everything
    from utils.warnings_control import suppress_noisy_warnings
except ModuleNotFoundError:
    from layer1.data.dataset import build_dataloaders
    from layer1.evaluation.metrics import compute_metrics
    from layer1.models.efficientnet_forensics import EfficientNetForensics
    from layer1.utils.checkpointing import load_checkpoint, save_checkpoint
    from layer1.utils.reproducibility import seed_everything
    from layer1.utils.warnings_control import suppress_noisy_warnings


def _default_dataset_root() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    for folder_name in ("DATA", "Data", "data"):
        candidate = repo_root / folder_name
        if candidate.exists():
            return str(candidate)
    return str(repo_root / "DATA")


def _resolve_num_workers(requested_workers: int) -> int:
    requested_workers = max(0, requested_workers)
    if platform.system().lower() != "windows":
        return requested_workers
    if requested_workers == 0:
        requested_workers = 2
    return min(max(requested_workers, 2), 4)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    channels_last: bool = False,
    grad_clip: float = 0.0,
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
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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


def compute_class_weights(train_loader: torch.utils.data.DataLoader, device: torch.device) -> torch.Tensor:
    labels = [sample.label for sample in train_loader.dataset.samples]
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = np.sum(counts) / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train VeriSight Layer 1 CNN module")

    parser.add_argument(
        "--dataset-root",
        default=_default_dataset_root(),
        help="Root path containing CASIA/CoMoFoD/CG1050",
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Minimum F1 improvement to reset patience")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm (0 disables)")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="CrossEntropy label smoothing")
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=3,
        help="Freeze the EfficientNet feature extractor for the first N epochs",
    )
    parser.add_argument(
        "--balanced-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use weighted random sampling for balanced class batches",
    )
    parser.add_argument(
        "--class-weighted-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use class-weighted CrossEntropy from train split distribution",
    )
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW on CUDA when the local PyTorch build supports it",
    )
    parser.add_argument(
        "--ela-cache-size",
        type=int,
        default=1024,
        help="Per-worker cache size for generated ELA maps",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision when CUDA is used")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest Layer 1 checkpoint if available")

    args = parser.parse_args()
    if args.epochs > 100:
        parser.error("--epochs cannot be greater than 100")
    return args


def main() -> None:
    suppress_noisy_warnings()
    args = parse_args()
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for layer1 training, but no CUDA device is available.")

    if str(args.device).lower() in {"cpu", "mps", "xpu"}:
        raise ValueError("--device must be CUDA for training.")

    device = torch.device("cuda")
    cuda_enabled = True
    amp_enabled = bool(args.amp)
    channels_last_enabled = bool(args.channels_last)

    requested_workers = args.num_workers
    args.num_workers = _resolve_num_workers(args.num_workers)
    if args.num_workers != requested_workers:
        print(f"Using {args.num_workers} DataLoader workers (requested {requested_workers}).")

    if platform.system().lower() == "windows" and args.multiprocessing_context != "spawn":
        print("Forcing multiprocessing_context=spawn on Windows.")
        args.multiprocessing_context = "spawn"

    if args.compile:
        print("--compile is currently disabled in this project for stability.")
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
        balanced_sampling=args.balanced_sampling,
        ela_cache_size=args.ela_cache_size,
    )

    print(
        "[DATA] Layer1 samples "
        f"train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    model = EfficientNetForensics(pretrained=True).to(device)
    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last)

    if cuda_enabled and args.data_parallel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using DataParallel across {torch.cuda.device_count()} GPUs")

    model_ref = _unwrap_model(model)
    latest_checkpoint = output_dir / "latest_model.pth"
    start_epoch = 1
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    if args.resume and latest_checkpoint.exists():
        checkpoint = load_checkpoint(str(latest_checkpoint), map_location=device)
        model_ref.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_f1 = float(checkpoint.get("best_f1", best_f1))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", epochs_without_improvement))
        loaded_history = checkpoint.get("history", [])
        if isinstance(loaded_history, list):
            history.extend(loaded_history)
        print(f"Resumed Layer 1 training from {latest_checkpoint} at epoch {start_epoch}")

    if args.freeze_backbone_epochs > 0:
        if start_epoch <= args.freeze_backbone_epochs:
            model_ref.freeze_backbone()
            print(f"Freezing backbone features for the first {args.freeze_backbone_epochs} epochs.")
        else:
            model_ref.unfreeze_backbone()

    class_weights = compute_class_weights(train_loader, device) if args.class_weighted_loss else None
    if class_weights is not None:
        print(f"Using class-weighted loss weights={class_weights.detach().cpu().tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=max(0.0, float(args.label_smoothing)),
    )
    optimizer_kwargs = {
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    if cuda_enabled and args.fused_adamw:
        try:
            optimizer = AdamW(model.parameters(), fused=True, **optimizer_kwargs)
            print("Using fused AdamW optimizer.")
        except TypeError:
            optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    else:
        optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)

    if args.resume and latest_checkpoint.exists():
        checkpoint = load_checkpoint(str(latest_checkpoint), map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", scheduler.state_dict()))
        if "scaler_state_dict" in checkpoint and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    for epoch in range(start_epoch, args.epochs + 1):
        if args.freeze_backbone_epochs > 0 and epoch == args.freeze_backbone_epochs + 1:
            model_ref.unfreeze_backbone()
            print("Unfroze backbone for full fine-tuning.")

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            channels_last=channels_last_enabled,
            grad_clip=max(0.0, float(args.grad_clip)),
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
            grad_clip=0.0,
        )

        scheduler.step(float(val_metrics["f1"]))

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        latest_state = {
            "epoch": epoch,
            "model_state_dict": model_ref.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "best_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
        }
        save_checkpoint(latest_state, str(latest_checkpoint))

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        if float(val_metrics["f1"]) > (best_f1 + float(args.min_delta)):
            best_f1 = float(val_metrics["f1"])
            epochs_without_improvement = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model_ref.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                    "best_val_f1": best_f1,
                    "args": vars(args),
                    "history": history,
                },
                str(output_dir / "best_model.pth"),
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    checkpoint = load_checkpoint(str(output_dir / "best_model.pth"), map_location=device)
    model_ref.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        scaler=None,
        amp_enabled=amp_enabled,
        channels_last=channels_last_enabled,
        grad_clip=0.0,
    )
    print(f"Test loss: {test_loss:.4f} | Test metrics: {test_metrics}")

    with open(output_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    import torch.multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    main()
