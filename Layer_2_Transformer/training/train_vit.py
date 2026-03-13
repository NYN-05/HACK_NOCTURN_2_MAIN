import argparse
import logging
import time
import warnings
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTForImageClassification
from transformers.utils import logging as hf_logging

from training.dataset_loader import build_dataloaders, prepare_dataset
from utils.config import (
    BEST_MODEL_PATH,
    NUM_CLASSES,
    ONNX_MODEL_PATH,
    PRETRAINED_MODEL_NAME,
    PROCESSED_DATASET_DIR,
)
from utils.metrics import compute_accuracy, summarize_epoch

LOGGER = logging.getLogger(__name__)


def build_model() -> ViTForImageClassification:
    prev_hf_verbosity = hf_logging.get_verbosity()
    try:
        # Suppress expected classifier-shape warnings when adapting ImageNet heads to 2 classes.
        hf_logging.set_verbosity_error()
        model = ViTForImageClassification.from_pretrained(
            PRETRAINED_MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    finally:
        hf_logging.set_verbosity(prev_hf_verbosity)
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False, max_steps=0):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum = 0.0
    acc_sum = 0.0
    steps = 0

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, total=len(loader), leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(pixel_values=pixel_values)
                logits = outputs.logits
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            batch_acc = compute_accuracy(logits.detach(), labels)
            loss_sum += loss.item()
            acc_sum += batch_acc
            steps += 1

            if max_steps > 0 and steps >= max_steps:
                break

    return summarize_epoch(loss_sum, acc_sum, steps)


def save_checkpoint(model: nn.Module, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def load_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)


def export_onnx(model: nn.Module, output_path: Path, device: torch.device, image_size: int = 224):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    class OnnxWrapper(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values):
            logits = self.wrapped(pixel_values=pixel_values).logits
            return logits

    wrapper = OnnxWrapper(model).to(device)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"You are using the legacy TorchScript-based ONNX export.*",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Converting a tensor to a Python boolean might cause the trace to be incorrect.*",
        )
        torch.onnx.export(
            wrapper,
            dummy_input,
            output_path.as_posix(),
            input_names=["pixel_values"],
            output_names=["logits"],
            dynamic_axes={"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=17,
        )
    LOGGER.info("ONNX model exported: %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train VeriSight Layer-2 ViT model")
    parser.add_argument("--dataset-root", type=Path, default=PROCESSED_DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/cudnn acceleration")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Cap train batches per epoch (0 = full)")
    parser.add_argument("--max-val-steps", type=int, default=0, help="Cap val batches per epoch (0 = full)")
    parser.add_argument("--max-test-steps", type=int, default=0, help="Cap test batches (0 = full)")
    parser.add_argument("--prepare-dataset", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--model-out", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--onnx-out", type=Path, default=ONNX_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    if args.epochs < 1:
        LOGGER.warning("Requested epochs %d is invalid. Using minimum allowed (1).", args.epochs)
        args.epochs = 1
    if args.epochs > 10:
        LOGGER.warning("Requested epochs %d exceeds max allowed (10). Capping to 10.", args.epochs)
        args.epochs = 10
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.max_train_steps < 0 or args.max_val_steps < 0 or args.max_test_steps < 0:
        raise ValueError("--max-*-steps values must be >= 0")
    return args


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for training, but CUDA is not available on this machine.")

    device = torch.device("cuda")
    LOGGER.info("Using device: %s", device)

    if not args.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    if args.prepare_dataset:
        prepare_dataset(output_root=args.dataset_root)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
    )

    model = build_model().to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_steps=args.max_train_steps,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            use_amp=use_amp,
            max_steps=args.max_val_steps,
        )
        scheduler.step()
        epoch_minutes = (time.perf_counter() - epoch_start) / 60.0

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | minutes=%.2f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
            epoch_minutes,
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(model, args.model_out)
            LOGGER.info("New best model saved to %s (val_acc=%.4f)", args.model_out, best_val_acc)

    load_checkpoint(model, args.model_out, device)
    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        use_amp=use_amp,
        max_steps=args.max_test_steps,
    )
    LOGGER.info("Test metrics | loss=%.4f acc=%.4f", test_metrics["loss"], test_metrics["accuracy"])

    if args.export_onnx:
        export_onnx(model, args.onnx_out, device=device, image_size=args.image_size)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
