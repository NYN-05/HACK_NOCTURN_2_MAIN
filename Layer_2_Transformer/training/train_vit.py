import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTForImageClassification

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
    model = ViTForImageClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    return model


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum = 0.0
    acc_sum = 0.0
    steps = 0

    with torch.set_grad_enabled(is_train):
        for batch in tqdm(loader, total=len(loader), leave=False):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_acc = compute_accuracy(logits.detach(), labels)
            loss_sum += loss.item()
            acc_sum += batch_acc
            steps += 1

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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prepare-dataset", action="store_true")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--model-out", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--onnx-out", type=Path, default=ONNX_MODEL_PATH)
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    if args.prepare_dataset:
        prepare_dataset(output_root=args.dataset_root)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["accuracy"],
            val_metrics["loss"],
            val_metrics["accuracy"],
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_checkpoint(model, args.model_out)
            LOGGER.info("New best model saved to %s (val_acc=%.4f)", args.model_out, best_val_acc)

    load_checkpoint(model, args.model_out, device)
    test_metrics = run_epoch(model, test_loader, criterion, device, optimizer=None)
    LOGGER.info("Test metrics | loss=%.4f acc=%.4f", test_metrics["loss"], test_metrics["accuracy"])

    if args.export_onnx:
        export_onnx(model, args.onnx_out, device=device, image_size=args.image_size)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
