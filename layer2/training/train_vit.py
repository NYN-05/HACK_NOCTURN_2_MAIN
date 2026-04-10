"""Train VeriSight Layer 2 with a pretrained Hugging Face ViT-B/16 backbone.

This version is optimized for local laptop training:
- pretrained Hugging Face ViT-B/16 fine-tuning
- head-only warmup with progressive unfreezing
- mixed precision on CUDA
- class-weighted loss with MixUp/CutMix opt-in only
- source-aware / duplicate-aware split handling
- gradient accumulation for small-memory setups
- early stopping driven by validation F1 with recall tie-breaking
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
import warnings
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layer2.training.dataset_loader_refactored import build_dataloaders, prepare_dataset
from layer2.utils.config import BEST_MODEL_PATH, CLEANED_DATA_ROOT, NUM_CLASSES, ONNX_MODEL_PATH
from layer2.utils.metrics import summarize_epoch

LOGGER = logging.getLogger(__name__)
DEFAULT_HF_MODEL_NAME = "google/vit-base-patch16-224"


def _resolve_num_workers(requested_workers: int, device: torch.device) -> int:
    requested_workers = max(0, requested_workers)
    if device.type == "cpu":
        return min(requested_workers, 2)
    if platform.system().lower() != "windows":
        return requested_workers
    if requested_workers == 0:
        requested_workers = 2
    return min(max(requested_workers, 2), 4)


def _resolve_model_name(requested_model: str, device: torch.device) -> str:
    if requested_model != "auto":
        return requested_model
    return DEFAULT_HF_MODEL_NAME


def build_model(model_name: str, drop_rate: float, drop_path_rate: float) -> nn.Module:
    """Build ViT model with enhanced regularization for forgery detection.
    
    Increases dropout rates to counter overfitting on synthetic artifacts.
    Applies L2 regularization via weight decay (handled in optimizer).
    """
    try:
        config = ViTConfig.from_pretrained(model_name)
        config.num_labels = NUM_CLASSES
        config.id2label = {0: "REAL", 1: "AI_GENERATED"}
        config.label2id = {"REAL": 0, "AI_GENERATED": 1}
        # Enhance dropout for forgery detection task (higher than default)
        if drop_rate > 0:
            config.hidden_dropout_prob = drop_rate * 1.5  # 0.1 -> 0.15
            config.attention_probs_dropout_prob = drop_rate * 1.5
            if hasattr(config, "classifier_dropout"):
                config.classifier_dropout = drop_rate * 1.5
        if drop_path_rate > 0:
            setattr(config, "drop_path_rate", drop_path_rate * 1.2)  # 0.1 -> 0.12
        model = ViTForImageClassification.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )
    except Exception as exc:  # pragma: no cover - guarded by runtime environment
        raise RuntimeError(
            f"Failed to create pretrained model '{model_name}'. Ensure transformers can download the weights."
        ) from exc
    return model


def _get_transformer_blocks(model: nn.Module) -> list[nn.Module]:
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        vit = getattr(model, "vit", None)
        if vit is not None:
            encoder = getattr(vit, "encoder", None)
            if encoder is not None:
                layers = getattr(encoder, "layer", None)
                if layers is not None:
                    return list(layers)
        return []
    return list(blocks)


def _set_module_trainable(module: nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad


def _set_trainable_stage(model: nn.Module, trainable_top_blocks: int) -> None:
    for param in model.parameters():
        param.requires_grad = False

    head_modules = [
        getattr(model, "head", None),
        getattr(model, "head_dist", None),
        getattr(model, "pre_logits", None),
        getattr(model, "classifier", None),
    ]
    for module in head_modules:
        _set_module_trainable(module, True)

    total_blocks = len(_get_transformer_blocks(model))
    if trainable_top_blocks <= 0 or total_blocks == 0:
        return

    if trainable_top_blocks >= total_blocks:
        for param in model.parameters():
            param.requires_grad = True
        return

    for block in _get_transformer_blocks(model)[-trainable_top_blocks:]:
        _set_module_trainable(block, True)

    _set_module_trainable(getattr(model, "norm", None), True)
    vit = getattr(model, "vit", None)
    if vit is not None:
        _set_module_trainable(getattr(vit, "layernorm", None), True)


def _compute_trainable_blocks(epoch: int, head_only_epochs: int, unfreeze_every: int, blocks_per_stage: int, total_blocks: int) -> int:
    if total_blocks <= 0:
        return 0
    if epoch <= head_only_epochs:
        return 0
    if unfreeze_every <= 0 or blocks_per_stage <= 0:
        return total_blocks
    stage_index = 1 + ((epoch - head_only_epochs - 1) // unfreeze_every)
    return min(total_blocks, stage_index * blocks_per_stage)


def _build_param_groups(model: nn.Module, head_lr: float, backbone_lr: float, weight_decay: float) -> list[dict]:
    head_decay: list[nn.Parameter] = []
    head_no_decay: list[nn.Parameter] = []
    backbone_decay: list[nn.Parameter] = []
    backbone_no_decay: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            # Parameters are frozen early, but keep them in the optimizer so they
            # automatically start updating when the progressive schedule unfreezes them.
            pass

        is_head = name.startswith(("head", "head_dist", "pre_logits", "classifier"))
        no_decay = name.endswith("bias") or "norm" in name.lower() or "bn" in name.lower()
        if is_head:
            (head_no_decay if no_decay else head_decay).append(param)
        else:
            (backbone_no_decay if no_decay else backbone_decay).append(param)

    return [
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0},
        {"params": head_decay, "lr": head_lr, "weight_decay": weight_decay},
        {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, head_lr: float, backbone_lr: float, weight_decay: float) -> AdamW:
    param_groups = _build_param_groups(model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=weight_decay)
    return AdamW(param_groups)


def _build_class_weights(train_label_counts: Counter, device: torch.device, soften_with_sampler: bool) -> torch.Tensor:
    """Build class weights for imbalanced datasets.
    
    When soften_with_sampler=True (balanced sampling enabled), apply sqrt softening
    to reduce double-correction (sampler + loss both don't overcorrect in same direction).
    When False, use raw inverse frequency weights.
    """
    train_total = sum(train_label_counts.values())
    weights = torch.tensor(
        [train_total / (2.0 * train_label_counts[0]), train_total / (2.0 * train_label_counts[1])],
        device=device,
        dtype=torch.float32,
    )
    if soften_with_sampler:
        # Sqrt softening: reduces aggressive class weighting when sampler already balances
        weights = torch.sqrt(weights)
        weights = weights / weights.mean().clamp_min(1e-6)
    return weights


def build_mixup(args, num_classes: int) -> Mixup | None:
    """Build MixUp/CutMix augmentation for corruption robustness.
    
    MixUp helps with blended fakes; CutMix helps with spliced regions.
    Enabled by default (now=True) because forgery detection benefits from interpolation robustness.
    """
    if not args.use_mixup:
        return None
    if args.mixup_alpha <= 0 and args.cutmix_alpha <= 0:
        return None
    LOGGER.info(
        "MixUp enabled: mixup_alpha=%.2f cutmix_alpha=%.2f prob=%.2f switch_prob=%.2f",
        args.mixup_alpha,
        args.cutmix_alpha,
        args.mixup_prob,
        args.mixup_switch_prob,
    )
    return Mixup(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_minmax=None,
        prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        mode=args.mixup_mode,
        label_smoothing=args.label_smoothing,
        num_classes=num_classes,
    )


def build_criteria(use_mixup: bool, label_smoothing: float, class_weights: torch.Tensor | None = None):
    if use_mixup:
        eval_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        return SoftTargetCrossEntropy(), eval_criterion
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing), nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
    mixup_fn=None,
    grad_accum_steps: int = 1,
    max_steps: int = 0,
    grad_clip: float = 1.0,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    loss_sum = 0.0
    prediction_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    steps = 0
    accumulation_counter = 0
    autocast_enabled = use_amp and device.type == "cuda"

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    with torch.set_grad_enabled(is_train):
        for batch_index, batch in enumerate(tqdm(loader, total=len(loader), leave=False)):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            hard_labels = labels

            if is_train and mixup_fn is not None:
                pixel_values, labels = mixup_fn(pixel_values, labels)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                logits = model(pixel_values=pixel_values).logits
                loss = criterion(logits, labels)
                loss_to_backprop = loss / max(1, grad_accum_steps)

            if is_train:
                accumulation_counter += 1
                if scaler is not None and autocast_enabled:
                    scaler.scale(loss_to_backprop).backward()
                else:
                    loss_to_backprop.backward()

                should_step = accumulation_counter >= grad_accum_steps or (max_steps > 0 and steps + 1 >= max_steps)
                if should_step:
                    if scaler is not None and autocast_enabled:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accumulation_counter = 0

            loss_sum += float(loss.item())
            prediction_batches.append(torch.argmax(logits.detach(), dim=1).to("cpu"))
            label_batches.append(hard_labels.detach().to("cpu"))
            steps += 1

            if max_steps > 0 and steps >= max_steps:
                break

    if is_train and accumulation_counter > 0:
        if scaler is not None and autocast_enabled:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    predictions = torch.cat(prediction_batches) if prediction_batches else torch.empty(0, dtype=torch.long)
    all_labels = torch.cat(label_batches) if label_batches else torch.empty(0, dtype=torch.long)
    return summarize_epoch(loss_sum, predictions, all_labels, steps)


@torch.no_grad()
def calibrate_decision_threshold(model: nn.Module, loader, device: torch.device, min_precision: float = 0.5) -> dict:
    model.eval()
    probability_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    for batch in tqdm(loader, total=len(loader), leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model(pixel_values=pixel_values).logits
        probabilities = torch.softmax(logits, dim=1)[:, 1]
        probability_batches.append(probabilities.detach().cpu())
        label_batches.append(labels.detach().cpu())

    if not probability_batches:
        return {
            "threshold": 0.5,
            "uncertain_lower": 0.4,
            "uncertain_upper": 0.6,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    probabilities = torch.cat(probability_batches)
    labels = torch.cat(label_batches).to(torch.int64)

    best_candidate = {
        "threshold": 0.5,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "meets_precision_floor": False,
    }

    for threshold in torch.linspace(0.05, 0.95, steps=91):
        predictions = (probabilities >= threshold).to(torch.int64)
        tp = int(((predictions == 1) & (labels == 1)).sum().item())
        tn = int(((predictions == 0) & (labels == 0)).sum().item())
        fp = int(((predictions == 1) & (labels == 0)).sum().item())
        fn = int(((predictions == 0) & (labels == 1)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        meets_floor = precision >= min_precision

        candidate_better = False
        if meets_floor and not best_candidate["meets_precision_floor"]:
            candidate_better = True
        elif meets_floor == best_candidate["meets_precision_floor"]:
            if f1 > best_candidate["f1"]:
                candidate_better = True
            elif abs(f1 - best_candidate["f1"]) <= 1e-9 and recall > best_candidate["recall"]:
                candidate_better = True
            elif abs(f1 - best_candidate["f1"]) <= 1e-9 and abs(recall - best_candidate["recall"]) <= 1e-9 and float(threshold) < best_candidate["threshold"]:
                candidate_better = True

        if candidate_better:
            best_candidate = {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "meets_precision_floor": meets_floor,
            }

    threshold = best_candidate["threshold"]
    return {
        "threshold": threshold,
        "uncertain_lower": max(0.0, threshold - 0.1),
        "uncertain_upper": min(1.0, threshold + 0.1),
        "precision": best_candidate["precision"],
        "recall": best_candidate["recall"],
        "f1": best_candidate["f1"],
    }


def save_checkpoint(state: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    return torch.load(checkpoint_path, map_location=device)


def export_onnx(model: nn.Module, output_path: Path, device: torch.device, image_size: int = 224):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    class OnnxWrapper(nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values):
            return self.wrapped(pixel_values=pixel_values).logits

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
    parser = argparse.ArgumentParser(description="Train VeriSight Layer 2 with transfer learning")
    parser.add_argument("--dataset-root", type=Path, default=CLEANED_DATA_ROOT)
    parser.add_argument("--model-name", type=str, default="auto", help="Hugging Face model name, or auto for the default google/vit-base-patch16-224")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-effective-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4, help="Head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=1e-5, help="Backbone learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-balanced-sampling", dest="balanced_sampling", action="store_false", help="Disable weighted sampling for class balance")
    parser.set_defaults(balanced_sampling=True)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul/cudnn acceleration")
    parser.add_argument("--channels-last", action="store_true", help="Use channels_last memory format")
    parser.add_argument("--head-only-epochs", type=int, default=2, help="Epochs to train only the classification head")
    parser.add_argument("--unfreeze-every", type=int, default=2, help="Epoch interval for progressive unfreezing")
    parser.add_argument("--blocks-per-stage", type=int, default=2, help="Number of transformer blocks to unfreeze at each stage")
    parser.add_argument("--max-train-steps", type=int, default=0, help="Cap train batches per epoch (0 = full)")
    parser.add_argument("--max-val-steps", type=int, default=0, help="Cap val batches per epoch (0 = full)")
    parser.add_argument("--max-test-steps", type=int, default=0, help="Cap test batches (0 = full)")
    parser.add_argument("--prepare-dataset", action="store_true")
    parser.add_argument("--use-labeled-subset", action="store_true", help="Train from the metadata-backed cleaned_data subset instead of the full cleaned_data corpus")
    parser.add_argument("--export-onnx", action="store_true")
    parser.add_argument("--model-out", type=Path, default=BEST_MODEL_PATH)
    parser.add_argument("--onnx-out", type=Path, default=ONNX_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=BEST_MODEL_PATH.parent / "vit_layer2_training_metrics.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.3, help="MixUp blending strength")
    parser.add_argument("--cutmix-alpha", type=float, default=0.5, help="CutMix blending strength")
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", type=str, default="batch")
    parser.add_argument("--use-mixup", action="store_true", help="Enable MixUp/CutMix augmentation (now default for better robustness)")
    parser.add_argument("--no-mixup", dest="use_mixup", action="store_false", help=argparse.SUPPRESS)
    parser.set_defaults(use_mixup=True)
    parser.add_argument("--small-dataset-threshold", type=int, default=5000, help="Below this dataset size, keep the backbone frozen")
    args = parser.parse_args()

    if args.epochs < 1:
        LOGGER.warning("Requested epochs %d is invalid. Using minimum allowed (1).", args.epochs)
        args.epochs = 1
    if args.epochs > 50:
        LOGGER.warning("Requested epochs %d exceeds max allowed (50). Capping to 50.", args.epochs)
        args.epochs = 50
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.target_effective_batch_size < args.batch_size:
        args.target_effective_batch_size = args.batch_size
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be >= 0")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be > 0")
    if not (0.0 <= args.label_smoothing < 1.0):
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.patience < 1:
        raise ValueError("--patience must be >= 1")
    if args.min_delta < 0:
        raise ValueError("--min-delta must be >= 0")
    if args.max_train_steps < 0 or args.max_val_steps < 0 or args.max_test_steps < 0:
        raise ValueError("--max-*-steps values must be >= 0")
    if args.head_only_epochs < 0:
        raise ValueError("--head-only-epochs must be >= 0")
    if args.unfreeze_every < 0:
        raise ValueError("--unfreeze-every must be >= 0")
    if args.blocks_per_stage < 0:
        raise ValueError("--blocks-per-stage must be >= 0")
    if args.drop_rate < 0 or args.drop_path_rate < 0:
        raise ValueError("--drop-rate and --drop-path-rate must be >= 0")
    if args.mixup_alpha < 0 or args.cutmix_alpha < 0:
        raise ValueError("--mixup-alpha and --cutmix-alpha must be >= 0")
    if args.mixup_prob < 0 or args.mixup_prob > 1:
        raise ValueError("--mixup-prob must be in [0, 1]")
    if args.mixup_switch_prob < 0 or args.mixup_switch_prob > 1:
        raise ValueError("--mixup-switch-prob must be in [0, 1]")
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    if args.prepare_dataset:
        args.dataset_root = prepare_dataset(output_root=args.dataset_root, use_labeled_only=args.use_labeled_subset)

    if device.type == "cuda" and args.batch_size == 8:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory_gb >= 16.0:
            args.batch_size = 32
        elif gpu_memory_gb >= 10.0:
            args.batch_size = 16
        elif gpu_memory_gb >= 7.0:
            args.batch_size = 8
        else:
            args.batch_size = 4
        LOGGER.info("Auto-tuned batch size to %d from GPU memory %.1f GB", args.batch_size, gpu_memory_gb)
    elif device.type == "cpu":
        args.batch_size = min(args.batch_size, 4)
        LOGGER.info("CPU fallback active; batch size capped at %d", args.batch_size)

    if not args.no_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = device.type == "cuda"
    torch.set_float32_matmul_precision("high")

    requested_workers = args.num_workers
    args.num_workers = _resolve_num_workers(args.num_workers, device)
    if args.num_workers != requested_workers:
        LOGGER.info("Using %d DataLoader workers (requested %d).", args.num_workers, requested_workers)

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        pin_memory=device.type == "cuda",
        multiprocessing_context="spawn",
        balanced_sampling=args.balanced_sampling,
        use_complete_dataset=not args.use_labeled_subset,
    )

    train_label_counts = Counter(sample.label for sample in getattr(train_loader.dataset, "samples", []))
    if len(train_label_counts) < 2 or any(train_label_counts.get(label, 0) == 0 for label in (0, 1)):
        raise ValueError(f"Layer 2 needs both classes in the training split; found counts={dict(train_label_counts)}")
    class_weights = _build_class_weights(train_label_counts, device=device, soften_with_sampler=args.balanced_sampling)

    train_dataset_size = len(train_loader.dataset)
    LOGGER.info(
        "[DATA] Layer2 samples train=%d val=%d test=%d",
        train_dataset_size,
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    if train_dataset_size < args.small_dataset_threshold:
        LOGGER.warning(
            "Dataset size %d is below threshold %d; keeping the backbone frozen for the entire run.",
            train_dataset_size,
            args.small_dataset_threshold,
        )
        args.head_only_epochs = args.epochs
        args.unfreeze_every = 0
        args.blocks_per_stage = 0

    model_name = _resolve_model_name(args.model_name, device)
    model = build_model(model_name, drop_rate=args.drop_rate, drop_path_rate=args.drop_path_rate).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    total_blocks = len(_get_transformer_blocks(model))
    _set_trainable_stage(model, trainable_top_blocks=0)

    optimizer = build_optimizer(model, head_lr=args.lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay)
    grad_accum_steps = max(1, math.ceil(args.target_effective_batch_size / args.batch_size))
    LOGGER.info(
        "Effective batch size=%d (micro-batch=%d, grad_accum_steps=%d)",
        args.batch_size * grad_accum_steps,
        args.batch_size,
        grad_accum_steps,
    )

    warmup_epochs = min(args.warmup_epochs, max(0, args.epochs - 1))
    if warmup_epochs > 0:
        # Enhanced warmup: gradual LR increase from low to target
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        # Cosine annealing for smooth convergence with flat tail
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=args.min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
        LOGGER.info("Using warmup (%d epochs) + cosine annealing", warmup_epochs)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr)
        LOGGER.info("Using cosine annealing without warmup")

    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    mixup_fn = build_mixup(args, NUM_CLASSES)
    train_criterion, eval_criterion = build_criteria(use_mixup=mixup_fn is not None, label_smoothing=args.label_smoothing, class_weights=class_weights)

    start_epoch = 1
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_val_precision = -1.0
    best_val_recall = -1.0
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    history = []

    if args.resume:
        if args.model_out.exists():
            latest_ckpt = load_checkpoint(args.model_out, device)
            model.load_state_dict(latest_ckpt["model_state_dict"])
            optimizer.load_state_dict(latest_ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in latest_ckpt:
                scheduler.load_state_dict(latest_ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in latest_ckpt and latest_ckpt["scaler_state_dict"] is not None:
                scaler.load_state_dict(latest_ckpt["scaler_state_dict"])
            start_epoch = int(latest_ckpt.get("epoch", 0)) + 1
            best_val_acc = float(latest_ckpt.get("best_val_acc", best_val_acc))
            best_val_loss = float(latest_ckpt.get("best_val_loss", best_val_loss))
            best_val_precision = float(latest_ckpt.get("best_val_precision", best_val_precision))
            best_val_recall = float(latest_ckpt.get("best_val_recall", best_val_recall))
            best_val_f1 = float(latest_ckpt.get("best_val_f1", best_val_f1))
            best_epoch = int(latest_ckpt.get("best_epoch", best_epoch))
            no_improve = int(latest_ckpt.get("no_improve", no_improve))
            loaded_history = latest_ckpt.get("history", [])
            if isinstance(loaded_history, list):
                history.extend(loaded_history)
            LOGGER.info("Resumed Layer 2 training from %s at epoch %d", args.model_out, start_epoch)
        else:
            LOGGER.warning("--resume was requested but no best checkpoint exists at %s", args.model_out)

    for epoch in range(start_epoch, args.epochs + 1):
        trainable_blocks = _compute_trainable_blocks(
            epoch=epoch,
            head_only_epochs=args.head_only_epochs,
            unfreeze_every=args.unfreeze_every,
            blocks_per_stage=args.blocks_per_stage,
            total_blocks=total_blocks,
        )
        _set_trainable_stage(model, trainable_top_blocks=trainable_blocks)
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        LOGGER.info(
            "Epoch %d/%d | trainable backbone blocks=%d | trainable params=%d",
            epoch,
            args.epochs,
            trainable_blocks,
            trainable_params,
        )

        epoch_start = time.perf_counter()
        train_metrics = run_epoch(
            model,
            train_loader,
            train_criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            mixup_fn=mixup_fn,
            grad_accum_steps=grad_accum_steps,
            max_steps=args.max_train_steps,
            grad_clip=args.grad_clip,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            eval_criterion,
            device,
            optimizer=None,
            use_amp=use_amp,
            max_steps=args.max_val_steps,
            grad_clip=args.grad_clip,
        )
        scheduler.step()
        epoch_minutes = (time.perf_counter() - epoch_start) / 60.0

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f train_recall=%.4f | val_loss=%.4f val_recall=%.4f | minutes=%.2f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["recall"],
            val_metrics["loss"],
            val_metrics["recall"],
            epoch_minutes,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "lr_head": optimizer.param_groups[2]["lr"],
                "lr_backbone": optimizer.param_groups[0]["lr"],
                "trainable_blocks": trainable_blocks,
            }
        )

        # Improved metric: optimize for F1 first (not recall), with precision tie-break
        # This prevents the model from overfitting to detect everything as fake (high recall, low precision)
        improved = (
            val_metrics["f1"] > best_val_f1 + args.min_delta
            or (
                abs(val_metrics["f1"] - best_val_f1) <= args.min_delta
                and val_metrics["precision"] > best_val_precision + args.min_delta
            )
            or (
                abs(val_metrics["f1"] - best_val_f1) <= args.min_delta
                and abs(val_metrics["precision"] - best_val_precision) <= args.min_delta
                and val_metrics["recall"] > best_val_recall + args.min_delta
            )
        )
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_acc = val_metrics["accuracy"]
            best_val_precision = val_metrics["precision"]
            best_val_recall = val_metrics["recall"]
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            no_improve = 0
            checkpoint_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
                "best_val_precision": best_val_precision,
                "best_val_recall": best_val_recall,
                "best_val_f1": best_val_f1,
                "best_epoch": best_epoch,
                "no_improve": no_improve,
                "history": history,
                "model_name": model_name,
            }
            save_checkpoint(checkpoint_state, args.model_out)
            # Determine which metric drove the improvement
            improvement_reason = "f1"
            if (
                abs(val_metrics["f1"] - best_val_f1) <= args.min_delta
                and val_metrics["precision"] > best_val_precision + args.min_delta
            ):
                improvement_reason = "precision (F1 tied)"
            elif (
                abs(val_metrics["f1"] - best_val_f1) <= args.min_delta
                and abs(val_metrics["precision"] - best_val_precision) <= args.min_delta
            ):
                improvement_reason = "recall (F1 and precision tied)"
            LOGGER.info(
                "✓ New best model saved (reason: %s) | val_f1=%.4f val_prec=%.4f val_rec=%.4f | path=%s",
                improvement_reason,
                best_val_f1,
                best_val_precision,
                best_val_recall,
                args.model_out,
            )
        else:
            no_improve += 1
            LOGGER.info("No validation improvement for %d/%d epoch(s)", no_improve, args.patience)
            if no_improve >= args.patience:
                LOGGER.info("Early stopping triggered at epoch %d (patience=%d)", epoch, args.patience)
                break

    best_ckpt = load_checkpoint(args.model_out, device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    threshold_calibration = calibrate_decision_threshold(model, val_loader, device=device)
    LOGGER.info(
        "Validation threshold calibration | threshold=%.2f uncertain=[%.2f, %.2f] precision=%.4f recall=%.4f f1=%.4f",
        threshold_calibration["threshold"],
        threshold_calibration["uncertain_lower"],
        threshold_calibration["uncertain_upper"],
        threshold_calibration["precision"],
        threshold_calibration["recall"],
        threshold_calibration["f1"],
    )
    test_metrics = run_epoch(
        model,
        test_loader,
        eval_criterion,
        device,
        optimizer=None,
        use_amp=use_amp,
        max_steps=args.max_test_steps,
        grad_clip=args.grad_clip,
    )
    LOGGER.info("Test metrics | loss=%.4f recall=%.4f f1=%.4f", test_metrics["loss"], test_metrics["recall"], test_metrics["f1"])

    metrics_payload = {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_val_precision": best_val_precision,
        "best_val_recall": best_val_recall,
        "best_val_f1": best_val_f1,
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_false_negative_rate": test_metrics["false_negative_rate"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "recommended_threshold": threshold_calibration["threshold"],
        "uncertain_lower": threshold_calibration["uncertain_lower"],
        "uncertain_upper": threshold_calibration["uncertain_upper"],
        "threshold_precision": threshold_calibration["precision"],
        "threshold_recall": threshold_calibration["recall"],
        "threshold_f1": threshold_calibration["f1"],
        "history": history,
    }
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    LOGGER.info("Training metrics saved to %s", args.metrics_out)

    if args.export_onnx:
        export_onnx(model, args.onnx_out, device=device, image_size=args.image_size)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
