import io
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class RandomJPEGCompressionTensor:
    def __init__(self, quality_min: int = 60, quality_max: int = 95, p: float = 0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor

        pil = transforms.ToPILImage()(tensor)
        quality = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        jpeg_img = Image.open(buf).convert("RGB")
        return transforms.ToTensor()(jpeg_img)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean()


class GanBinaryDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([float(label)], dtype=torch.float32)


@dataclass
class TrainConfig:
    dataset_root: str = "dataset"
    real_dir: str = "real"
    fake_dir: str = "gan_fake"
    checkpoints_dir: str = "checkpoints"
    best_ckpt_path: str = "checkpoints/layer3_best.pth"
    centroid_path: str = "checkpoints/clip_real_centroid.pt"
    epochs: int = 25
    batch_size: int = 64
    small_vram_batch_size: int = 32
    validation_split: float = 0.15
    test_split: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    gamma: float = 2.0
    alpha: float = 0.25
    grad_clip_max_norm: float = 1.0
    early_stopping_patience: int = 7
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    img_size: int = 224
    t0: int = 5
    t_mult: int = 2
    eta_min: float = 1e-6
    max_real_for_calibration: int = 300
    seed: int = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_samples(dataset_root: str, real_subdir: str, fake_subdir: str) -> List[Tuple[str, int]]:
    real_dir = Path(dataset_root) / real_subdir
    fake_dir = Path(dataset_root) / fake_subdir

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders not found: {real_dir} and {fake_dir}"
        )

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    real_samples = [
        (str(p), 0)
        for p in sorted(real_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]
    fake_samples = [
        (str(p), 1)
        for p in sorted(fake_dir.iterdir())
        if p.is_file() and p.suffix.lower() in exts
    ]

    # Keep binary classes balanced (max 1:1) to prevent training skew.
    target_count = min(len(real_samples), len(fake_samples))
    if target_count == 0:
        raise RuntimeError("Both classes are required with at least one image each")

    rng = random.Random(42)
    if len(real_samples) > target_count:
        real_samples = rng.sample(real_samples, target_count)
    if len(fake_samples) > target_count:
        fake_samples = rng.sample(fake_samples, target_count)

    samples = real_samples + fake_samples
    rng.shuffle(samples)
    if not samples:
        raise RuntimeError("No images found in dataset/real or dataset/gan_fake")

    labels = [lbl for _, lbl in samples]
    if len(set(labels)) < 2:
        raise RuntimeError("Both classes are required (real=0, gan_fake=1)")

    return samples


def make_transforms(cfg: TrainConfig):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.15, hue=0.08
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3
            ),
            transforms.RandomResizedCrop(
                cfg.img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            RandomJPEGCompressionTensor(quality_min=60, quality_max=95, p=0.5),
            normalize,
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def split_dataset(
    samples: List[Tuple[str, int]], val_split: float, test_split: float, seed: int
):
    paths = [p for p, _ in samples]
    labels = [y for _, y in samples]

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        paths,
        labels,
        test_size=(val_split + test_split),
        random_state=seed,
        stratify=labels,
    )

    val_ratio_within_tmp = val_split / (val_split + test_split)
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp,
        y_tmp,
        test_size=(1.0 - val_ratio_within_tmp),
        random_state=seed,
        stratify=y_tmp,
    )

    train_samples = list(zip(x_train, y_train))
    val_samples = list(zip(x_val, y_val))
    test_samples = list(zip(x_test, y_test))
    return train_samples, val_samples, test_samples


def build_dataloaders(cfg: TrainConfig):
    all_samples = build_samples(cfg.dataset_root, cfg.real_dir, cfg.fake_dir)
    train_samples, val_samples, test_samples = split_dataset(
        all_samples, cfg.validation_split, cfg.test_split, cfg.seed
    )

    train_transform, eval_transform = make_transforms(cfg)

    train_ds = GanBinaryDataset(train_samples, transform=train_transform)
    val_ds = GanBinaryDataset(val_samples, transform=eval_transform)
    test_ds = GanBinaryDataset(test_samples, transform=eval_transform)

    loader_kwargs = {
        "num_workers": cfg.num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers,
        "prefetch_factor": cfg.prefetch_factor,
    }

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader, train_samples, val_samples, test_samples


def build_models(device: torch.device):
    clip_model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    clip_model = clip_model.to(device)

    for p in clip_model.parameters():
        p.requires_grad = False

    clip_model.eval()

    head = nn.Sequential(
        nn.Linear(1024, 512),
        nn.BatchNorm1d(512),
        nn.GELU(),
        nn.Dropout(0.35),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(256, 64),
        nn.BatchNorm1d(64),
        nn.GELU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    ).to(device)

    return clip_model, head


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return 0.5


@torch.no_grad()
def evaluate(
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
):
    head.eval()

    losses = []
    all_probs = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = encoder.encode_image(imgs)
        probs = head(embeddings)

        loss = criterion(probs, labels)
        losses.append(loss.item())

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    probs_np = torch.cat(all_probs, dim=0).squeeze(1).numpy()
    labels_np = torch.cat(all_labels, dim=0).squeeze(1).numpy().astype(np.int64)

    pred_np = (probs_np >= 0.5).astype(np.int64)
    val_loss = float(np.mean(losses)) if losses else 0.0
    val_acc = float(accuracy_score(labels_np, pred_np))
    val_auc = safe_auc(labels_np, probs_np)

    return {
        "loss": val_loss,
        "acc": val_acc,
        "auc": val_auc,
        "probs": probs_np,
        "labels": labels_np,
        "preds": pred_np,
    }


def calibrate_real_centroid(
    encoder: nn.Module,
    real_dir: Path,
    eval_transform,
    device: torch.device,
    max_images: int = 300,
):
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    real_paths = [
        p for p in sorted(real_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts
    ][:max_images]

    if not real_paths:
        raise RuntimeError(f"No real images found for centroid calibration in {real_dir}")

    embs = []
    encoder.eval()
    with torch.no_grad():
        for p in tqdm(real_paths, desc="Calibrating real centroid", leave=False):
            img = Image.open(p).convert("RGB")
            x = eval_transform(img).unsqueeze(0).to(device, non_blocking=True)
            emb = encoder.encode_image(x)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embs.append(emb.cpu())

    centroid = torch.cat(embs, dim=0).mean(dim=0)
    centroid = centroid / centroid.norm()
    return centroid, len(real_paths)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this script. Install CUDA-enabled PyTorch and retry."
        )

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if vram_gb < 6.0:
        cfg.batch_size = cfg.small_vram_batch_size

    if cfg.batch_size > 64:
        cfg.batch_size = 64

    print(f"INFO | GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    print(f"INFO | batch_size={cfg.batch_size}")

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()

    train_loader, val_loader, test_loader, train_samples, val_samples, test_samples = build_dataloaders(cfg)
    print(
        f"INFO | Dataset: {len(train_samples)} train | {len(val_samples)} val | {len(test_samples)} test | classes: real=0 gan_fake=1"
    )

    encoder, head = build_models(device)

    criterion = FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha)
    optimizer = AdamW(
        head.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.t0, T_mult=cfg.t_mult, eta_min=cfg.eta_min
    )

    scaler = GradScaler()

    best_auc = -1.0
    best_acc = 0.0
    best_epoch = -1
    no_improve = 0

    print("INFO | Starting training")
    print(
        f"INFO | epochs={cfg.epochs} | optimizer=AdamW | scheduler=CosineAnnealingWarmRestarts | loss=FocalLoss"
    )

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.time()

        head.train()
        encoder.eval()

        running_loss = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs}", leave=False)

        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                embeddings = encoder.encode_image(imgs)

            with autocast(enabled=True):
                preds = head(embeddings)

            # Compute focal loss in fp32 outside autocast to keep BCE numerically safe.
            loss = criterion(preds.float(), labels.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=cfg.grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step(epoch - 1 + (len(running_loss) + 1) / max(len(train_loader), 1))

            running_loss.append(loss.item())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = float(np.mean(running_loss)) if running_loss else 0.0

        val_metrics = evaluate(
            encoder=encoder,
            head=head,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
        )

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_auc = val_metrics["auc"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start

        lr_now = optimizer.param_groups[0]["lr"]
        mem_alloc = torch.cuda.memory_allocated(device=device) / 1e9

        print("-" * 110)
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | "
            f"LR: {lr_now:.2e} | Time: {epoch_time:.1f}s | GPU Mem: {mem_alloc:.2f} GB"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            ckpt = {
                "encoder_state": encoder.state_dict(),
                "head_state": head.state_dict(),
                "val_auc": best_auc,
                "val_acc": best_acc,
                "epoch": best_epoch,
                "config": asdict(cfg),
            }
            torch.save(ckpt, cfg.best_ckpt_path, pickle_protocol=2)
            print(
                f"INFO | Checkpoint saved -> {cfg.best_ckpt_path} | val_auc={best_auc:.4f} | val_acc={best_acc:.4f}"
            )
        else:
            no_improve += 1
            print(
                f"INFO | No AUC improvement for {no_improve} epoch(s). Best AUC={best_auc:.4f}"
            )

        if no_improve >= cfg.early_stopping_patience:
            print(
                f"INFO | Early stopping triggered at epoch {epoch}. Patience={cfg.early_stopping_patience}."
            )
            break

    if os.path.exists(cfg.best_ckpt_path):
        best_ckpt = torch.load(cfg.best_ckpt_path, map_location=device)
        encoder.load_state_dict(best_ckpt["encoder_state"])
        head.load_state_dict(best_ckpt["head_state"])
        print(
            f"INFO | Loaded best checkpoint from epoch {best_ckpt['epoch']} with AUC={best_ckpt['val_auc']:.4f}"
        )

    _, eval_transform = make_transforms(cfg)
    centroid, n_real = calibrate_real_centroid(
        encoder=encoder,
        real_dir=Path(cfg.dataset_root) / cfg.real_dir,
        eval_transform=eval_transform,
        device=device,
        max_images=cfg.max_real_for_calibration,
    )
    torch.save(centroid, cfg.centroid_path, pickle_protocol=2)
    print(
        f"CLIP centroid calibrated on {n_real} real images. Saved to {cfg.centroid_path}"
    )

    test_metrics = evaluate(
        encoder=encoder,
        head=head,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
    )

    test_labels = test_metrics["labels"]
    test_preds = test_metrics["preds"]
    test_probs = test_metrics["probs"]

    test_acc = float(accuracy_score(test_labels, test_preds))
    test_precision = float(precision_score(test_labels, test_preds, zero_division=0))
    test_recall = float(recall_score(test_labels, test_preds, zero_division=0))
    test_f1 = float(f1_score(test_labels, test_preds, zero_division=0))
    test_auc = safe_auc(test_labels, test_probs)
    test_cm = confusion_matrix(test_labels, test_preds, labels=[0, 1])

    print("=" * 110)
    print("Final Evaluation (Test Set)")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print(f"AUC-ROC:   {test_auc:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    print(f"Layer 3 training COMPLETE. Best AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
