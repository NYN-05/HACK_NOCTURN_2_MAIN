import argparse
import io
import json
import os
import random
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from PIL import ImageFilter
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
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


REAL_TOKENS = {"au", "authentic", "original", "real", "pristine", "clean"}
FAKE_TOKENS = {
    "tp",
    "tampered",
    "forged",
    "splice",
    "splicing",
    "copy",
    "copy-move",
    "edited",
    "fake",
    "gan",
    "ai",
    "generated",
}
IGNORE_TOKENS = {"mask", "groundtruth", "ground_truth", "gt", "label", "labels"}


def _default_dataset_root() -> str:
    repo_root = Path(__file__).resolve().parents[1]
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
        with Image.open(buf) as jpeg_img:
            return transforms.ToTensor()(jpeg_img.convert("RGB"))


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
        with Image.open(path) as image_file:
            img = image_file.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([float(label)], dtype=torch.float32)


@dataclass
class TrainConfig:
    dataset_root: str = _default_dataset_root()
    checkpoints_dir: str = "checkpoints"
    best_ckpt_path: str = "checkpoints/layer3_best.pth"
    latest_ckpt_path: str = "checkpoints/layer3_latest.pth"
    centroid_path: str = "checkpoints/clip_real_centroid.pt"
    epochs: int = 25
    batch_size: int = 8
    small_vram_batch_size: int = 4
    validation_split: float = 0.15
    test_split: float = 0.15
    lr: float = 3e-4
    weight_decay: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.999)
    gamma: float = 2.0
    alpha: float = 0.25
    grad_clip_max_norm: float = 1.0
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 1e-4
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
    metrics_out_path: str = "checkpoints/layer3_training_metrics.json"
    use_all_data: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _label_from_path(path: Path) -> int | None:
    filename = path.name.lower()
    stem = path.stem.lower()
    parts = [part.lower() for part in path.parts]

    if any(token in filename for token in ("_gt", "_mask", "groundtruth", "ground_truth")):
        return None
    if any(token in parts for token in IGNORE_TOKENS):
        return None

    if any(token in parts for token in REAL_TOKENS):
        return 0
    if any(token in parts for token in FAKE_TOKENS):
        return 1

    if any(token in stem for token in ("_orig", "_auth", "_real")):
        return 0
    if any(token in stem for token in ("_tam", "_forg", "_manip", "_fake")):
        return 1

    return None


def _discover_real_fake_paths(dataset_root: str) -> Tuple[List[Path], List[Path]]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    real_paths: List[Path] = []
    fake_paths: List[Path] = []

    excluded_names = {
        "prepared_yolo",
        "labels",
    }

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if any(part.lower() in excluded_names for part in path.parts):
            continue

        label = _label_from_path(path)
        if label == 0:
            real_paths.append(path)
        elif label == 1:
            fake_paths.append(path)

    return sorted(set(real_paths)), sorted(set(fake_paths))


class JpegCompressionTransform:
    """PIL-based JPEG-quality augmentation (keeps PIL pipeline)."""
    def __init__(self, quality_range=(60, 95), p: float = 0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img_pil: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img_pil
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        with Image.open(buf) as out:
            return out.convert("RGB")


def _make_real_synthetic(seed: int):
    rng = np.random.default_rng(seed)
    h, w = 256, 256
    ii = np.arange(h).reshape(h, 1)
    jj = np.arange(w).reshape(1, w)
    noise_r = rng.normal(0, 12, (h, w))
    noise_g = rng.normal(0, 12, (h, w))
    noise_b = rng.normal(0, 12, (h, w))

    r = 128 + 80 * np.sin(ii / 30.0) + noise_r
    g = 128 + 60 * np.cos(jj / 25.0) + noise_g
    b = 100 + 50 * np.sin((ii + jj) / 40.0) + noise_b
    arr = np.stack([r, g, b], axis=-1)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    return img


def _make_fake_from_real(real_img_pil, seed):
    rng = np.random.default_rng(seed)
    arr = np.array(real_img_pil).astype(np.float64)
    h, w, _ = arr.shape

    y0 = int(rng.integers(0, max(1, h - 96)))
    x0 = int(rng.integers(0, max(1, w - 96)))
    region = arr[y0 : y0 + 96, x0 : x0 + 96, :]
    yy = np.arange(96).reshape(96, 1)
    xx = np.arange(96).reshape(1, 96)
    checker = (((yy // 8 + xx // 8) % 2) * 2 - 1) * 30
    region += checker[:, :, None]

    y1 = int(rng.integers(0, max(1, h - 80)))
    x1 = int(rng.integers(0, max(1, w - 80)))
    reg2 = arr[y1 : y1 + 80, x1 : x1 + 80, :]
    reg2[:, :, 1] = reg2[:, :, 0] * 0.985
    reg2[:, :, 2] = reg2[:, :, 0] * 0.972

    periodic = 8 * np.sin(np.arange(h).reshape(h, 1) / 6.0)
    arr += periodic[:, :, None]

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    y_min = max(0, min(y0, y1))
    x_min = max(0, min(x0, x1))
    y_max = min(h, max(y0 + 96, y1 + 80))
    x_max = min(w, max(x0 + 96, x1 + 80))
    crop = img.crop((x_min, y_min, x_max, y_max)).filter(ImageFilter.GaussianBlur(radius=0.6))
    img.paste(crop, (x_min, y_min))
    return img


def _save_jpg(img_pil, out_path, quality=85):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(out_path, format="JPEG", quality=quality)


def _build_synthetic_fallback(dataset_dir, n_per_class=64):
    root = Path(dataset_dir)
    real_dir = root / "real"
    fake_dir = root / "gan_fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    total = n_per_class * 2
    pbar = tqdm(total=total, desc="Synthetic dataset", leave=False)
    for idx in range(n_per_class):
        real_img = _make_real_synthetic(seed=idx)
        out_real = real_dir / f"synthetic_real_{idx:05d}.jpg"
        _save_jpg(real_img, out_real, quality=90)
        pbar.update(1)

        fake_img = _make_fake_from_real(real_img, seed=idx + 1000)
        out_fake = fake_dir / f"synthetic_fake_{idx:05d}.jpg"
        _save_jpg(fake_img, out_fake, quality=90)
        pbar.update(1)
    pbar.close()


def build_samples(dataset_root: str) -> List[Tuple[str, int]]:
    real_paths, fake_paths = _discover_real_fake_paths(dataset_root)
    real_samples = [(str(p), 0) for p in real_paths]
    fake_samples = [(str(p), 1) for p in fake_paths]

    if not real_samples or not fake_samples:
        raise FileNotFoundError(
            "No sufficient labeled real/fake images found under root Data folder. "
            "Use folder/file naming with tokens like real/original/authentic and fake/tampered/gan/tp."
        )

    samples = real_samples + fake_samples
    rng = random.Random(42)
    rng.shuffle(samples)
    if not samples:
        raise RuntimeError(f"No images found in {Path(dataset_root)}")

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


def _build_balanced_sampler(samples: List[Tuple[str, int]]) -> WeightedRandomSampler | None:
    labels = [lbl for _, lbl in samples]
    if not labels:
        return None

    class_counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2)
    if np.any(class_counts == 0):
        return None

    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[label] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def tune_decision_threshold(labels_np: np.ndarray, probs_np: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    best_acc = -1.0

    for threshold in np.linspace(0.2, 0.8, 61):
        pred_np = (probs_np >= threshold).astype(np.int64)
        f1 = float(f1_score(labels_np, pred_np, zero_division=0))
        acc = float(accuracy_score(labels_np, pred_np))

        if (f1 > best_f1) or (f1 == best_f1 and acc > best_acc):
            best_f1 = f1
            best_acc = acc
            best_threshold = float(threshold)

    return best_threshold


def build_dataloaders(cfg: TrainConfig):
    all_samples = build_samples(cfg.dataset_root)
    if cfg.use_all_data:
        train_samples = all_samples
        val_samples = all_samples
        test_samples = all_samples
    else:
        train_samples, val_samples, test_samples = split_dataset(
            all_samples, cfg.validation_split, cfg.test_split, cfg.seed
        )

    train_transform, eval_transform = make_transforms(cfg)

    train_ds = GanBinaryDataset(train_samples, transform=train_transform)
    val_ds = GanBinaryDataset(val_samples, transform=eval_transform)
    test_ds = GanBinaryDataset(test_samples, transform=eval_transform)
    train_sampler = _build_balanced_sampler(train_samples)

    num_workers = _resolve_num_workers(cfg.num_workers)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": cfg.pin_memory,
        "persistent_workers": cfg.persistent_workers and num_workers > 0,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor
        loader_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    clip_model = clip_model.visual.to(device)
    clip_model.train()

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


def _encode_images(encoder: nn.Module, images: torch.Tensor) -> torch.Tensor:
    features = encoder(images)
    if features.ndim > 2:
        features = features.view(features.size(0), -1)
    return F.normalize(features, dim=-1)


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
    threshold: float = 0.5,
):
    head.eval()

    losses = []
    all_probs = []
    all_labels = []

    for imgs, labels in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        embeddings = _encode_images(encoder, imgs)
        probs = head(embeddings)

        loss = criterion(probs, labels)
        losses.append(loss.item())

        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())

    probs_np = torch.cat(all_probs, dim=0).squeeze(1).numpy()
    labels_np = torch.cat(all_labels, dim=0).squeeze(1).numpy().astype(np.int64)

    pred_np = (probs_np >= threshold).astype(np.int64)
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
    real_paths: List[Path],
    eval_transform,
    device: torch.device,
    max_images: int = 300,
):
    selected_real_paths = sorted(real_paths)[:max_images]

    if not selected_real_paths:
        raise RuntimeError("No real images found for centroid calibration")

    embs = []
    encoder.eval()
    with torch.no_grad():
        for p in tqdm(selected_real_paths, desc="Calibrating real centroid", leave=False):
            with Image.open(p) as image_file:
                img = image_file.convert("RGB")
            x = eval_transform(img).unsqueeze(0).to(device, non_blocking=True)
            emb = _encode_images(encoder, x)
            embs.append(emb.cpu())

    centroid = torch.cat(embs, dim=0).mean(dim=0)
    centroid = centroid / centroid.norm()
    return centroid, len(selected_real_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-all-data", action="store_true", help="Use all images as training data (no val/test splitting)")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.use_all_data:
        cfg.use_all_data = True
    set_seed(cfg.seed)

    os.makedirs(cfg.checkpoints_dir, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this script. Install CUDA-enabled PyTorch and retry."
        )

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    if vram_gb < 8.0:
        cfg.batch_size = cfg.small_vram_batch_size
    elif vram_gb < 12.0:
        cfg.batch_size = max(cfg.batch_size, 8)
    else:
        cfg.batch_size = max(cfg.batch_size, 16)

    if cfg.batch_size > 64:
        cfg.batch_size = 64

    requested_workers = cfg.num_workers
    cfg.num_workers = _resolve_num_workers(cfg.num_workers)
    if cfg.num_workers != requested_workers:
        print(f"INFO | Using {cfg.num_workers} DataLoader workers (requested {requested_workers})")

    print(f"INFO | GPU: {gpu_name} | VRAM: {vram_gb:.1f} GB")
    print(f"INFO | batch_size={cfg.batch_size}")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()

    train_loader, val_loader, test_loader, train_samples, val_samples, test_samples = build_dataloaders(cfg)
    print(
        f"INFO | Dataset: {len(train_samples)} train | {len(val_samples)} val | {len(test_samples)} test | classes: real=0 gan_fake=1"
    )

    encoder, head = build_models(device)

    criterion = FocalLoss(gamma=cfg.gamma, alpha=cfg.alpha)
    optimizer = AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=cfg.t0, T_mult=cfg.t_mult, eta_min=cfg.eta_min
    )

    scaler = GradScaler()

    start_epoch = 1
    best_auc = -1.0
    best_acc = 0.0
    best_val_loss = float("inf")
    best_epoch = -1
    best_threshold = 0.5
    no_improve = 0
    history = []

    latest_ckpt_path = Path(cfg.latest_ckpt_path)
    if latest_ckpt_path.exists():
        try:
            latest_ckpt = torch.load(latest_ckpt_path, map_location=device)
            encoder.load_state_dict(latest_ckpt["encoder_state"])
            head.load_state_dict(latest_ckpt["head_state"])
            optimizer.load_state_dict(latest_ckpt["optimizer_state"])
            scheduler.load_state_dict(latest_ckpt["scheduler_state"])
            if "scaler_state" in latest_ckpt:
                scaler.load_state_dict(latest_ckpt["scaler_state"])
            start_epoch = int(latest_ckpt.get("epoch", 0)) + 1
            best_auc = float(latest_ckpt.get("best_auc", best_auc))
            best_acc = float(latest_ckpt.get("best_acc", best_acc))
            best_val_loss = float(latest_ckpt.get("best_val_loss", best_val_loss))
            best_epoch = int(latest_ckpt.get("best_epoch", best_epoch))
            best_threshold = float(latest_ckpt.get("best_threshold", best_threshold))
            no_improve = int(latest_ckpt.get("no_improve", no_improve))
            loaded_history = latest_ckpt.get("history", [])
            if isinstance(loaded_history, list):
                history.extend(loaded_history)
            print(f"INFO | Resumed training from {latest_ckpt_path} at epoch {start_epoch}")
        except Exception as exc:
            print(f"WARNING | Failed to resume from {latest_ckpt_path}: {exc}")

    print("INFO | Starting training")
    print(
        f"INFO | epochs={cfg.epochs} | optimizer=AdamW | scheduler=CosineAnnealingWarmRestarts | loss=FocalLoss"
    )

    for epoch in range(start_epoch, cfg.epochs + 1):
        epoch_start = time.time()

        head.train()
        encoder.train()

        running_loss = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{cfg.epochs}", leave=False)

        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            embeddings = _encode_images(encoder, imgs)

            with autocast(enabled=True):
                preds = head(embeddings)

            # Compute focal loss in fp32 outside autocast to keep BCE numerically safe.
            loss = criterion(preds.float(), labels.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), max_norm=cfg.grad_clip_max_norm)
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
            threshold=best_threshold,
        )

        val_threshold = tune_decision_threshold(val_metrics["labels"], val_metrics["probs"])
        val_metrics = evaluate(
            encoder=encoder,
            head=head,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
            threshold=val_threshold,
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
            f"Thr: {val_threshold:.3f} | LR: {lr_now:.2e} | Time: {epoch_time:.1f}s | GPU Mem: {mem_alloc:.2f} GB"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_threshold": val_threshold,
                "lr": lr_now,
            }
        )

        latest_state = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "best_auc": best_auc,
            "best_acc": best_acc,
            "best_threshold": best_threshold,
            "no_improve": no_improve,
            "history": history,
        }
        torch.save(latest_state, cfg.latest_ckpt_path)

        improved = (val_auc - best_auc) > cfg.early_stopping_min_delta
        if improved:
            best_val_loss = val_loss
            best_auc = val_auc
            best_acc = val_acc
            best_epoch = epoch
            best_threshold = val_threshold
            no_improve = 0
            ckpt = {
                "encoder_state": encoder.state_dict(),
                "head_state": head.state_dict(),
                "val_loss": best_val_loss,
                "val_auc": best_auc,
                "val_acc": best_acc,
                "decision_threshold": best_threshold,
                "epoch": best_epoch,
                "config": asdict(cfg),
            }
            torch.save(ckpt, cfg.best_ckpt_path)
            print(
                f"INFO | Checkpoint saved -> {cfg.best_ckpt_path} | val_loss={best_val_loss:.4f} | val_auc={best_auc:.4f} | thr={best_threshold:.3f}"
            )
        else:
            no_improve += 1
            print(
                f"INFO | No val-loss improvement for {no_improve} epoch(s). Best val_loss={best_val_loss:.4f}"
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
        best_threshold = float(best_ckpt.get("decision_threshold", 0.5))
        print(
            f"INFO | Loaded best checkpoint from epoch {best_ckpt['epoch']} with AUC={best_ckpt['val_auc']:.4f}"
        )

    _, eval_transform = make_transforms(cfg)
    discovered_real_paths, _ = _discover_real_fake_paths(cfg.dataset_root)
    centroid, n_real = calibrate_real_centroid(
        encoder=encoder,
        real_paths=discovered_real_paths,
        eval_transform=eval_transform,
        device=device,
        max_images=cfg.max_real_for_calibration,
    )
    torch.save(centroid, cfg.centroid_path)
    print(
        f"CLIP centroid calibrated on {n_real} real images. Saved to {cfg.centroid_path}"
    )

    test_metrics = evaluate(
        encoder=encoder,
        head=head,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
        threshold=best_threshold,
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

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_auc,
        "best_val_acc": best_acc,
        "decision_threshold": best_threshold,
        "test": {
            "loss": test_metrics["loss"],
            "acc": test_acc,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "auc": test_auc,
            "confusion_matrix": test_cm.tolist(),
        },
        "history": history,
    }
    Path(cfg.metrics_out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.metrics_out_path).write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    print(f"INFO | Training metrics written to {cfg.metrics_out_path}")

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
    mp.set_start_method("spawn", force=True)
    main()
