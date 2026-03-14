"""
VeriSight Layer 3 — GAN Detector Training Pipeline
Team: return0 | Hack-Nocturne '26
GPU : NVIDIA RTX 4060 Notebook 8 GB (CPU fallback included)
Run : python train_gan.py
Out : checkpoints/layer3_best.pth + checkpoints/clip_real_centroid.pt
"""

import os, io, time, random, shutil, argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler as TorchGradScaler
    from torch.amp import autocast as torch_autocast
    AMP_API_NEW = True
except ImportError:
    from torch.cuda.amp import GradScaler as TorchGradScaler
    from torch.cuda.amp import autocast as torch_autocast
    AMP_API_NEW = False
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

try:
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score
)


def setup_device() -> Tuple[torch.device, bool]:
    """Returns (device, use_amp). GPU if available, CPU fallback."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        torch.cuda.empty_cache()
        print(
            f"[GPU] {props.name} | VRAM: {props.total_memory/1e9:.1f} GB | "
            f"Compute: {props.major}.{props.minor} | TF32: ON | AMP: ON"
        )
    else:
        import multiprocessing

        n_cpu = multiprocessing.cpu_count()
        torch.set_num_threads(n_cpu)
        torch.set_num_interop_threads(max(1, n_cpu // 2))
        print(f"[CPU] {n_cpu} threads | AMP: OFF | Batch size will be 16")

    return device, use_amp


class JpegAug:
    """Simulate JPEG compression artifact for training augmentation."""

    def __init__(self, lo=60, hi=95):
        """Initialize JPEG quality bounds for augmentation."""
        self.lo = lo
        self.hi = hi

    def __call__(self, pil_img):
        """Apply randomized JPEG recompression to a PIL image."""
        quality = random.randint(self.lo, self.hi)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class FocalLoss(nn.Module):
    """Focal loss gamma=2.0 alpha=0.25 for class-imbalanced GAN detection."""

    def __init__(self, gamma=2.0, alpha=0.25):
        """Configure focal loss hyperparameters."""
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss from probabilities and binary targets."""
        pred = pred.float().clamp(1e-7, 1 - 1e-7)
        target = target.float()
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1.0, pred, 1.0 - pred)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return loss.mean()


class GANHead(nn.Module):
    """MLP head on top of frozen CLIP-RN50 embeddings. Input dim 1024."""

    def __init__(self):
        """Build the classifier head for binary GAN detection."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run forward pass for embedding batch."""
        return self.net(x)


class BinaryPathDataset(Dataset):
    """Dataset backed by (path, label) tuples with guaranteed binary labels."""

    def __init__(self, samples: List[Tuple[str, int]], transform):
        """Store prefiltered binary samples and transform."""
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Load image from disk and return transformed tensor + binary label."""
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


def build_dataset(dataset_dir="dataset", max_per_class=4000) -> Tuple[int, int]:
    """Downloads Lunahera/genimagepp from HuggingFace. Falls back to synthetic if download fails."""
    real_dir = Path(dataset_dir) / "real"
    gan_dir = Path(dataset_dir) / "gan_fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    gan_dir.mkdir(parents=True, exist_ok=True)

    for p in real_dir.glob("*.jpg"):
        p.unlink(missing_ok=True)
    for p in gan_dir.glob("*.jpg"):
        p.unlink(missing_ok=True)

    try:
        from datasets import load_dataset

        ds = load_dataset(
            "Lunahera/genimagepp",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        n_real = 0
        n_gan = 0
        counter = 0

        pbar = tqdm(total=max_per_class * 2, desc="Downloading dataset")

        for record in ds:
            if n_real >= max_per_class and n_gan >= max_per_class:
                break

            img = record["image"].convert("RGB").resize((224, 224))
            if "label" in record:
                target_label = int(record["label"])
            else:
                target_label = counter % 2

            if target_label == 0:
                if n_real < max_per_class:
                    out_path = real_dir / f"r_{n_real:06d}.jpg"
                    img.save(str(out_path), "JPEG", quality=92)
                    n_real += 1
                    pbar.update(1)
            else:
                if n_gan < max_per_class:
                    out_path = gan_dir / f"g_{n_gan:06d}.jpg"
                    img.save(str(out_path), "JPEG", quality=92)
                    n_gan += 1
                    pbar.update(1)

            counter += 1
            if counter > max_per_class * 20:
                break

        pbar.close()

    except Exception as e:
        print(f"[WARN] HuggingFace download failed: {e}")
        print("[INFO] Building synthetic fallback dataset...")

        def _make_real_img() -> np.ndarray:
            """Create one pseudo-real image with natural signal and JPEG artifacts."""
            i_idx, j_idx = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")
            r = 128 + 70 * np.sin(i_idx / 28.0) + np.random.normal(0, 10, (224, 224))
            g = 128 + 55 * np.cos(j_idx / 22.0) + np.random.normal(0, 10, (224, 224))
            b = 100 + 45 * np.sin((i_idx + j_idx) / 36.0) + np.random.normal(0, 10, (224, 224))
            arr = np.stack([r, g, b], axis=2)
            arr = np.clip(arr, 0, 255).astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG", quality=88)
            buf.seek(0)
            return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)

        def _make_gan_img(real_arr: np.ndarray) -> np.ndarray:
            """Create one pseudo-GAN image by injecting synthetic artifact patterns."""
            fake = real_arr.astype(np.float64).copy()

            y0 = random.randint(0, 128)
            x0 = random.randint(0, 128)
            for i in range(96):
                for j in range(96):
                    sign = 1 if (i // 8 + j // 8) % 2 == 0 else -1
                    fake[y0 + i, x0 + j, 0] += sign * 28

            y1 = random.randint(0, 144)
            x1 = random.randint(0, 144)
            fake[y1:y1 + 80, x1:x1 + 80, 1] = fake[y1:y1 + 80, x1:x1 + 80, 0] * 0.982
            fake[y1:y1 + 80, x1:x1 + 80, 2] = fake[y1:y1 + 80, x1:x1 + 80, 0] * 0.968

            for row in range(224):
                fake[row, :, :] += np.sin(row * 0.18) * 7

            return np.clip(fake, 0, 255).astype(np.uint8)

        pbar = tqdm(total=max_per_class * 2, desc="Generating synthetic dataset")
        for i in range(max_per_class):
            real_img = _make_real_img()
            Image.fromarray(real_img).save(real_dir / f"r_{i:06d}.jpg", "JPEG", quality=92)
            pbar.update(1)

            gan_img = _make_gan_img(real_img)
            Image.fromarray(gan_img).save(gan_dir / f"g_{i:06d}.jpg", "JPEG", quality=92)
            pbar.update(1)
        pbar.close()

    n_real = len(list(real_dir.glob("*.jpg")))
    n_gan = len(list(gan_dir.glob("*.jpg")))

    print(f"[DATA] Real: {n_real} | GAN: {n_gan} | Total: {n_real + n_gan}")

    if n_real < 50 or n_gan < 50:
        raise RuntimeError(
            f"Dataset too small. Real={n_real}, GAN={n_gan}. "
            "Need at least 50 per class. Check internet or disk space."
        )

    return n_real, n_gan


def load_encoder(device: torch.device) -> nn.Module:
    """Load CLIP-RN50 visual encoder frozen. Raises ImportError if open_clip missing."""
    if not CLIP_AVAILABLE:
        raise ImportError(
            "open_clip is required. Install with: pip install open-clip-torch"
        )

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "RN50", pretrained="openai"
    )
    encoder = clip_model.visual.to(device)
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[ENC] CLIP-RN50 loaded. Params: {n_params:,} | Frozen: YES | Embed: 1024-d")
    return encoder


def make_loaders(dataset_dir: str, device: torch.device) -> Tuple[DataLoader, DataLoader, dict]:
    """Build train/val DataLoaders. Returns (train_loader, val_loader, meta_dict)."""
    is_gpu = device.type == "cuda"
    batch_size = 64 if is_gpu else 16
    n_workers = 4 if is_gpu else 0
    pin_mem = True if is_gpu else False

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.88, 1.14)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.28, contrast=0.28, saturation=0.18, hue=0.09),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.8))], p=0.3),
        transforms.RandomGrayscale(p=0.08),
        JpegAug(lo=60, hi=95),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std),
    ])

    try:
        full_ds = ImageFolder(root=dataset_dir, allow_empty=True)
    except TypeError:
        full_ds = ImageFolder(root=dataset_dir)
    print(f"[DATA] Class map: {full_ds.class_to_idx}")

    if "real" not in full_ds.class_to_idx or "gan_fake" not in full_ds.class_to_idx:
        raise RuntimeError(
            "Dataset must contain class folders: dataset/real and dataset/gan_fake"
        )

    binary_samples: List[Tuple[str, int]] = []
    for path, idx in full_ds.samples:
        cls_name = full_ds.classes[idx]
        if cls_name == "real":
            binary_samples.append((path, 0))
        elif cls_name == "gan_fake":
            binary_samples.append((path, 1))

    if len(binary_samples) < 2:
        raise RuntimeError("Not enough binary samples found under dataset/real and dataset/gan_fake")

    random.Random(42).shuffle(binary_samples)
    n_val = max(1, int(0.15 * len(binary_samples)))
    n_train = len(binary_samples) - n_val
    train_samples = binary_samples[:n_train]
    val_samples = binary_samples[n_train:]

    train_ds = BinaryPathDataset(train_samples, transform=train_tf)
    val_ds = BinaryPathDataset(val_samples, transform=val_tf)
    label_flip = False

    print(
        "[DATA] Label mapping fixed: real->0, gan_fake->1 "
        f"| usable samples: {len(binary_samples)}"
    )

    train_kwargs = dict(
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_mem,
        shuffle=True,
        drop_last=True,
    )
    val_kwargs = dict(
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=pin_mem,
        shuffle=False,
        drop_last=False,
    )

    if n_workers > 0:
        train_kwargs["persistent_workers"] = True
        train_kwargs["prefetch_factor"] = 2
        val_kwargs["persistent_workers"] = True
        val_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = DataLoader(val_ds, **val_kwargs)

    print(
        f"[DATA] Train: {n_train} imgs / {len(train_loader)} batches | "
        f"Val: {n_val} imgs / {len(val_loader)} batches | batch={batch_size}"
    )

    meta = {
        "batch_size": batch_size,
        "label_flip": label_flip,
        "n_train": n_train,
        "n_val": n_val,
    }
    return train_loader, val_loader, meta


def train(
    epochs=25,
    dataset_dir="dataset",
    save_dir="checkpoints",
    patience=7,
    early_stop=True,
) -> float:
    """Full training loop. Returns best val AUC achieved."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device, use_amp = setup_device()
    encoder = load_encoder(device)
    train_loader, val_loader, meta = make_loaders(dataset_dir, device)
    label_flip = meta["label_flip"]

    head = GANHead().to(device)
    criterion = FocalLoss(gamma=2.0, alpha=0.25).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=3e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=6,
        T_mult=2,
        eta_min=1e-6,
    )
    if AMP_API_NEW:
        scaler = TorchGradScaler("cuda", enabled=use_amp)
    else:
        scaler = TorchGradScaler(enabled=use_amp)

    try:
        import importlib

        if importlib.util.find_spec("triton") is not None:
            torch_dynamo = importlib.import_module("torch._dynamo")
            torch_dynamo.config.suppress_errors = True
            head = torch.compile(head, mode="reduce-overhead")
            print("[OPT] torch.compile: reduce-overhead mode active")
        else:
            print("[OPT] torch.compile unavailable — eager mode")
    except Exception:
        print("[OPT] torch.compile unavailable — eager mode")

    os.makedirs(save_dir, exist_ok=True)
    best_auc = 0.0
    no_improve = 0
    ckpt_path = os.path.join(save_dir, "layer3_best.pth")
    latest_ckpt_path = os.path.join(save_dir, "layer3_latest.pth")

    print(f"\n{'='*58}")
    print(f"  Training: {epochs} epochs | Device: {device}")
    print(f"{'='*58}\n")

    for epoch in range(1, epochs + 1):
        t_start = time.time()

        head.train()
        encoder.eval()
        t_loss = 0.0

        pbar = tqdm(train_loader, desc=f"E{epoch:02d}/{epochs} TRAIN", leave=False, ncols=88)
        for imgs, raw_labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = raw_labels.clone()
            if label_flip:
                labels = 1 - labels
            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            if AMP_API_NEW:
                amp_ctx = torch_autocast(device_type="cuda", enabled=use_amp)
            else:
                amp_ctx = torch_autocast(enabled=use_amp)

            with amp_ctx:
                with torch.no_grad():
                    feats = encoder(imgs)
                    feats = feats.view(feats.size(0), -1)
                    feats = F.normalize(feats, dim=-1)
                preds = head(feats)

            loss = criterion(preds.float(), labels.float())

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        pbar.close()
        avg_t_loss = t_loss / max(len(train_loader), 1)

        head.eval()
        v_preds, v_labels = [], []
        v_loss_total = 0.0

        with torch.no_grad():
            for imgs, raw_labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels_v = raw_labels.clone()
                if label_flip:
                    labels_v = 1 - labels_v
                labels_gpu = labels_v.float().to(device, non_blocking=True).view(-1, 1)

                if AMP_API_NEW:
                    amp_ctx = torch_autocast(device_type="cuda", enabled=use_amp)
                else:
                    amp_ctx = torch_autocast(enabled=use_amp)

                with amp_ctx:
                    feats = encoder(imgs)
                    feats = feats.view(feats.size(0), -1)
                    feats = F.normalize(feats, dim=-1)
                    preds = head(feats)

                v_loss_total += criterion(preds.float(), labels_gpu.float()).item()

                p_list = preds.cpu().float().squeeze().tolist()
                v_preds += ([p_list] if isinstance(p_list, float) else p_list)
                v_labels += labels_v.tolist()

        avg_v_loss = v_loss_total / max(len(val_loader), 1)
        v_bin = [1 if p > 0.5 else 0 for p in v_preds]

        try:
            val_auc = roc_auc_score(v_labels, v_preds)
        except ValueError:
            val_auc = 0.5

        val_acc = accuracy_score(v_labels, v_bin)
        val_pre = precision_score(v_labels, v_bin, zero_division=0)
        val_rec = recall_score(v_labels, v_bin, zero_division=0)
        val_f1 = f1_score(v_labels, v_bin, zero_division=0)

        scheduler.step(epoch)
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t_start

        vram_str = ""
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(0) / 1e9
            resv = torch.cuda.memory_reserved(0) / 1e9
            vram_str = f" | VRAM {alloc:.1f}/{resv:.1f}GB"

        print(
            f"[{epoch:02d}/{epochs}] "
            f"TLoss={avg_t_loss:.4f} VLoss={avg_v_loss:.4f} "
            f"AUC={val_auc:.4f} Acc={val_acc:.4f} "
            f"F1={val_f1:.4f} Pre={val_pre:.4f} Rec={val_rec:.4f} "
            f"LR={cur_lr:.1e} t={elapsed:.0f}s{vram_str}"
        )

        epoch_ckpt = {
            "epoch": epoch,
            "encoder_state": encoder.state_dict(),
            "head_state": head.state_dict(),
            "val_auc": val_auc,
            "val_acc": val_acc,
        }
        epoch_ckpt_path = os.path.join(save_dir, f"layer3_epoch_{epoch:03d}.pth")
        torch.save(epoch_ckpt, epoch_ckpt_path)
        torch.save(epoch_ckpt, latest_ckpt_path)
        print(f"  • Epoch checkpoint: {epoch_ckpt_path}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save(epoch_ckpt, ckpt_path)
            print(f"  ✓ Saved  {ckpt_path}  AUC={val_auc:.4f}  F1={val_f1:.4f}")
        else:
            no_improve += 1
            if early_stop:
                print(f"  – No improve ({no_improve}/{patience})")
            else:
                print(f"  – No improve ({no_improve})")
            if early_stop and no_improve >= patience:
                print(f"  Early stop. Best AUC: {best_auc:.4f}")
                break

    print(f"\n{'='*58}")
    print(f"  DONE. Best Val AUC: {best_auc:.4f}")
    print(f"{'='*58}\n")
    return best_auc


def calibrate(dataset_dir="dataset", save_dir="checkpoints", n=300) -> str:
    """Compute real-image centroid in CLIP embedding space. Saves to checkpoints/."""
    device, use_amp = setup_device()
    encoder = load_encoder(device)

    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(clip_mean, clip_std),
    ])

    real_paths = sorted(Path(dataset_dir, "real").glob("*.jpg"))
    random.seed(42)
    random.shuffle(real_paths)
    real_paths = real_paths[:min(n, len(real_paths))]

    if len(real_paths) == 0:
        raise FileNotFoundError(
            "No real images found in dataset/real/. "
            "Run build_dataset() before calibrate()."
        )

    embeddings = []
    for i in range(0, len(real_paths), 32):
        batch_paths = real_paths[i:i + 32]
        tensors = []
        for p in batch_paths:
            try:
                img = Image.open(str(p)).convert("RGB")
                tensors.append(tf(img))
            except Exception:
                continue
        if not tensors:
            continue

        batch = torch.stack(tensors).to(device, non_blocking=True)
        with torch.no_grad():
            if AMP_API_NEW:
                amp_ctx = torch_autocast(device_type="cuda", enabled=use_amp)
            else:
                amp_ctx = torch_autocast(enabled=use_amp)

            with amp_ctx:
                feats = encoder(batch)
                feats = feats.view(feats.size(0), -1)
                feats = F.normalize(feats, dim=-1)
        embeddings.append(feats.cpu())

    if not embeddings:
        raise RuntimeError("No embeddings computed. Check dataset/real/ folder.")

    all_emb = torch.cat(embeddings, dim=0)
    centroid = all_emb.mean(dim=0)
    centroid = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "clip_real_centroid.pt")
    torch.save(centroid, out_path)
    print(f"[CAL] Centroid from {len(all_emb)} real imgs → {out_path}")
    print(f"[CAL] Shape: {centroid.shape} | L2 norm: {centroid.norm().item():.6f}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Layer 3 GAN detector")
    parser.add_argument("--dataset-dir", default="dataset", help="Dataset root directory")
    parser.add_argument("--save-dir", default="checkpoints", help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--no-early-stop", action="store_true", help="Disable early stopping and run all epochs")
    parser.add_argument("--max-per-class", type=int, default=4000, help="Max images per class to build")
    parser.add_argument("--calib-n", type=int, default=300, help="Number of real images for centroid calibration")
    parser.add_argument("--smoke", action="store_true", help="Run tiny smoke pipeline (1 epoch, 64/class)")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.max_per_class = min(args.max_per_class, 64)
        args.calib_n = min(args.calib_n, 32)

    print("\n" + "=" * 58)
    print("  VeriSight Layer 3 — Training Pipeline")
    print("  Team: return0 | Hack-Nocturne '26")
    print("=" * 58 + "\n")

    t0 = time.time()

    print("[1/3] Building dataset...")
    n_real, n_gan = build_dataset(dataset_dir=args.dataset_dir, max_per_class=args.max_per_class)

    print(f"\n[2/3] Training ({args.epochs} epochs)...")
    best_auc = train(
        epochs=args.epochs,
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
        patience=args.patience,
        early_stop=not args.no_early_stop,
    )

    print("\n[3/3] Calibrating CLIP centroid...")
    calibrate(dataset_dir=args.dataset_dir, save_dir=args.save_dir, n=args.calib_n)

    total_min = (time.time() - t0) / 60
    print(f"\n{'='*58}")
    print(f"  Pipeline done in {total_min:.1f} minutes")
    print(f"  Best AUC : {best_auc:.4f}")
    print(f"  Target   : 0.92")
    print(f"  Status   : {'ACHIEVED ✓' if best_auc >= 0.92 else 'BELOW TARGET — run more epochs'}")
    print(f"{'='*58}\n")
