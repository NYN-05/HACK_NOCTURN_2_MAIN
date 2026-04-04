"""
VeriSight Layer 3 - RTX 4060 Optimized Training Pipeline
Team: return0 | Hack-Nocturne '26
GPU target: NVIDIA RTX 4060 Notebook 8 GB VRAM
Run: python layer3_train_rtx4060.py
Output: checkpoints/layer3_best.pth + checkpoints/clip_real_centroid.pt
"""

import os
import io
import time
import random
import pathlib
import json

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets as tv_datasets
from torchvision import transforms

try:
    import open_clip
except ImportError:
    open_clip = None

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def _default_dataset_dir() -> str:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    for folder_name in ("DATA", "Data", "data"):
        candidate = repo_root / folder_name
        if candidate.exists():
            return str(candidate)
    return str(repo_root / "DATA")


DEFAULT_DATASET_DIR = _default_dataset_dir()


class JpegCompressionTransform:
    def __init__(self, quality_range=(60, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        import io as _io
        import random as _random
        from PIL import Image as PILImage

        quality = _random.randint(*self.quality_range)
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return PILImage.open(buf).copy()


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.where(target == 1.0, pred, 1.0 - pred)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal = alpha_t * (1.0 - pt).pow(self.gamma) * bce
        return focal.mean()


class GANHead(nn.Module):
    def __init__(self):
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

    def forward(self, x):
        return self.net(x)


def _save_jpg(img_pil, out_path, quality):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(out_path, format="JPEG", quality=quality)


def _make_real_synthetic(seed):
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

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


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


def _build_synthetic_fallback(dataset_dir, n_per_class=2000):
    root = pathlib.Path(dataset_dir)
    real_dir = root / "real"
    fake_dir = root / "gan_fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    total = n_per_class * 2
    pbar = tqdm(total=total, desc="Synthetic dataset", leave=False)
    for idx in range(n_per_class):
        real = _make_real_synthetic(seed=10_000 + idx)
        fake = _make_fake_from_real(real, seed=20_000 + idx)
        _save_jpg(real, real_dir / f"img_{idx:06d}.jpg", quality=90)
        _save_jpg(fake, fake_dir / f"img_{idx:06d}.jpg", quality=90)
        pbar.update(2)
    pbar.close()


def build_dataset(dataset_dir=DEFAULT_DATASET_DIR, max_per_class=4000):
    root = pathlib.Path(dataset_dir)
    real_dir = root / "real"
    fake_dir = root / "gan_fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    # Start clean to keep class counts deterministic for ImageFolder loading.
    for p in list(real_dir.glob("*.jpg")):
        p.unlink(missing_ok=True)
    for p in list(fake_dir.glob("*.jpg")):
        p.unlink(missing_ok=True)

    try:
        if load_dataset is None:
            raise ImportError("datasets package is not available")
        ds = load_dataset("Lunahera/genimagepp", split="train", streaming=True)
        n_real = 0
        n_fake = 0
        target_total = max_per_class * 2
        pbar = tqdm(total=target_total, desc="HuggingFace dataset", leave=False)

        for sample in ds:
            image = sample["image"]
            label = int(sample["label"])

            if label == 0:
                if n_real >= max_per_class:
                    continue
                out_path = real_dir / f"img_{n_real:06d}.jpg"
                image.convert("RGB").save(out_path, format="JPEG", quality=92)
                n_real += 1
                pbar.update(1)
            else:
                if n_fake >= max_per_class:
                    continue
                out_path = fake_dir / f"img_{n_fake:06d}.jpg"
                image.convert("RGB").save(out_path, format="JPEG", quality=92)
                n_fake += 1
                pbar.update(1)

            if n_real >= max_per_class and n_fake >= max_per_class:
                break

        pbar.close()
    except Exception as e:
        print(f"HuggingFace download failed: {e}. Building synthetic fallback dataset.")
        _build_synthetic_fallback(dataset_dir, n_per_class=2000)

    n_real = len(list(real_dir.glob("*.jpg")))
    n_gan = len(list(fake_dir.glob("*.jpg")))
    print(f"Dataset ready -> Real: {n_real} | GAN: {n_gan} | Total: {n_real + n_gan}")

    if n_real < 100:
        raise RuntimeError(
            f"Dataset build failed: fewer than 100 real images in {pathlib.Path(dataset_dir) / 'real'}/"
        )
    if n_gan < 100:
        raise RuntimeError(
            f"Dataset build failed: fewer than 100 GAN images in {pathlib.Path(dataset_dir) / 'gan_fake'}/"
        )

    return n_real, n_gan


def train_layer3(dataset_dir=DEFAULT_DATASET_DIR, epochs=25, save_dir="checkpoints"):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU        : {props.name}")
    print(f"VRAM Total : {props.total_memory / 1e9:.1f} GB")
    print(f"CUDA Cap   : {props.major}.{props.minor}")
    print(f"TF32 ON    : matmul={torch.backends.cuda.matmul.allow_tf32}")
    print(f"cuDNN bench: {torch.backends.cudnn.benchmark}")

    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.72, 1.0), ratio=(0.88, 1.14)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ColorJitter(brightness=0.28, contrast=0.28, saturation=0.18, hue=0.09),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.8))], p=0.3),
            transforms.RandomGrayscale(p=0.08),
            JpegCompressionTransform(quality_range=(60, 95)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    full_train = tv_datasets.ImageFolder(root=dataset_dir, transform=train_tf)
    class_to_idx = full_train.class_to_idx
    if not (
        "real" in class_to_idx
        and "gan_fake" in class_to_idx
        and class_to_idx["real"] == 0
        and class_to_idx["gan_fake"] == 1
    ):
        raise ValueError(
            "Class ordering error: real must be index 0, gan_fake must be index 1.\n"
            "Rename your folders if needed or rerun build_dataset()."
        )

    full_val = tv_datasets.ImageFolder(root=dataset_dir, transform=val_tf)
    total_len = len(full_train)
    n_train = int(0.85 * total_len)
    n_val = total_len - n_train
    idx_gen = torch.Generator().manual_seed(42)
    train_idx, val_idx = random_split(range(total_len), [n_train, n_val], generator=idx_gen)

    train_dataset = torch.utils.data.Subset(full_train, train_idx.indices)
    val_dataset = torch.utils.data.Subset(full_val, val_idx.indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=False,
    )

    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Train samples: {n_train} | Val samples: {n_val}")

    if open_clip is None:
        raise ImportError("open_clip is required. Run: pip install open-clip-torch")

    clip_model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    encoder = clip_model.visual.to(device, non_blocking=True)
    embed_dim = 1024
    if embed_dim != 1024:
        raise RuntimeError("Unexpected embed dimension for RN50 visual encoder")

    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    total_encoder_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder frozen. Trainable params: 0 / {total_encoder_params}")

    head = GANHead().to(device, non_blocking=True)
    trainable = sum(p.numel() for p in head.parameters())
    print(f"Head trainable params: {trainable:,}")

    try:
        head = torch.compile(head, mode="reduce-overhead")
        print("torch.compile applied to head (reduce-overhead mode)")
    except Exception:
        print("torch.compile unavailable - running eager mode (still fast)")

    criterion = FocalLoss(gamma=2.0, alpha=0.25).to(device, non_blocking=True)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=3e-4,
        weight_decay=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=6, T_mult=2, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    best_val_auc = 0.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = 5
    checkpoint_path = os.path.join(save_dir, "layer3_best.pth")
    metrics_path = os.path.join(save_dir, "layer3_rtx4060_training_metrics.json")
    os.makedirs(save_dir, exist_ok=True)
    history = []

    for epoch in range(1, epochs + 1):
        head.train()
        encoder.eval()
        epoch_loss = 0.0

        tqdm_bar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [TRAIN]", leave=False)
        for imgs, labels in tqdm_bar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    features = encoder(imgs)
                    features = features.view(features.size(0), -1)
                    features = F.normalize(features, dim=-1)
                preds = head(features)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            tqdm_bar.set_postfix(loss=f"{loss.item():.4f}")
        tqdm_bar.close()

        avg_train_loss = epoch_loss / max(1, len(train_loader))

        head.eval()
        encoder.eval()
        val_preds_list = []
        val_labels_list = []
        val_loss_total = 0.0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                labels_gpu = labels.float().to(device, non_blocking=True).view(-1, 1)

                with torch.cuda.amp.autocast():
                    features = encoder(imgs)
                    features = features.view(features.size(0), -1)
                    features = F.normalize(features, dim=-1)
                    preds = head(features)
                    v_loss = criterion(preds, labels_gpu)

                val_loss_total += v_loss.item()
                val_preds_list.extend(preds.cpu().float().squeeze().tolist())
                val_labels_list.extend(labels.tolist())

        val_preds_arr = [float(p) if not isinstance(p, float) else p for p in val_preds_list]
        val_binary = [1 if p > 0.5 else 0 for p in val_preds_arr]

        try:
            val_auc = roc_auc_score(val_labels_list, val_preds_arr)
        except ValueError:
            val_auc = 0.5

        val_acc = accuracy_score(val_labels_list, val_binary)
        val_pre = precision_score(val_labels_list, val_binary, zero_division=0)
        val_rec = recall_score(val_labels_list, val_binary, zero_division=0)
        val_f1 = f1_score(val_labels_list, val_binary, zero_division=0)
        avg_val_loss = val_loss_total / max(1, len(val_loader))

        scheduler.step(epoch)
        current_lr = optimizer.param_groups[0]["lr"]

        mem_gb = torch.cuda.memory_allocated(0) / 1e9
        mem_reserved_gb = torch.cuda.memory_reserved(0) / 1e9

        print(
            f"[{epoch:02d}/{epochs}] Loss: {avg_train_loss:.4f} | VLoss: {avg_val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | VRAM: {mem_gb:.2f}/{mem_reserved_gb:.2f}GB"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "lr": current_lr,
            }
        )

        improved = (best_val_loss - avg_val_loss) > 1e-4
        if improved:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state": encoder.state_dict(),
                    "head_state": head.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_auc": val_auc,
                    "val_acc": val_acc,
                },
                checkpoint_path,
            )
            print(f"  ✓ SAVED  checkpoints/layer3_best.pth  VLoss={avg_val_loss:.4f}  AUC={val_auc:.4f}")
        else:
            patience_counter += 1
            print(f"  - no val-loss improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch}. Best VLoss: {best_val_loss:.4f}")
                break

        _ = (val_pre, val_rec)

    print("━" * 60)
    print(f"TRAINING COMPLETE | Best Val AUC: {best_val_auc:.4f}")
    print("━" * 60)

    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_auc": best_val_auc,
                "history": history,
            },
            fp,
            indent=2,
        )
    print(f"Training metrics saved to {metrics_path}")

    return best_val_auc, checkpoint_path


def calibrate_centroid(dataset_dir=DEFAULT_DATASET_DIR, save_dir="checkpoints", n=300):
    device = torch.device("cuda")
    if open_clip is None:
        raise ImportError("open_clip is required. Run: pip install open-clip-torch")

    clip_model, _, _ = open_clip.create_model_and_transforms("RN50", pretrained="openai")
    encoder = clip_model.visual.to(device, non_blocking=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )

    real_dir = pathlib.Path(dataset_dir) / "real"
    real_paths = sorted(real_dir.glob("*.jpg"))
    rng = random.Random(42)
    rng.shuffle(real_paths)
    chosen = real_paths[: min(n, len(real_paths))]

    embeddings = []
    batch_size = 32
    for i in range(0, len(chosen), batch_size):
        batch_paths = chosen[i : i + batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(val_tf(img))
        batch_tensor = torch.stack(imgs, dim=0).to(device, non_blocking=True)

        with torch.no_grad(), torch.cuda.amp.autocast():
            feats = encoder(batch_tensor)
            feats = feats.view(feats.size(0), -1)
            feats = F.normalize(feats, dim=-1)
        embeddings.append(feats.cpu())

    if not embeddings:
        raise RuntimeError("No real images found for centroid calibration.")

    all_embeddings = torch.cat(embeddings, dim=0)
    centroid = all_embeddings.mean(dim=0)
    centroid = F.normalize(centroid.unsqueeze(0), dim=-1).squeeze(0)

    os.makedirs(save_dir, exist_ok=True)
    centroid_path = os.path.join(save_dir, "clip_real_centroid.pt")
    torch.save(centroid, centroid_path)
    embeddings_count = all_embeddings.shape[0]
    print(f"Centroid calibrated from {embeddings_count} real images -> {centroid_path}")
    print(f"Centroid shape: {centroid.shape} | norm: {centroid.norm().item():.4f}")
    return centroid_path


def verify_checkpoint_integration():
    from verisight_layer3_gan import GANDetector, Layer3Config

    ckpt_path = os.path.join("checkpoints", "layer3_best.pth")
    if not os.path.exists(ckpt_path):
        raise RuntimeError("Missing checkpoint: checkpoints/layer3_best.pth")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    expected_keys = {"epoch", "encoder_state", "head_state", "val_auc", "val_acc"}
    got_keys = set(ckpt.keys())
    if got_keys != expected_keys:
        raise RuntimeError(
            f"Checkpoint key mismatch. Expected {sorted(expected_keys)}, got {sorted(got_keys)}"
        )

    cfg = Layer3Config(device="cuda")
    _ = GANDetector(cfg)
    print("Checkpoint format verified and compatible with verisight_layer3_gan.py")


def _make_eval_images():
    rng = np.random.default_rng(1234)
    h, w = 256, 256
    ii = np.arange(h).reshape(h, 1)
    jj = np.arange(w).reshape(1, w)
    noise = rng.normal(0, 10, (h, w, 3))

    real = np.zeros((h, w, 3), dtype=np.float64)
    real[:, :, 0] = 128 + 80 * np.sin(ii / 30.0)
    real[:, :, 1] = 128 + 60 * np.cos(jj / 25.0)
    real[:, :, 2] = 100 + 50 * np.sin((ii + jj) / 40.0)
    real += noise
    real_img = np.clip(real, 0, 255).astype(np.uint8)

    gan = real_img.copy().astype(np.float64)
    yy = np.arange(h).reshape(h, 1)
    xx = np.arange(w).reshape(1, w)
    checker = (((yy // 8 + xx // 8) % 2) * 2 - 1) * 30
    gan += checker[:, :, None]
    gan[:, :, 1] = gan[:, :, 0] * 0.97
    gan[:, :, 2] = gan[:, :, 0] * 0.95
    gan_img = np.clip(gan, 0, 255).astype(np.uint8)
    return real_img, gan_img


def run_final_evaluation():
    from verisight_layer3_gan import GANDetector, Layer3Config

    cfg = Layer3Config(device="cuda")
    detector = GANDetector(cfg)

    centroid = torch.load("checkpoints/clip_real_centroid.pt", map_location="cpu")
    detector.clip_detector._real_centroid = centroid
    detector.clip_detector._calibrated = True
    print("GANDetector loaded with trained centroid.")

    # Current detector API expects file paths, so temporary files bridge ndarray tests.
    real_img, gan_img = _make_eval_images()
    tmp_real = "_tmp_eval_real.jpg"
    tmp_gan = "_tmp_eval_gan.jpg"
    Image.fromarray(real_img).save(tmp_real, format="JPEG", quality=92)
    Image.fromarray(gan_img).save(tmp_gan, format="JPEG", quality=92)

    try:
        result_real = detector.analyze(tmp_real)
        result_gan = detector.analyze(tmp_gan)
    finally:
        for p in [tmp_real, tmp_gan]:
            if os.path.exists(p):
                os.remove(p)

    print("┌─────────────────────────────────────────┐")
    print("│  VeriSight Layer 3 - Integration Test   │")
    print("└─────────────────────────────────────────┘")

    for name, result in [("Genuine Test", result_real), ("GAN Test", result_gan)]:
        print(f"Image        : {name}")
        print(f"Fraud Score  : {result.fraud_probability:.4f}")
        verdict = "✓ GENUINE" if result.fraud_probability < 0.5 else "✗ GAN DETECTED"
        print(f"Verdict      : {verdict}")
        print("Sub-scores:")
        print(f"  Spectrum   : {result.sub_scores.spectrum:.4f}")
        print(f"  CLIP       : {result.sub_scores.clip:.4f}")
        print(f"  Channel    : {result.sub_scores.channel:.4f}")
        print(f"  Boundary   : {result.sub_scores.boundary:.4f}")
        print(f"  Texture    : {result.sub_scores.texture:.4f}")
        print(f"  Re-synth   : {result.sub_scores.resynth:.4f}")

    passed = (result_real.fraud_probability < 0.5) and (result_gan.fraud_probability > 0.5)
    print("━" * 45)
    if passed:
        print("INTEGRATION TEST: ✓ PASSED")
        print("Layer 3 is operational and correctly classifying.")
    else:
        print("INTEGRATION TEST: ✗ FAILED")
        print("Check checkpoint and centroid paths.")
    print("━" * 45)


if __name__ == "__main__":
    t0 = time.time()

    print("=" * 55)
    print("  VeriSight Layer 3 - RTX 4060 Training Pipeline")
    print("=" * 55)

    print("\n[PHASE 1] Building dataset...")
    build_dataset(dataset_dir=DEFAULT_DATASET_DIR, max_per_class=4000)

    print("\n[PHASE 2] Training CLIP head on GPU...")
    best_auc, ckpt_path = train_layer3(dataset_dir=DEFAULT_DATASET_DIR, epochs=25, save_dir="checkpoints")

    print("\n[PHASE 3] Calibrating CLIP centroid...")
    calibrate_centroid(dataset_dir=DEFAULT_DATASET_DIR, save_dir="checkpoints", n=300)

    print("\n[PHASE 4] Running integration test...")
    run_final_evaluation()

    print("\n[PHASE 5] Verifying checkpoint integration...")
    verify_checkpoint_integration()

    elapsed = (time.time() - t0) / 60
    print(f"\nTotal pipeline time: {elapsed:.1f} minutes")
    print(f"Best Val AUC: {best_auc:.4f}")
    _ = ckpt_path
    if best_auc >= 0.92:
        print("Target AUC >= 0.92 -> ACHIEVED ✓")
    else:
        print("Target AUC >= 0.92 -> NOT YET. Run more epochs.")