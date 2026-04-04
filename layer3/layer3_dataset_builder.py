import io
import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter
from datasets import load_dataset
from tqdm import tqdm


def _default_dataset_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    for folder_name in ("DATA", "Data", "data"):
        candidate = repo_root / folder_name
        if candidate.exists():
            return candidate
    return repo_root / "DATA"


DATASET_ROOT = _default_dataset_root()
REAL_DIR = DATASET_ROOT / "real"
GAN_DIR = DATASET_ROOT / "gan_fake"


def ensure_dirs() -> None:
    REAL_DIR.mkdir(parents=True, exist_ok=True)
    GAN_DIR.mkdir(parents=True, exist_ok=True)


def save_jpeg(img: Image.Image, out_path: Path, quality: int = 90) -> None:
    img = img.convert("RGB")
    img.save(out_path, format="JPEG", quality=quality)


def _to_pil(image_obj) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, np.ndarray):
        arr = image_obj.astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    if hasattr(image_obj, "convert"):
        return image_obj.convert("RGB")
    raise TypeError(f"Unsupported image object type: {type(image_obj)}")


def infer_binary_label(sample: dict) -> int:
    label = sample.get("label", None)
    if isinstance(label, (int, np.integer)):
        return int(0 if int(label) == 0 else 1)

    if isinstance(label, str):
        low = label.strip().lower()
        if low in {"real", "authentic", "genuine", "0"}:
            return 0
        return 1

    for key in ("source", "generator", "prompt", "class", "category"):
        value = sample.get(key)
        if isinstance(value, str):
            low = value.lower()
            if "real" in low or "auth" in low:
                return 0
            if any(k in low for k in ["gan", "stable", "diffusion", "dalle", "midjourney", "synthetic", "ai"]):
                return 1

    return 1


def blend_horizontal_seam(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    h, w, _ = img1.shape
    seam = random.randint(h // 3, (2 * h) // 3)
    out = img1.copy().astype(np.float32)
    out[seam:, :, :] = img2[seam:, :, :].astype(np.float32)

    blend_h = 10
    y1 = max(0, seam - blend_h)
    y2 = min(h, seam + blend_h)
    if y2 > y1:
        alpha = np.linspace(0, 1, y2 - y1, dtype=np.float32).reshape(-1, 1, 1)
        out[y1:y2] = img1[y1:y2] * (1 - alpha) + img2[y1:y2] * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def add_checkerboard_patch(arr: np.ndarray, patch_size: int = 80, amplitude: float = 25.0) -> np.ndarray:
    h, w, _ = arr.shape
    if h < patch_size + 2 or w < patch_size + 2:
        return arr

    y = random.randint(0, h - patch_size)
    x = random.randint(0, w - patch_size)

    tile = 8
    grid = np.indices((patch_size, patch_size)).sum(axis=0)
    checker = ((grid // tile) % 2).astype(np.float32)
    checker = (checker * 2.0 - 1.0) * amplitude

    out = arr.astype(np.float32)
    out[y : y + patch_size, x : x + patch_size, 0] += checker
    out[y : y + patch_size, x : x + patch_size, 1] += checker
    out[y : y + patch_size, x : x + patch_size, 2] += checker

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_region_channel_collapse(arr: np.ndarray, patch_size: int = 100) -> np.ndarray:
    h, w, _ = arr.shape
    if h < patch_size + 2 or w < patch_size + 2:
        return arr

    y = random.randint(0, h - patch_size)
    x = random.randint(0, w - patch_size)

    out = arr.astype(np.float32)
    region = out[y : y + patch_size, x : x + patch_size]
    r = region[..., 0]
    region[..., 1] = np.clip(r * 0.98, 0, 255)
    region[..., 2] = np.clip(r * 0.97, 0, 255)

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gan_like_smoothing(arr: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(arr)
    tmp = io.BytesIO()
    pil.save(tmp, format="JPEG", quality=random.randint(60, 95))
    tmp.seek(0)
    arr2 = np.array(Image.open(tmp).convert("RGB"))
    return np.array(Image.fromarray(arr2).filter(ImageFilter.GaussianBlur(radius=0.5)))


def generate_synthetic_augmentations(n: int = 500) -> None:
    real_paths = [p for p in REAL_DIR.glob("*.jpg")]
    if not real_paths:
        return

    for i in tqdm(range(n), desc="Synthetic GAN augmentations"):
        src = random.choice(real_paths)
        arr = np.array(Image.open(src).convert("RGB"))

        arr = add_checkerboard_patch(arr, patch_size=80, amplitude=25)
        arr = apply_region_channel_collapse(arr, patch_size=100)
        arr = apply_gan_like_smoothing(arr)

        out = GAN_DIR / f"aug_gan_{i:05d}.jpg"
        save_jpeg(Image.fromarray(arr), out, quality=90)


def generate_secondary_blends(n: int = 300) -> None:
    real_paths = [p for p in REAL_DIR.glob("*.jpg")]
    fake_paths = [p for p in GAN_DIR.glob("*.jpg")]
    if not real_paths or not fake_paths:
        return

    for i in tqdm(range(n), desc="Secondary seam blends"):
        a = np.array(Image.open(random.choice(real_paths)).convert("RGB").resize((224, 224)))
        b = np.array(Image.open(random.choice(fake_paths)).convert("RGB").resize((224, 224)))

        blend = blend_horizontal_seam(a, b)
        blend = add_checkerboard_patch(blend, patch_size=60, amplitude=18)
        out = GAN_DIR / f"blend_gan_{i:05d}.jpg"
        save_jpeg(Image.fromarray(blend), out, quality=90)


def fallback_synthetic_generation() -> Tuple[int, int]:
    real_n = 2000
    fake_n = 2000

    for i in tqdm(range(real_n), desc="Fallback real synth"):
        h, w = 224, 224
        x = np.linspace(0, 1, w, dtype=np.float32)
        y = np.linspace(0, 1, h, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)

        base = np.zeros((h, w, 3), dtype=np.float32)
        base[..., 0] = 0.4 + 0.4 * xv
        base[..., 1] = 0.35 + 0.45 * yv
        base[..., 2] = 0.3 + 0.5 * (1.0 - xv)
        noise = np.random.normal(0, 0.02, size=(h, w, 3)).astype(np.float32)
        arr = np.clip((base + noise) * 255.0, 0, 255).astype(np.uint8)

        save_jpeg(Image.fromarray(arr), REAL_DIR / f"fallback_real_{i:05d}.jpg", quality=90)

    for i in tqdm(range(fake_n), desc="Fallback gan synth"):
        h, w = 224, 224
        arr = np.random.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
        arr = add_checkerboard_patch(arr, patch_size=80, amplitude=35)
        arr = apply_region_channel_collapse(arr, patch_size=100)
        save_jpeg(Image.fromarray(arr), GAN_DIR / f"fallback_gan_{i:05d}.jpg", quality=90)

    return real_n, fake_n


def build_from_huggingface(max_per_class: int = 4000) -> Tuple[int, int]:
    real_count = 0
    fake_count = 0

    ds = load_dataset("Lunahera/genimagepp", split="train", streaming=True)

    progress = tqdm(total=max_per_class * 2, desc="Streaming GenImage++")
    idx_real = 0
    idx_fake = 0

    for sample in ds:
        try:
            label = infer_binary_label(sample)
            image = _to_pil(sample["image"])
            if label == 0 and real_count < max_per_class:
                save_jpeg(image, REAL_DIR / f"real_{idx_real:05d}.jpg", quality=90)
                idx_real += 1
                real_count += 1
                progress.update(1)
            elif label == 1 and fake_count < max_per_class:
                save_jpeg(image, GAN_DIR / f"gan_{idx_fake:05d}.jpg", quality=90)
                idx_fake += 1
                fake_count += 1
                progress.update(1)

            if real_count >= max_per_class and fake_count >= max_per_class:
                break
        except Exception:
            continue

    progress.close()
    return real_count, fake_count


def class_counts() -> Tuple[int, int]:
    real = len(list(REAL_DIR.glob("*.jpg")))
    fake = len(list(GAN_DIR.glob("*.jpg")))
    return real, fake


def main() -> None:
    ensure_dirs()

    try:
        real_count, fake_count = build_from_huggingface(max_per_class=4000)
        print(f"Dataset ready: {real_count} real images, {fake_count} gan_fake images")

        generate_synthetic_augmentations(n=500)
        generate_secondary_blends(n=300)
    except Exception as exc:
        print(f"HuggingFace download failed: {exc}")
        print("Falling back to synthetic dataset generation...")
        fallback_synthetic_generation()

    real_total, fake_total = class_counts()
    ratio = (min(real_total, fake_total) / max(real_total, fake_total)) if max(real_total, fake_total) else 0.0

    print(f"Total class counts -> real: {real_total}, gan_fake: {fake_total}")
    print(f"Class balance ratio: {ratio:.3f}")


if __name__ == "__main__":
    main()
