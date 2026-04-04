import argparse
import logging
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms

from engine.data.manifest_utils import LabeledImage, discover_labeled_images, split_labeled_images
from utils.config import (
    CIFAKE_DIR,
    IMAGENET_MINI_DIR,
    LABEL_TO_ID,
    PROCESSED_DATASET_DIR,
    RANDOM_SEED,
)

LOGGER = logging.getLogger(__name__)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SPLIT_VARIANTS = {
    "train": ("train", "TRAIN"),
    "val": ("val", "VAL", "validation", "VALIDATION"),
    "test": ("test", "TEST"),
}

CLASS_VARIANTS = {
    "real": ("real", "REAL"),
    "fake": ("fake", "FAKE", "ai_generated", "AI_GENERATED"),
}

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
IGNORE_TOKENS = {"mask", "groundtruth", "ground_truth", "gt"}


class VeriSightImageDataset(Dataset):
    def __init__(self, image_paths: Sequence[Path], labels: Sequence[int], transform=None):
        if len(image_paths) != len(labels):
            raise ValueError("image_paths and labels must have the same length")
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "labels": label,
        }


def _collect_images(path: Path) -> List[Path]:
    if not path.exists():
        return []
    files = []
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            files.append(file_path)
    return files


def _find_existing_dir(base_dir: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = base_dir / name
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _resolve_cifake_class_dirs(root: Path, split_name: str) -> Tuple[Path | None, Path | None]:
    split_dir = _find_existing_dir(root, SPLIT_VARIANTS[split_name])
    if split_dir is None:
        return None, None

    real_dir = _find_existing_dir(split_dir, CLASS_VARIANTS["real"])
    fake_dir = _find_existing_dir(split_dir, CLASS_VARIANTS["fake"])
    return real_dir, fake_dir


def _extract_zip_archives(dataset_root: Path) -> Path:
    extracted_root = dataset_root / "_extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(dataset_root.glob("*.zip"))
    for zip_file in zip_files:
        target_dir = extracted_root / zip_file.stem
        marker = target_dir / ".done"
        if marker.exists():
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Extracting %s -> %s", zip_file, target_dir)
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        marker.write_text("ok", encoding="utf-8")

    return extracted_root


def _find_cifake_root(search_roots: Sequence[Path]) -> Path | None:
    def is_cifake_layout(root: Path) -> bool:
        train_real, train_fake = _resolve_cifake_class_dirs(root, "train")
        test_real, test_fake = _resolve_cifake_class_dirs(root, "test")
        return all([train_real, train_fake, test_real, test_fake])

    for root in search_roots:
        if not root.exists():
            continue
        if is_cifake_layout(root):
            return root
        for train_real in root.rglob("train/REAL"):
            candidate = train_real.parent.parent
            if is_cifake_layout(candidate):
                return candidate
        for train_real in root.rglob("train/real"):
            candidate = train_real.parent.parent
            if is_cifake_layout(candidate):
                return candidate
    return None


def _find_imagenet_root(search_roots: Sequence[Path]) -> Path | None:
    for root in search_roots:
        if not root.exists():
            continue

        for candidate in root.rglob("*"):
            if not candidate.is_dir():
                continue
            if "imagenet-mini" in candidate.name.lower() and _collect_images(candidate):
                return candidate

    return None


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


def _discover_labeled_images(dataset_root: Path) -> Tuple[List[Path], List[Path]]:
    real_images: List[Path] = []
    fake_images: List[Path] = []

    # Skip already-processed split folders to avoid circular reuse.
    excluded_names = {"train", "val", "test", "_extracted", "prepared_yolo", "labels"}

    for image_path in _collect_images(dataset_root):
        if any(part.lower() in excluded_names for part in image_path.parts):
            continue

        label = _label_from_path(image_path)
        if label == 0:
            real_images.append(image_path)
        elif label == 1:
            fake_images.append(image_path)

    return real_images, fake_images


def _has_processed_dataset(output_root: Path) -> bool:
    for split in ("train", "val", "test"):
        split_dir = _find_existing_dir(output_root, SPLIT_VARIANTS[split])
        if split_dir is None:
            return False

        real_dir = _find_existing_dir(split_dir, CLASS_VARIANTS["real"])
        fake_dir = _find_existing_dir(split_dir, CLASS_VARIANTS["fake"])
        if real_dir is None or fake_dir is None:
            return False

        if not _collect_images(real_dir) or not _collect_images(fake_dir):
            return False

    return True


def _safe_copy(source_file: Path, destination_file: Path) -> bool:
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    if destination_file.exists():
        return True
    try:
        shutil.copy2(source_file, destination_file)
        return True
    except (UnidentifiedImageError, OSError):
        LOGGER.warning("Skipping invalid image file: %s", source_file)
        return False


def _split_data(paths: List[Path], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Path]]:
    if not paths:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    paths_copy = list(paths)
    rng.shuffle(paths_copy)

    total = len(paths_copy)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    return {
        "train": paths_copy[:train_end],
        "val": paths_copy[train_end:val_end],
        "test": paths_copy[val_end:],
    }


def prepare_dataset(
    cifake_dir: Path = CIFAKE_DIR,
    imagenet_mini_dir: Path = IMAGENET_MINI_DIR,
    output_root: Path = PROCESSED_DATASET_DIR,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED,
) -> None:
    """
    Build final dataset layout:
    dataset/train/{real,fake}
    dataset/val/{real,fake}
    dataset/test/{real,fake}

    Label encoding for training:
    real -> 0 (REAL)
    fake -> 1 (AI_GENERATED)
    """
    LOGGER.info("Preparing dataset from available CIFAKE-like data + optional ImageNet Mini")

    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be between 0 and 1")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1")

    if _has_processed_dataset(output_root):
        LOGGER.info("Processed dataset already exists at %s. Skipping rebuild.", output_root)
        return

    extracted_root = _extract_zip_archives(output_root)

    candidate_roots: list[tuple[Path, str, int | None]] = []
    if cifake_dir.exists():
        candidate_roots.append((cifake_dir, "cifake", None))
    if extracted_root.exists():
        candidate_roots.append((extracted_root, "extracted", None))
    if imagenet_mini_dir.exists():
        candidate_roots.append((imagenet_mini_dir, "imagenet-mini", 0))
    if output_root.exists():
        candidate_roots.append((output_root, "generic", None))

    samples_by_path: Dict[str, LabeledImage] = {}
    for source_root, dataset_name, default_label in candidate_roots:
        discovered = discover_labeled_images(
            source_root,
            default_label=default_label,
            dataset_name=dataset_name,
        )
        for sample in discovered:
            samples_by_path[str(sample.path.resolve()).lower()] = sample

    all_samples = list(samples_by_path.values())
    manifest_samples = [sample for sample in all_samples if sample.source == "manifest"]
    if manifest_samples:
        LOGGER.info(
            "Explicit manifest records detected; using %d manifest-labeled samples and ignoring path-derived fallback data.",
            len(manifest_samples),
        )
        all_samples = manifest_samples

    real_count = sum(1 for sample in all_samples if sample.label == 0)
    fake_count = sum(1 for sample in all_samples if sample.label == 1)

    LOGGER.info(
        "Discovered labeled images under %s -> real=%d fake=%d total=%d",
        output_root,
        real_count,
        fake_count,
        len(all_samples),
    )

    if not real_count or not fake_count:
        raise FileNotFoundError(
            "Dataset preparation failed: no sufficient labeled real/fake images were found. "
            "Add explicit manifests or ensure folder/file names include real/original/authentic and fake/tampered/gan/tp tokens."
        )

    train_samples, val_samples, test_samples = split_labeled_images(
        all_samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio,
        seed=seed,
    )

    for split in ["train", "val", "test"]:
        for class_name in ["real", "fake"]:
            split_dir = output_root / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for split_name, split_samples in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
        for idx, sample in enumerate(split_samples):
            class_name = "real" if sample.label == 0 else "fake"
            dst = output_root / split_name / class_name / f"{class_name}_{idx:07d}{sample.path.suffix.lower()}"
            if _safe_copy(sample.path, dst):
                copied_count += 1

    LOGGER.info("Dataset prepared successfully. Files copied: %d", copied_count)


def load_split(split_dir: Path) -> Tuple[List[Path], List[int]]:
    image_paths: List[Path] = []
    labels: List[int] = []

    for class_name, label in LABEL_TO_ID.items():
        class_dir = _find_existing_dir(split_dir, CLASS_VARIANTS[class_name])
        if class_dir is None:
            continue
        class_images = _collect_images(class_dir)
        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))

    return image_paths, labels


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.04),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.08), ratio=(0.3, 3.3), value="random"),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return train_transform, eval_transform


def build_dataloaders(
    dataset_root: Path,
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    balanced_sampling: bool = False,
):
    train_transform, eval_transform = build_transforms(image_size=image_size)

    train_paths, train_labels = load_split(dataset_root / "train")
    val_paths, val_labels = load_split(dataset_root / "val")
    test_paths, test_labels = load_split(dataset_root / "test")

    if not train_paths or not val_paths or not test_paths:
        raise RuntimeError(
            "Missing processed data in dataset/{train,val,test}/{real,fake}. "
            "Run dataset preparation first."
        )

    train_ds = VeriSightImageDataset(train_paths, train_labels, transform=train_transform)
    val_ds = VeriSightImageDataset(val_paths, val_labels, transform=eval_transform)
    test_ds = VeriSightImageDataset(test_paths, test_labels, transform=eval_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    if balanced_sampling:
        class_counts: Dict[int, int] = {0: 0, 1: 0}
        for label in train_labels:
            class_counts[label] = class_counts.get(label, 0) + 1

        sample_weights = [1.0 / max(class_counts[label], 1) for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds,
            sampler=sampler,
            shuffle=False,
            **loader_kwargs,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            shuffle=True,
            **loader_kwargs,
        )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VeriSight Layer-2 dataset")
    parser.add_argument("--cifake-dir", type=Path, default=CIFAKE_DIR)
    parser.add_argument("--imagenet-mini-dir", type=Path, default=IMAGENET_MINI_DIR)
    parser.add_argument("--output-root", type=Path, default=PROCESSED_DATASET_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("--train-ratio + --val-ratio must be < 1")
    return args


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    prepare_dataset(
        cifake_dir=args.cifake_dir,
        imagenet_mini_dir=args.imagenet_mini_dir,
        output_root=args.output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
