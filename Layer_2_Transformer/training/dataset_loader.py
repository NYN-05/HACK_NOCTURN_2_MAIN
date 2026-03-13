import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils.config import (
    CIFAKE_DIR,
    IMAGENET_MINI_DIR,
    LABEL_TO_ID,
    PROCESSED_DATASET_DIR,
    RANDOM_SEED,
)

LOGGER = logging.getLogger(__name__)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def _safe_copy(source_file: Path, destination_file: Path) -> bool:
    destination_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source_file) as img:
            img.verify()
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
    LOGGER.info("Preparing dataset from CIFAKE + ImageNet Mini")

    cifake_real = _collect_images(cifake_dir / "train" / "REAL") + _collect_images(cifake_dir / "test" / "REAL")
    cifake_fake = _collect_images(cifake_dir / "train" / "FAKE") + _collect_images(cifake_dir / "test" / "FAKE")
    imagenet_real = _collect_images(imagenet_mini_dir)

    all_real = cifake_real + imagenet_real
    all_fake = cifake_fake

    if not all_real or not all_fake:
        raise FileNotFoundError(
            "Dataset preparation failed: ensure dataset/cifake and dataset/imagenet_mini contain valid images"
        )

    split_real = _split_data(all_real, train_ratio, val_ratio, seed)
    split_fake = _split_data(all_fake, train_ratio, val_ratio, seed)

    for split in ["train", "val", "test"]:
        for class_name in ["real", "fake"]:
            split_dir = output_root / split / class_name
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    for split, files in split_real.items():
        for idx, src in enumerate(files):
            dst = output_root / split / "real" / f"real_{idx:07d}{src.suffix.lower()}"
            if _safe_copy(src, dst):
                copied_count += 1

    for split, files in split_fake.items():
        for idx, src in enumerate(files):
            dst = output_root / split / "fake" / f"fake_{idx:07d}{src.suffix.lower()}"
            if _safe_copy(src, dst):
                copied_count += 1

    LOGGER.info("Dataset prepared successfully. Files copied: %d", copied_count)


def load_split(split_dir: Path) -> Tuple[List[Path], List[int]]:
    image_paths: List[Path] = []
    labels: List[int] = []

    for class_name, label in LABEL_TO_ID.items():
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        class_images = _collect_images(class_dir)
        image_paths.extend(class_images)
        labels.extend([label] * len(class_images))

    return image_paths, labels


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
    return parser.parse_args()


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
