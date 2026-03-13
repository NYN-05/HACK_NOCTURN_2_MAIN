from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from preprocessing.ela import ELAGenerator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    path: Path
    label: int
    dataset_name: str


class ForensicsDataset(Dataset):
    """6-channel RGB+ELA dataset for image forensics."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 224,
        training: bool = False,
        jpeg_quality: int = 90,
        ela_scale: float = 10.0,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.training = training
        self.ela_generator = ELAGenerator(jpeg_quality=jpeg_quality, ela_scale=ela_scale)

        self.geom_augment = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=8),
            ]
        )

        self.base_resize = transforms.Resize((image_size, image_size))
        self.to_tensor = transforms.ToTensor()

        self.rgb_normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        self.ela_normalize = transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        image = self.base_resize(image)

        if self.training:
            image = self.geom_augment(image)

        ela = self.ela_generator.generate(image)

        rgb_tensor = self.rgb_normalize(self.to_tensor(image))
        ela_tensor = self.ela_normalize(self.to_tensor(ela))
        fusion = torch.cat((rgb_tensor, ela_tensor), dim=0)

        return {
            "image": fusion,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": str(sample.path),
            "dataset": sample.dataset_name,
        }


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _iter_images(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if _is_image(p)]


def _label_from_path(path: Path) -> Optional[int]:
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]

    auth_tokens = {"au", "authentic", "original", "real", "pristine", "clean"}
    manip_tokens = {
        "tp",
        "tampered",
        "forged",
        "splice",
        "splicing",
        "copy",
        "copy-move",
        "edited",
        "fake",
    }

    if any(token in name for token in ["_gt", "_mask", "mask", "groundtruth"]):
        return None

    if any(token in parts for token in auth_tokens):
        return 0
    if any(token in parts for token in manip_tokens):
        return 1

    if "_orig" in name or "_auth" in name:
        return 0
    if "_tam" in name or "_forg" in name or "_manip" in name:
        return 1

    return None


def discover_samples(dataset_root: str) -> List[Sample]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    samples: List[Sample] = []

    dataset_dirs = {
        "casia": ["CASIA2", "CASIA2.0_Groundtruth"],
        "comofod": ["comofod_small", "CoMoFoD_small_v2"],
        "cg1050": ["CG-1050", "CG1050", "cg1050"],
    }

    discovered_paths = set()

    for dataset_name, hints in dataset_dirs.items():
        for hint in hints:
            for match in root.rglob(hint):
                if match.is_dir():
                    discovered_paths.add((dataset_name, match.resolve()))

    if not discovered_paths:
        discovered_paths.add(("generic", root.resolve()))

    for dataset_name, folder in sorted(discovered_paths, key=lambda x: str(x[1])):
        for image_path in _iter_images(folder):
            label = _label_from_path(image_path)
            if label is None:
                continue
            samples.append(Sample(path=image_path, label=label, dataset_name=dataset_name))

    if not samples:
        raise RuntimeError(
            "No labeled images were discovered. Check dataset folder names and image organization."
        )

    return samples


def stratified_split(
    samples: Sequence[Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    grouped: Dict[int, List[Sample]] = {0: [], 1: []}
    for sample in samples:
        grouped[sample.label].append(sample)

    rng = random.Random(seed)

    train, val, test = [], [], []

    for label, group in grouped.items():
        if not group:
            raise RuntimeError(f"No samples found for label {label}")

        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(group[:n_train])
        val.extend(group[n_train : n_train + n_val])
        test.extend(group[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_dataloaders(
    dataset_root: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    samples = discover_samples(dataset_root)
    train_samples, val_samples, test_samples = stratified_split(samples, seed=seed)

    train_dataset = ForensicsDataset(train_samples, image_size=image_size, training=True)
    val_dataset = ForensicsDataset(val_samples, image_size=image_size, training=False)
    test_dataset = ForensicsDataset(test_samples, image_size=image_size, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader
