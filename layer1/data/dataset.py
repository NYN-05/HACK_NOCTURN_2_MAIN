from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import InterpolationMode, functional as F
from torchvision import transforms

from engine.data.manifest_utils import LabeledImage as Sample
from engine.data.manifest_utils import discover_labeled_images, split_labeled_images

try:
    from preprocessing.ela import ELAGenerator
except ModuleNotFoundError:
    from layer1.preprocessing.ela import ELAGenerator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ForensicsDataset(Dataset):
    """6-channel RGB+ELA dataset for image forensics."""

    def __init__(
        self,
        samples: Sequence[Sample],
        image_size: int = 224,
        training: bool = False,
        jpeg_quality: int = 90,
        ela_scale: float = 10.0,
        ela_cache_size: int = 1024,
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.training = training
        self.ela_generator = ELAGenerator(jpeg_quality=jpeg_quality, ela_scale=ela_scale, cache_size=ela_cache_size)

        self.photo_augment = transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12, hue=0.04),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=0.25,
                ),
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

    def _apply_shared_geometric_augment(
        self,
        rgb_image: Image.Image,
        ela_image: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            rgb_image = F.hflip(rgb_image)
            ela_image = F.hflip(ela_image)

        if random.random() < 0.2:
            rgb_image = F.vflip(rgb_image)
            ela_image = F.vflip(ela_image)

        rotation_degrees = random.uniform(-8.0, 8.0)
        if abs(rotation_degrees) > 1e-6:
            rgb_image = F.rotate(rgb_image, rotation_degrees, interpolation=InterpolationMode.BILINEAR, fill=0)
            ela_image = F.rotate(ela_image, rotation_degrees, interpolation=InterpolationMode.BILINEAR, fill=0)

        return rgb_image, ela_image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        with Image.open(sample.path) as image_file:
            image = image_file.convert("RGB")
        image = self.base_resize(image)
        ela = self.ela_generator.generate_from_path(sample.path, size=(self.image_size, self.image_size))

        if self.training:
            image, ela = self._apply_shared_geometric_augment(image, ela)
            image = self.photo_augment(image)

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

    samples = discover_labeled_images(root, dataset_name=root.name.lower())

    if not samples:
        raise RuntimeError(
            "No labeled images were discovered. Check dataset folder names, manifests, and image organization."
        )

    return samples


def stratified_split(
    samples: Sequence[Sample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    return split_labeled_images(
        samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def build_dataloaders(
    dataset_root: str,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    multiprocessing_context: str = "spawn",
    balanced_sampling: bool = True,
    ela_cache_size: int = 1024,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    samples = discover_samples(dataset_root)
    train_samples, val_samples, test_samples = stratified_split(samples, seed=seed)

    train_dataset = ForensicsDataset(train_samples, image_size=image_size, training=True, ela_cache_size=ela_cache_size)
    val_dataset = ForensicsDataset(val_samples, image_size=image_size, training=False, ela_cache_size=ela_cache_size)
    test_dataset = ForensicsDataset(test_samples, image_size=image_size, training=False, ela_cache_size=ela_cache_size)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = multiprocessing_context

    train_loader_kwargs = {
        "batch_size": batch_size,
        **loader_kwargs,
    }

    if balanced_sampling:
        labels = [sample.label for sample in train_samples]
        class_counts = np.bincount(labels, minlength=2)
        class_weights = [1.0 / max(1, class_counts[label]) for label in labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(class_weights, dtype=torch.double),
            num_samples=len(class_weights),
            replacement=True,
        )
        train_loader_kwargs["sampler"] = sampler
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader
