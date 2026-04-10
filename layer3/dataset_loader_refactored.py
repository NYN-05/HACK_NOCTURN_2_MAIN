"""
Layer 3 - Refactored Dataset Loader (Uses Cleaned Data)
CLIP model training on all 824,682 images
"""

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import numpy as np

# Import unified loader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.data.unified_loader import discover_samples_from_cleaned_data, stratified_split, ImageSample

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class GanBinaryDataset(Dataset):
    """CLIP model dataset (uses cleaned data)"""

    def __init__(self, samples: Sequence[ImageSample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        with Image.open(sample.path) as image_file:
            img = image_file.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor([float(sample.label)], dtype=torch.float32)


def get_default_transform(image_size: int = 224, is_training: bool = False):
    """Get default transforms for CLIP"""
    if is_training:
        base_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    else:
        base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])
    return base_transform


def build_dataloaders(
    dataset_root: str = None,
    image_size: int = 224,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    multiprocessing_context: str = "spawn",
    balanced_sampling: bool = True,
    use_complete_dataset: bool = True,  # NEW: Use all 824,682 images
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build dataloaders for CLIP from cleaned_data

    Args:
        dataset_root: Ignored (uses cleaned_data by default)
        use_complete_dataset: If True, uses all 824,682 images. If False, uses 8,859 labeled images.
        **other args: Standard dataloader parameters
    """
    print(f"Loading {'complete' if use_complete_dataset else 'labeled'} dataset from cleaned_data...")

    # Load samples from cleaned_data
    samples = discover_samples_from_cleaned_data(
        use_labeled_only=not use_complete_dataset,
        cleaned_data_root="cleaned_data"
    )

    print(f"Total samples loaded: {len(samples)}")
    print(f"Sample distribution: {sum(1 for s in samples if s.label == 0)} authentic, {sum(1 for s in samples if s.label == 1)} tampered")

    # Stratified split
    train_samples, val_samples, test_samples = stratified_split(samples, seed=seed)

    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")

    # Create datasets with transforms
    train_transform = get_default_transform(image_size=image_size, is_training=True)
    val_transform = get_default_transform(image_size=image_size, is_training=False)

    train_dataset = GanBinaryDataset(train_samples, transform=train_transform)
    val_dataset = GanBinaryDataset(val_samples, transform=val_transform)
    test_dataset = GanBinaryDataset(test_samples, transform=val_transform)

    # Create dataloaders
    if balanced_sampling:
        sampler = WeightedRandomSampler(
            weights=_compute_sample_weights(train_samples),
            num_samples=len(train_samples),
            replacement=True,
        )
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            multiprocessing_context=multiprocessing_context,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
    )

    return train_loader, val_loader, test_loader


def _compute_sample_weights(samples: List[ImageSample]) -> np.ndarray:
    """Compute weights for balanced sampling"""
    labels = np.array([s.label for s in samples])
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    return class_weights[labels]
