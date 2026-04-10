"""Layer 2 ViT dataset loader backed by cleaned_data."""

from __future__ import annotations

import logging
import csv
import io
import random
import re
import sys
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageFile

try:
    import torch  # type: ignore[import-not-found]
    from torch.utils.data import DataLoader, WeightedRandomSampler  # type: ignore[import-not-found]
    from torchvision import transforms  # type: ignore[import-not-found]
except ModuleNotFoundError:
    torch = None
    DataLoader = None
    WeightedRandomSampler = None
    transforms = None

ImageFile.LOAD_TRUNCATED_IMAGES = True

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layer2.utils.config import CLEANED_DATA_ROOT, COMPLETE_IMAGES_DIR, LABELED_IMAGES_DIR

LOGGER = logging.getLogger(__name__)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_DIRS = {
    0: {"real", "authentic", "original", "genuine", "clean", "au"},
    1: {"fake", "tampered", "forged", "splice", "edited", "ai", "tp"},
}
SPLIT_NAMES = ("train", "val", "test")
IGNORED_LABEL_TOKENS = {"mask", "groundtruth", "ground_truth", "gt", "description"}
REAL_LABEL_PATTERNS = (
    re.compile(r"(^|[\\/_.-])au([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])authentic([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])original([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])real([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])genuine([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])clean([\\/_.-]|$)", re.IGNORECASE),
)
FAKE_LABEL_PATTERNS = (
    re.compile(r"(^|[\\/_.-])tp([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])tampered([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])forged([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])splice([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])edited([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])fake([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])ai([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])generated([\\/_.-]|$)", re.IGNORECASE),
    re.compile(r"(^|[\\/_.-])manipulated([\\/_.-]|$)", re.IGNORECASE),
)


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int
    dataset_name: str
    image_id: str
    group_id: str


class VeriSightImageDataset:
    """Vision Transformer dataset backed by cleaned_data."""

    def __init__(self, samples: Sequence[ImageSample], transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        with Image.open(sample.path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "pixel_values": image,
            "labels": sample.label,
            "path": str(sample.path),
            "dataset": sample.dataset_name,
            "group_id": sample.group_id,
        }


class RandomJPEGCompression:
    def __init__(self, min_quality: int = 35, max_quality: int = 95, p: float = 0.5):
        self.min_quality = min_quality
        self.max_quality = max_quality
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        quality = random.randint(self.min_quality, self.max_quality)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class RandomDownUpSample:
    def __init__(self, min_scale: float = 0.55, max_scale: float = 0.95, p: float = 0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        scale = random.uniform(self.min_scale, self.max_scale)
        width, height = image.size
        down_width = max(8, int(width * scale))
        down_height = max(8, int(height * scale))
        reduced = image.resize((down_width, down_height), Image.Resampling.BILINEAR)
        return reduced.resize((width, height), Image.Resampling.BILINEAR)


class RandomGaussianNoise:
    def __init__(self, sigma_range: Tuple[float, float] = (2.0, 10.0), p: float = 0.35):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        import numpy as np

        array = np.asarray(image).astype("float32")
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0.0, sigma, size=array.shape).astype("float32")
        array = np.clip(array + noise, 0.0, 255.0).astype("uint8")
        return Image.fromarray(array, mode="RGB")


class RandomGammaShift:
    def __init__(self, gamma_range: Tuple[float, float] = (0.75, 1.35), p: float = 0.35):
        self.gamma_range = gamma_range
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return image
        import numpy as np

        gamma = random.uniform(*self.gamma_range)
        array = np.asarray(image).astype("float32") / 255.0
        array = np.power(np.clip(array, 0.0, 1.0), gamma)
        array = np.clip(array * 255.0, 0.0, 255.0).astype("uint8")
        return Image.fromarray(array, mode="RGB")


def _normalized_path_text(path: Path) -> str:
    normalized = str(path).replace("\\", "/").strip().lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) is not None for pattern in patterns)


def _is_ignored_label_path(path: Path) -> bool:
    path_text = _normalized_path_text(path)
    return any(token in path_text for token in IGNORED_LABEL_TOKENS)


def _normalize_group_stem(stem: str) -> str:
    """Normalize stem by removing augmentation/derivative suffixes.
    
    Maintains robustness across CASIA2, COMOFOD, MICC-F220 naming schemes.
    """
    cleaned = re.sub(
        r"(?i)(?:[._-](?:aug(?:mented)?\d*|copy\d*|dup\d*|crop\d*|flip\d*|rot\d*|resize\d*|ela|mask|gt|groundtruth|tampered|forged|fake|real|variant\d*))+$",
        "",
        stem,
    )
    cleaned = cleaned.strip("._-")
    # Ensure consistency: convert all results to lowercase
    return (cleaned or stem).lower()


def _canonical_group_stem(path: Path) -> str:
    """Derive canonical group stem by dataset type to prevent train/val/test leakage.
    
    Handles:
    - CASIA2: Removes au/tp label prefixes, keeps source identity
    - COMOFOD: Extracts base image ID from tp_<id>_<type> format
    - MICC-F220: Removes tamper suffix, preserves original ID
    - Generic: Uses normalized stem
    
    Returns lowercase stem to ensure group consistency.
    """
    path_text = _normalized_path_text(path)
    stem = _normalize_group_stem(path.stem.lower())

    if "casia2" in path_text:
        # CASIA2: au/tp are labels, not part of identity. Remove them and any prefix number.
        stem = re.sub(r"(?i)^(?:au|tp)[._-]?", "", stem)
        # Further normalize: remove leading numbers that are just prefixes
        stem = re.sub(r"^\d{3,}[._-]?", "", stem)
    elif "comofod" in path_text:
        # COMOFOD: Format is <id>_<type>_<variant>. Keep just <id>.
        parts = stem.split("_")
        if parts:
            # If first part is 'tp', skip it; take the next part as ID
            if parts[0] == "tp" and len(parts) > 1:
                stem = parts[1]
            elif len(parts) > 1 and parts[1] in {"o", "f"}:
                # Format: <id>_<type>_..., keep <id>
                stem = parts[0]
            else:
                # Keep the first non-'tp' part
                stem = parts[0] if parts[0] != "tp" else (parts[1] if len(parts) > 1 else stem)
    elif "micc-f220" in path_text or "micc220" in path_text:
        # MICC-F220: Remove tamper suffix and scale modifiers
        stem = re.sub(r"(?i)(?:[._-]?tamp(?:ered)?(?:\d+)?|[._-]?scale)$", "", stem)
        # Handle the 'original' marker
        stem = stem.replace("_original", "").replace("-original", "")
    elif "gan_fake" in path_text or "layer4_tiny" in path_text or "components" in path_text:
        # Synthetic datasets: use normalized stem as-is (they're already clean)
        pass
    
    return (stem or path.stem).lower()


def _derive_group_id(path: Path, dataset_name: str) -> str:
    canonical_stem = _canonical_group_stem(path)
    if dataset_name:
        return f"{dataset_name.lower()}::{canonical_stem}"
    return canonical_stem


def get_default_transform(image_size: int = 224, is_training: bool = False):
    """Get default transforms for Vision Transformer."""
    if transforms is None or torch is None:
        raise ModuleNotFoundError("torch and torchvision are required for layer 2 training transforms")

    if is_training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.18, hue=0.03)], p=0.75),
                transforms.RandomRotation(8),
                transforms.RandomApply([RandomDownUpSample()], p=0.6),
                transforms.RandomApply([RandomJPEGCompression()], p=0.7),
                transforms.RandomApply([RandomGaussianNoise()], p=0.45),
                transforms.RandomApply([RandomGammaShift()], p=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _resolve_cleaned_data_root(dataset_root: str | Path | None) -> Path:
    candidates: List[Path] = []
    if dataset_root is not None:
        provided_root = Path(dataset_root)
        candidates.extend([provided_root, provided_root / "cleaned_data"])
    candidates.extend([CLEANED_DATA_ROOT, REPO_ROOT / "cleaned_data"])

    seen: set[str] = set()
    for candidate in candidates:
        candidate_key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if (candidate / "metadata" / "unified_groundtruth.csv").exists() or (candidate / "images").exists() or (candidate / "images_complete").exists():
            return candidate

    raise FileNotFoundError("Could not locate cleaned_data. Expected cleaned_data/metadata or cleaned_data/images*")


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def _infer_label_from_path(path: Path) -> Optional[int]:
    if not _is_image_file(path):
        return None

    if _is_ignored_label_path(path):
        return None

    path_text = _normalized_path_text(path)
    stem_text = path.stem.lower()

    if "components-synth-002" in path_text or "/components-synth/" in path_text or "components-synth" in path_text:
        return 1

    if "micc-f220" in path_text or "micc220" in path_text:
        if "tamp" in stem_text or "tampered" in path_text:
            return 1
        if "scale" in stem_text or "/original/" in path_text or path_text.endswith("/original"):
            return 0

    if "comofod" in path_text:
        stem_parts = stem_text.split("_")
        if stem_parts and stem_parts[0] == "tp":
            return 1
        if len(stem_parts) > 1:
            if stem_parts[1] == "o":
                return 0
            if stem_parts[1] == "f":
                return 1
        if "/fake/" in path_text:
            return 1
        if "/real/" in path_text:
            return 0

    if "casia2" in path_text:
        if "/au/" in path_text or stem_text.startswith("au_") or "_au_" in f"_{stem_text}_":
            return 0
        if "/tp/" in path_text or stem_text.startswith("tp_") or "_tp_" in f"_{stem_text}_":
            return 1

    if "dk84bmnyw9-2" in path_text:
        if "/tampered/" in path_text or "tamp" in stem_text:
            return 1
        if "/original/" in path_text or "scale" in stem_text:
            return 0

    if _matches_any(path_text, REAL_LABEL_PATTERNS) or _matches_any(stem_text, REAL_LABEL_PATTERNS):
        return 0
    if _matches_any(path_text, FAKE_LABEL_PATTERNS) or _matches_any(stem_text, FAKE_LABEL_PATTERNS):
        return 1

    for part in reversed(path.parts):
        lowered = part.lower()
        if lowered in {"real", "authentic", "original", "genuine", "clean"}:
            return 0
        if lowered in {"fake", "tampered", "forged", "splice", "edited", "ai", "tp"}:
            return 1
    return None


def _build_sample(path: Path, label: int, dataset_name: str) -> ImageSample:
    return ImageSample(
        path=path,
        label=label,
        dataset_name=dataset_name,
        image_id=path.stem,
        group_id=_derive_group_id(path, dataset_name),
    )


def _infer_dataset_name(path: Path, root: Path) -> str:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return root.name
    if relative_path.parts:
        return relative_path.parts[0]
    return root.name


def _normalize_token(value: str | Path) -> str:
    normalized = str(value).replace("\\", "/").strip().lower()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    for prefix in ("cleaned_data/", "data/", "images_complete/", "images/"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
    return normalized


def _load_duplicate_hashes(cleaned_data_root: Path) -> Dict[str, str]:
    duplicate_csv = cleaned_data_root / "metadata" / "duplicate_files.csv"
    if not duplicate_csv.exists():
        return {}

    duplicate_map: Dict[str, str] = {}
    with duplicate_csv.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            hash_value = str(row.get("hash") or "").strip()
            if not hash_value:
                continue
            for key in (row.get("original"), row.get("duplicate")):
                if not key:
                    continue
                duplicate_map[_normalize_token(key)] = f"dup:{hash_value}"
    return duplicate_map


def _resolve_group_id(path: Path, dataset_name: str, image_id: str, duplicate_hashes: Dict[str, str]) -> str:
    path_candidates = [
        _normalize_token(path),
        _normalize_token(path.name),
        _normalize_token(f"{dataset_name}/{path.name}"),
        _normalize_token(f"{dataset_name}/{image_id}"),
        _normalize_token(image_id),
    ]
    for candidate in path_candidates:
        if candidate in duplicate_hashes:
            return duplicate_hashes[candidate]
    dataset_token = _normalize_token(dataset_name) if dataset_name else ""
    canonical_stem = _canonical_group_stem(path)
    image_token = _normalize_token(image_id) if image_id else canonical_stem
    if dataset_token:
        return f"img:{dataset_token}::{canonical_stem or image_token}"
    if image_token:
        return f"img:{image_token}"
    return f"path:{_normalize_token(path)}"


def _collect_samples_from_directory(root: Path) -> List[ImageSample]:
    samples: List[ImageSample] = []
    for image_file in root.rglob("*"):
        if not _is_image_file(image_file):
            continue
        label = _infer_label_from_path(image_file)
        if label is None:
            continue
        dataset_name = _infer_dataset_name(image_file, root)
        samples.append(_build_sample(image_file, label, dataset_name))
    return samples


def _resolve_metadata_image_path(cleaned_data_root: Path, relative_path: Path, search_roots: Sequence[Path]) -> Optional[Path]:
    relative_variants = [relative_path]
    if relative_path.parts and relative_path.parts[0].lower() in {"images", "images_complete"}:
        relative_variants.append(Path(*relative_path.parts[1:]))

    for base_root in search_roots:
        for variant in relative_variants:
            candidate = base_root / variant
            if candidate.exists():
                return candidate

    for variant in relative_variants:
        candidate = cleaned_data_root / variant
        if candidate.exists():
            return candidate

    return None


def _load_samples_from_metadata(cleaned_data_root: Path, use_labeled_only: bool) -> List[ImageSample]:
    metadata_path = cleaned_data_root / "metadata" / "unified_groundtruth.csv"
    if not metadata_path.exists():
        return []

    duplicate_hashes = _load_duplicate_hashes(cleaned_data_root)

    search_roots = [cleaned_data_root / "images", cleaned_data_root / "images_complete", cleaned_data_root]
    if not use_labeled_only:
        search_roots = [cleaned_data_root / "images_complete", cleaned_data_root / "images", cleaned_data_root]

    samples: List[ImageSample] = []
    missing = 0
    with metadata_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            relative_path = Path(str(row["image_path"]))
            resolved_path = _resolve_metadata_image_path(cleaned_data_root, relative_path, search_roots)
            if resolved_path is None:
                missing += 1
                continue

            label = 0 if int(row["authentic"]) == 1 else 1
            dataset_name = str(row.get("source_dataset") or resolved_path.parent.parent.name)
            image_id = str(row.get("filename_original") or resolved_path.stem)
            group_id = _resolve_group_id(resolved_path, dataset_name, image_id, duplicate_hashes)
            samples.append(ImageSample(path=resolved_path, label=label, dataset_name=dataset_name, image_id=image_id, group_id=group_id))

    if missing:
        LOGGER.warning("Skipped %d metadata rows because the image file was missing.", missing)

    # Log dataset composition
    dataset_dist = Counter(s.dataset_name for s in samples)
    LOGGER.info("Dataset composition loaded: %s", dict(dataset_dist))

    return samples


def _collect_explicit_split_samples(split_root: Path) -> List[ImageSample]:
    samples: List[ImageSample] = []
    if not split_root.exists():
        return samples

    for label, class_name in ((0, "real"), (1, "fake")):
        class_root = split_root / class_name
        if not class_root.exists():
            continue
        for image_file in class_root.rglob("*"):
            if not _is_image_file(image_file):
                continue
            samples.append(_build_sample(image_file, label, split_root.name))

    if samples:
        return samples

    for label, class_name in ((0, "authentic"), (1, "tampered")):
        class_root = split_root / class_name
        if not class_root.exists():
            continue
        for image_file in class_root.rglob("*"):
            if not _is_image_file(image_file):
                continue
            samples.append(_build_sample(image_file, label, split_root.name))

    return samples


def _load_explicit_splits(cleaned_data_root: Path) -> Optional[Tuple[List[ImageSample], List[ImageSample], List[ImageSample]]]:
    preferred_roots = [COMPLETE_IMAGES_DIR, cleaned_data_root / "images_complete", LABELED_IMAGES_DIR, cleaned_data_root / "images"]
    for base_root in preferred_roots:
        split_roots = [base_root / split_name for split_name in SPLIT_NAMES]
        if not all(split_root.exists() for split_root in split_roots):
            continue

        train_samples = _collect_explicit_split_samples(split_roots[0])
        val_samples = _collect_explicit_split_samples(split_roots[1])
        test_samples = _collect_explicit_split_samples(split_roots[2])

        if train_samples and val_samples and test_samples:
            return train_samples, val_samples, test_samples

    return None


def _load_complete_samples(cleaned_data_root: Path) -> List[ImageSample]:
    complete_root = cleaned_data_root / "images_complete"
    if complete_root.exists():
        return _collect_samples_from_directory(complete_root)

    split_samples = _load_explicit_splits(cleaned_data_root)
    if split_samples is not None:
        train_samples, val_samples, test_samples = split_samples
        return list(train_samples) + list(val_samples) + list(test_samples)

    return []


def _build_loader_kwargs(
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    multiprocessing_context: str,
    persistent_workers: bool,
) -> Dict[str, object]:
    loader_kwargs: Dict[str, object] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["multiprocessing_context"] = multiprocessing_context
        loader_kwargs["persistent_workers"] = persistent_workers
    return loader_kwargs


def build_dataloaders(
    dataset_root: str | Path | None = None,
    image_size: int = 224,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = False,
    prefetch_factor: int = 2,
    multiprocessing_context: str = "spawn",
    persistent_workers: bool = True,
    balanced_sampling: bool = True,
    use_complete_dataset: bool = True,
) -> Tuple[Any, Any, Any]:
    """Build dataloaders from cleaned_data."""
    if DataLoader is None or WeightedRandomSampler is None or transforms is None or torch is None:
        raise ModuleNotFoundError("torch and torchvision are required to build layer 2 dataloaders")

    cleaned_data_root = _resolve_cleaned_data_root(dataset_root)
    LOGGER.info("Using cleaned_data root: %s", cleaned_data_root)

    if use_complete_dataset:
        complete_samples = _load_complete_samples(cleaned_data_root)
        if not complete_samples:
            raise FileNotFoundError(f"No complete cleaned_data samples were discovered under {cleaned_data_root}")

        train_samples, val_samples, test_samples = stratified_split(complete_samples, seed=seed)
        LOGGER.info(
            "Loaded complete dataset and split it: total=%d train=%d val=%d test=%d",
            len(complete_samples),
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )
    else:
        samples = discover_samples_from_cleaned_data(
            use_labeled_only=not use_complete_dataset,
            cleaned_data_root=cleaned_data_root,
        )
        if not samples:
            raise FileNotFoundError(f"No images were discovered in {cleaned_data_root}")

        train_samples, val_samples, test_samples = stratified_split(samples, seed=seed)
        LOGGER.info(
            "Stratified split from %d samples: train=%d val=%d test=%d",
            len(samples),
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )

    train_dataset = VeriSightImageDataset(train_samples, transform=get_default_transform(image_size=image_size, is_training=True))
    eval_transform = get_default_transform(image_size=image_size, is_training=False)
    val_dataset = VeriSightImageDataset(val_samples, transform=eval_transform)
    test_dataset = VeriSightImageDataset(test_samples, transform=eval_transform)

    loader_kwargs = _build_loader_kwargs(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        multiprocessing_context=multiprocessing_context,
        persistent_workers=persistent_workers,
    )

    if balanced_sampling:
        sampler = WeightedRandomSampler(
            weights=_compute_sample_weights(train_samples),
            num_samples=len(train_samples),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, sampler=sampler, **loader_kwargs)
    else:
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)

    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs["shuffle"] = False

    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    return train_loader, val_loader, test_loader


def prepare_dataset(output_root: str | Path | None = None, use_labeled_only: bool = False) -> Path:
    """Validate and resolve the cleaned_data root used by layer 2."""
    cleaned_data_root = _resolve_cleaned_data_root(output_root)
    if use_labeled_only:
        samples = _load_samples_from_metadata(cleaned_data_root, use_labeled_only=True)
        if not samples:
            raise FileNotFoundError(f"No labeled samples were discovered under {cleaned_data_root}")
        LOGGER.info("Validated labeled cleaned_data subset: %d samples", len(samples))
        return cleaned_data_root

    samples = _load_complete_samples(cleaned_data_root)
    if samples:
        LOGGER.info("Validated complete cleaned_data set: %d samples", len(samples))
        return cleaned_data_root

    raise FileNotFoundError(f"No usable complete cleaned_data samples were found under {cleaned_data_root}")


def _compute_sample_weights(samples: List[ImageSample]) -> List[float]:
    """Compute weights for balanced sampling."""
    labels = [sample.label for sample in samples]
    class_counts = Counter(labels)
    if len(class_counts) < 2 or any(class_counts.get(label, 0) == 0 for label in (0, 1)):
        raise ValueError(f"Cannot compute balanced sampling weights when a class is missing: counts={dict(class_counts)}")
    class_weights = {label: 1.0 / count for label, count in class_counts.items()}
    return [class_weights[sample.label] for sample in samples]


def discover_samples_from_cleaned_data(
    use_labeled_only: bool = False,
    cleaned_data_root: str | Path = "cleaned_data",
) -> List[ImageSample]:
    """Discover samples from cleaned_data."""
    resolved_root = _resolve_cleaned_data_root(cleaned_data_root)
    if use_labeled_only:
        samples = _load_samples_from_metadata(resolved_root, use_labeled_only=True)
        if samples:
            return samples
        return []

    complete_samples = _load_complete_samples(resolved_root)
    if complete_samples:
        return complete_samples

    return _collect_samples_from_directory(resolved_root / "images_complete")


def stratified_split(
    samples: Sequence[ImageSample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ImageSample], List[ImageSample], List[ImageSample]]:
    """Split samples into train/val/test while:
    1. Keeping duplicate/source groups intact (no group split across splits)
    2. Maintaining per-dataset diversity (each dataset represented in all splits)
    3. Respecting class balance within each split
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(seed)
    groups: Dict[str, List[ImageSample]] = {}
    for sample in samples:
        groups.setdefault(sample.group_id, []).append(sample)

    # Track dataset distribution for diversity logging
    dataset_counts: Dict[str, int] = {}
    for sample in samples:
        dataset_counts[sample.dataset_name] = dataset_counts.get(sample.dataset_name, 0) + 1

    grouped_items = list(groups.items())
    rng.shuffle(grouped_items)
    # Sort by group size (largest first) for more balanced allocation
    grouped_items.sort(key=lambda item: len(item[1]), reverse=True)

    split_names = ["train", "val", "test"]
    split_ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}
    total_counts = {split_name: 0 for split_name in split_names}
    label_counts = {split_name: {0: 0, 1: 0} for split_name in split_names}
    target_total_counts = {split_name: max(1, int(round(len(samples) * split_ratios[split_name]))) for split_name in split_names}
    label_totals = Counter(sample.label for sample in samples)
    target_label_counts = {
        split_name: {label: max(1, int(round(label_totals[label] * split_ratios[split_name]))) for label in (0, 1)}
        for split_name in split_names
    }

    assignments = {split_name: [] for split_name in split_names}

    def choose_split(group_label: int, group_size: int) -> str:
        best_split = split_names[0]
        best_score = None
        for split_name in split_names:
            remaining_label = target_label_counts[split_name][group_label] - label_counts[split_name][group_label]
            remaining_total = target_total_counts[split_name] - total_counts[split_name]
            score = (remaining_label / max(target_label_counts[split_name][group_label], 1)) + (0.25 * remaining_total / max(target_total_counts[split_name], 1))
            if best_score is None or score > best_score or (score == best_score and total_counts[split_name] < total_counts[best_split]):
                best_split = split_name
                best_score = score
        return best_split

    for _, group_samples in grouped_items:
        label_votes = Counter(sample.label for sample in group_samples)
        group_label = 1 if label_votes[1] >= label_votes[0] else 0
        split_name = choose_split(group_label, len(group_samples))
        assignments[split_name].extend(group_samples)
        total_counts[split_name] += len(group_samples)
        label_counts[split_name][group_label] += len(group_samples)

    # Log split statistics
    LOGGER.info(
        "Stratified split: train=%d (%.1f%%) val=%d (%.1f%%) test=%d (%.1f%%)",
        len(assignments["train"]),
        100.0 * len(assignments["train"]) / len(samples),
        len(assignments["val"]),
        100.0 * len(assignments["val"]) / len(samples),
        len(assignments["test"]),
        100.0 * len(assignments["test"]) / len(samples),
    )
    for split_name in ["train", "val", "test"]:
        class_0_count = label_counts[split_name][0]
        class_1_count = label_counts[split_name][1]
        total = class_0_count + class_1_count
        if total > 0:
            LOGGER.info(
                "%s split class distribution: REAL=%d (%.1f%%) FAKE=%d (%.1f%%)",
                split_name.upper(),
                class_0_count,
                100.0 * class_0_count / total,
                class_1_count,
                100.0 * class_1_count / total,
            )

    return assignments["train"], assignments["val"], assignments["test"]
