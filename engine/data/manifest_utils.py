from __future__ import annotations

import csv
import hashlib
import json
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MANIFEST_EXTENSIONS = {".csv", ".json", ".jsonl"}
MANIFEST_NAME_HINTS = (
    "manifest",
    "annotation",
    "annotations",
    "label",
    "labels",
    "sample",
    "samples",
    "split",
)

REAL_LABEL_TOKENS = {
    "au",
    "authentic",
    "original",
    "real",
    "pristine",
    "clean",
    "genuine",
}
FAKE_LABEL_TOKENS = {
    "tp",
    "tampered",
    "forged",
    "splice",
    "splicing",
    "copy",
    "copy-move",
    "edited",
    "fake",
    "manipulated",
    "ai",
    "generated",
    "ai_generated",
}
IGNORE_TOKENS = {"mask", "groundtruth", "ground_truth", "gt"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
}

PATH_FIELD_CANDIDATES = (
    "path",
    "image_path",
    "file_path",
    "relative_path",
    "sample_path",
    "source_path",
    "filename",
    "file",
    "image",
)
LABEL_FIELD_CANDIDATES = (
    "label",
    "target",
    "class",
    "class_name",
    "category",
    "y",
    "is_fake",
    "fake",
    "is_manipulated",
    "manipulated",
)
GROUP_FIELD_CANDIDATES = (
    "group_id",
    "group",
    "source_id",
    "source_group",
    "product_id",
    "item_id",
    "case_id",
    "request_id",
    "session_id",
    "brand_id",
    "brand",
    "batch_id",
    "pair_id",
    "origin_id",
)
SPLIT_FIELD_CANDIDATES = ("split", "split_name", "subset", "partition", "fold")
DATASET_FIELD_CANDIDATES = ("dataset_name", "dataset", "source_dataset")


@dataclass(frozen=True)
class LabeledImage:
    path: Path
    label: int
    dataset_name: str
    group_id: str
    split_name: str | None = None
    source: str = "path"
    manifest_path: str | None = None


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _is_manifest_file(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in MANIFEST_EXTENSIONS:
        return False
    name = path.name.lower()
    return any(hint in name for hint in MANIFEST_NAME_HINTS)


def _normalize_text(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_")


def _normalize_label(value: Any) -> int | None:
    if value is None:
        return None

    if isinstance(value, bool):
        return int(bool(value))

    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric in (0, 1):
            return numeric
        return None

    normalized = _normalize_text(value)
    if not normalized:
        return None

    if normalized in {"0", "real", "genuine", "authentic", "original", "clean", "pristine"}:
        return 0
    if normalized in {
        "1",
        "fake",
        "manipulated",
        "tampered",
        "forged",
        "edited",
        "ai_generated",
        "generated",
    }:
        return 1

    if normalized in {"true", "yes", "y", "positive"}:
        return 1
    if normalized in {"false", "no", "n", "negative"}:
        return 0

    return None


def _normalize_split(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _normalize_text(value)
    return SPLIT_ALIASES.get(normalized)


def _file_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_payload(manifest_path: Path) -> list[dict[str, Any]]:
    text = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
        return rows

    payload = json.loads(text)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]

    for key in ("samples", "items", "records", "data", "annotations", "entries"):
        value = payload.get(key) if isinstance(payload, dict) else None
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]

    if isinstance(payload, dict):
        return [payload]
    return []


def _read_csv_payload(manifest_path: Path) -> list[dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row]


def _load_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    suffix = manifest_path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return _read_json_payload(manifest_path)
    if suffix == ".csv":
        return _read_csv_payload(manifest_path)
    return []


def _iter_manifest_files(root: Path) -> list[Path]:
    manifests: list[Path] = []
    for candidate in root.rglob("*"):
        if _is_manifest_file(candidate):
            manifests.append(candidate)
    return sorted(manifests)


def _resolve_path(manifest_path: Path, root: Path, raw_path: Any) -> Path | None:
    if raw_path is None:
        return None

    candidate = Path(str(raw_path).strip())
    if not str(candidate):
        return None

    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    relative_candidates = [manifest_path.parent / candidate, root / candidate]
    for relative_candidate in relative_candidates:
        if relative_candidate.exists():
            return relative_candidate.resolve()

    if candidate.suffix:
        matches = list(root.rglob(candidate.name))
        if len(matches) == 1:
            return matches[0].resolve()

    return None


def _infer_label_from_path(path: Path) -> int | None:
    name = path.name.lower()
    parts = [part.lower() for part in path.parts]

    if any(token in name for token in ["_gt", "_mask", "mask", "groundtruth", "ground_truth"]):
        return None

    if any(token in parts for token in REAL_LABEL_TOKENS):
        return 0
    if any(token in parts for token in FAKE_LABEL_TOKENS):
        return 1

    if "_orig" in name or "_auth" in name or "_real" in name:
        return 0
    if "_tam" in name or "_forg" in name or "_manip" in name or "_fake" in name:
        return 1

    return None


def infer_label_from_path(path: str | Path) -> int | None:
    return _infer_label_from_path(Path(path))


_AUGMENTATION_SUFFIX = re.compile(
    r"(?i)(?:[._-](?:aug(?:mented)?\d*|copy\d*|dup\d*|crop\d*|flip\d*|rot\d*|resize\d*|ela|mask|gt|groundtruth|tampered|forged|fake|real|variant\d*))+$"
)
_TRAILING_INDEX = re.compile(r"(?:[._-]\d+)+$")


def _normalize_group_stem(stem: str) -> str:
    cleaned = _AUGMENTATION_SUFFIX.sub("", stem)
    cleaned = _TRAILING_INDEX.sub("", cleaned)
    cleaned = cleaned.strip("._-")
    return cleaned or stem.lower()


def _infer_group_id(path: Path, root: Path, dataset_name: str) -> str:
    try:
        relative_path = path.resolve().relative_to(root.resolve())
        parts = [part.lower() for part in relative_path.parts[:-1] if part.lower() not in IGNORE_TOKENS]
    except Exception:
        parts = [part.lower() for part in path.parts[:-1] if part.lower() not in IGNORE_TOKENS]

    stem = _normalize_group_stem(path.stem.lower())
    if stem:
        parts.append(stem)

    group_parts = [dataset_name.lower()] if dataset_name else []
    group_parts.extend(part for part in parts if part)
    return "/".join(group_parts) if group_parts else stem or path.stem.lower()


def infer_group_id(path: str | Path, root: str | Path, dataset_name: str) -> str:
    return _infer_group_id(Path(path), Path(root), dataset_name)


def _row_to_sample(
    manifest_path: Path,
    root: Path,
    row: Dict[str, Any],
    default_dataset_name: str,
) -> LabeledImage | None:
    path_value = None
    for field in PATH_FIELD_CANDIDATES:
        if field in row and row[field] not in (None, ""):
            path_value = row[field]
            break

    resolved_path = _resolve_path(manifest_path, root, path_value)
    if resolved_path is None or not _is_image_file(resolved_path):
        return None

    label_value = None
    for field in LABEL_FIELD_CANDIDATES:
        if field in row and row[field] not in (None, ""):
            label_value = row[field]
            break

    label = _normalize_label(label_value)
    if label is None:
        return None

    dataset_name = default_dataset_name
    for field in DATASET_FIELD_CANDIDATES:
        if field in row and row[field] not in (None, ""):
            dataset_name = _normalize_text(row[field])
            break

    split_name = None
    for field in SPLIT_FIELD_CANDIDATES:
        if field in row and row[field] not in (None, ""):
            split_name = _normalize_split(row[field])
            if split_name is not None:
                break

    group_value = None
    for field in GROUP_FIELD_CANDIDATES:
        if field in row and row[field] not in (None, ""):
            group_value = str(row[field]).strip()
            break

    group_id = group_value or _infer_group_id(resolved_path, root, dataset_name)
    group_id = f"{group_id}::{_file_content_hash(resolved_path)}"
    return LabeledImage(
        path=resolved_path,
        label=label,
        dataset_name=dataset_name,
        group_id=group_id,
        split_name=split_name,
        source="manifest",
        manifest_path=str(manifest_path),
    )


def _scan_labeled_images(
    root: Path,
    default_label: int | None,
    dataset_name: str,
) -> list[LabeledImage]:
    samples: list[LabeledImage] = []
    seen_paths: set[str] = set()

    for candidate in root.rglob("*"):
        if not _is_image_file(candidate):
            continue

        resolved = str(candidate.resolve()).lower()
        if resolved in seen_paths:
            continue

        label = _infer_label_from_path(candidate)
        if label is None:
            if default_label is None:
                continue
            label = default_label

        seen_paths.add(resolved)
        group_id = _infer_group_id(candidate, root, dataset_name)
        group_id = f"{group_id}::{_file_content_hash(candidate)}"
        samples.append(
            LabeledImage(
                path=candidate.resolve(),
                label=label,
                dataset_name=dataset_name,
            group_id=group_id,
                split_name=None,
                source="path",
                manifest_path=None,
            )
        )

    return samples


def discover_labeled_images(
    root: str | Path,
    default_label: int | None = None,
    dataset_name: str | None = None,
) -> list[LabeledImage]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    dataset_name = (dataset_name or root_path.name or "dataset").lower()
    manifest_files = _iter_manifest_files(root_path)
    manifest_samples: list[LabeledImage] = []

    for manifest_path in manifest_files:
        for row in _load_manifest_rows(manifest_path):
            sample = _row_to_sample(manifest_path, root_path, row, dataset_name)
            if sample is not None:
                manifest_samples.append(sample)

    if manifest_samples:
        deduped: Dict[str, LabeledImage] = {}
        for sample in manifest_samples:
            deduped[str(sample.path.resolve()).lower()] = sample
        return list(deduped.values())

    return _scan_labeled_images(root_path, default_label=default_label, dataset_name=dataset_name)


def _group_samples(samples: Sequence[LabeledImage]) -> list[list[LabeledImage]]:
    grouped: Dict[str, list[LabeledImage]] = {}
    for sample in samples:
        grouped.setdefault(sample.group_id, []).append(sample)
    return list(grouped.values())


def _assign_groups_to_splits(
    groups: Sequence[Sequence[LabeledImage]],
    train_target: int,
    val_target: int,
    test_target: int,
    seed: int,
) -> Tuple[list[LabeledImage], list[LabeledImage], list[LabeledImage]]:
    rng = random.Random(seed)
    shuffled_groups = [list(group) for group in groups]
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=len, reverse=True)

    targets = {"train": train_target, "val": val_target, "test": test_target}
    split_buckets: Dict[str, list[LabeledImage]] = {"train": [], "val": [], "test": []}
    split_counts = {"train": 0, "val": 0, "test": 0}

    for group in shuffled_groups:
        chosen_split = max(
            targets,
            key=lambda split_name: (targets[split_name] - split_counts[split_name], -split_counts[split_name]),
        )
        split_buckets[chosen_split].extend(group)
        split_counts[chosen_split] += len(group)

    for split_name in split_buckets:
        rng.shuffle(split_buckets[split_name])

    return split_buckets["train"], split_buckets["val"], split_buckets["test"]


def split_labeled_images(
    samples: Sequence[LabeledImage],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[list[LabeledImage], list[LabeledImage], list[LabeledImage]]:
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) <= 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    if not samples:
        return [], [], []

    explicit_splits = [sample for sample in samples if sample.split_name is not None]
    if explicit_splits:
        if len(explicit_splits) != len(samples):
            raise ValueError(
                "Manifest split assignments must either be present for every sample or omitted entirely."
            )

        train = [sample for sample in samples if sample.split_name == "train"]
        val = [sample for sample in samples if sample.split_name == "val"]
        test = [sample for sample in samples if sample.split_name == "test"]
        if not train or not val or not test:
            raise ValueError("Manifest split assignments must include train, val, and test samples.")
        rng = random.Random(seed)
        rng.shuffle(train)
        rng.shuffle(val)
        rng.shuffle(test)
        return train, val, test

    label_buckets: Dict[int, list[LabeledImage]] = {0: [], 1: []}
    for sample in samples:
        label_buckets.setdefault(int(sample.label), []).append(sample)

    train: list[LabeledImage] = []
    val: list[LabeledImage] = []
    test: list[LabeledImage] = []

    for label_samples in label_buckets.values():
        if not label_samples:
            continue

        grouped = _group_samples(label_samples)
        total = len(label_samples)
        train_target = int(total * train_ratio)
        val_target = int(total * val_ratio)
        test_target = total - train_target - val_target
        label_train, label_val, label_test = _assign_groups_to_splits(
            grouped,
            train_target=train_target,
            val_target=val_target,
            test_target=test_target,
            seed=seed,
        )
        train.extend(label_train)
        val.extend(label_val)
        test.extend(label_test)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
