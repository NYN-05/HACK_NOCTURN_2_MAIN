from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from engine.data.manifest_utils import LabeledImage, discover_labeled_images

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_dataset_root() -> Path:
    for candidate_name in ("Data", "data", "dataset"):
        candidate = PROJECT_ROOT / candidate_name
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "Data"


def resolve_dataset_root(dataset_root: str | Path, split: str = "test") -> Path:
    root = Path(dataset_root)
    if split != "all":
        split_root = root / split
        if split_root.exists():
            return split_root
    return root


def load_samples(dataset_root: str | Path, split: str = "test") -> List[LabeledImage]:
    root = resolve_dataset_root(dataset_root, split=split)
    samples = discover_labeled_images(root, dataset_name=root.name.lower())
    if not samples and split != "all":
        samples = discover_labeled_images(Path(dataset_root), dataset_name=Path(dataset_root).name.lower())
    return samples


def decision_to_binary(decision: str) -> int:
    if decision in {"AUTO_APPROVE", "FAST_TRACK"}:
        return 0
    return 1


def fake_probability_from_score(score: float) -> float:
    return float(np.clip(1.0 - (float(score) / 100.0), 0.0, 1.0))


def summarize_latencies(latencies_ms: Sequence[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0}

    values = np.asarray(list(latencies_ms), dtype=float)
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p90_ms": float(np.percentile(values, 90)),
        "p99_ms": float(np.percentile(values, 99)),
    }


def append_history(history_path: str | Path, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    history: List[Dict[str, Any]] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                history = [item for item in existing if isinstance(item, dict)]
            elif isinstance(existing, dict) and isinstance(existing.get("runs"), list):
                history = [item for item in existing["runs"] if isinstance(item, dict)]
        except json.JSONDecodeError:
            history = []

    history.append(entry)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
