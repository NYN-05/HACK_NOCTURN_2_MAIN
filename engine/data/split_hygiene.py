from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect duplicate images across split folders.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    root = args.dataset_root
    split_map: Dict[str, List[Path]] = {"train": [], "val": [], "test": []}

    for split in split_map:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for p in split_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                split_map[split].append(p)

    hashes_by_split: Dict[str, Dict[str, str]] = {split: {} for split in split_map}
    for split, files in split_map.items():
        for p in files:
            hashes_by_split[split][str(p)] = hash_file(p)

    collisions = []
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for left, right in pairs:
        left_hash = {v: k for k, v in hashes_by_split[left].items()}
        right_hash = {v: k for k, v in hashes_by_split[right].items()}
        dup = set(left_hash).intersection(set(right_hash))
        for digest in sorted(dup):
            collisions.append(
                {
                    "hash": digest,
                    "left_split": left,
                    "left_file": left_hash[digest],
                    "right_split": right,
                    "right_file": right_hash[digest],
                }
            )

    payload = {
        "dataset_root": str(root),
        "counts": {split: len(files) for split, files in split_map.items()},
        "cross_split_duplicates": collisions,
    }

    print(json.dumps(payload, indent=2))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 1 if collisions else 0


if __name__ == "__main__":
    raise SystemExit(main())
