from __future__ import annotations

import itertools
import os
import shutil
import sys
from pathlib import Path


def _copy_subset(source_dir: Path, target_dir: Path, limit: int) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    source_files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    for source_path in itertools.islice(source_files, limit):
        target_path = target_dir / source_path.name
        if target_path.exists():
            continue
        shutil.copy2(source_path, target_path)
        copied += 1
    return copied


def main() -> int:
    repo_root = Path(os.environ["REPO_ROOT"])
    fast_root = Path(os.environ["FAST_DATASET_ROOT"])
    subset_size = int(os.environ.get("FAST_SUBSET_SIZE", "64"))

    sources = {
        "real": repo_root / "Data" / "real",
        "fake": repo_root / "Data" / "gan_fake",
    }

    missing_sources = [name for name, path in sources.items() if not path.exists()]
    if missing_sources:
        print(f"Missing source folders for fast subset: {', '.join(missing_sources)}")
        return 1

    fast_root.mkdir(parents=True, exist_ok=True)
    total_copied = 0
    for split_name, source_dir in sources.items():
        total_copied += _copy_subset(source_dir, fast_root / split_name, subset_size)

    print(f"prepared {total_copied} files under {fast_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
