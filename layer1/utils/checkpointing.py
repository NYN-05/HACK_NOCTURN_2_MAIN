from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_checkpoint(state: Dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, map_location: str = "cpu", weights_only: bool = True) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint format is invalid. Expected dictionary.")
    return checkpoint
