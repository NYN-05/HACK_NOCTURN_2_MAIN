from __future__ import annotations

import torch


def resolve_device(requested_device: str | None = None) -> torch.device:
    """Resolve runtime device with safe CUDA fallback to CPU."""
    if requested_device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    req = requested_device.lower().strip()
    if req.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")

    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return torch.device(req)


def use_cuda(device: torch.device) -> bool:
    return device.type == "cuda"
