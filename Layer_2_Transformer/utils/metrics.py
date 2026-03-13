from typing import Dict

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    if total == 0:
        return 0.0
    return correct / total


def summarize_epoch(loss_sum: float, acc_sum: float, steps: int) -> Dict[str, float]:
    if steps == 0:
        return {"loss": 0.0, "accuracy": 0.0}
    return {
        "loss": loss_sum / steps,
        "accuracy": acc_sum / steps,
    }
