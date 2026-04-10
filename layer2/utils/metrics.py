from typing import Dict, List

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    if total == 0:
        return 0.0
    return correct / total


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def summarize_epoch(
    loss_sum: float,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    steps: int,
) -> Dict[str, float | List[List[int]]]:
    if steps == 0:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_negative_rate": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    if predictions.numel() == 0 or labels.numel() == 0:
        return {
            "loss": loss_sum / steps,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "false_negative_rate": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
        }

    predictions = predictions.to(torch.int64).view(-1)
    labels = labels.to(torch.int64).view(-1)

    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())
    total = int(labels.numel())

    accuracy = _safe_div(tp + tn, total)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)

    return {
        "loss": loss_sum / steps,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": _safe_div(fn, fn + tp),
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }
