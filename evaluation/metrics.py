from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    return true_negative, false_positive, false_negative, true_positive


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    true_negative, false_positive, false_negative, true_positive = _binary_confusion(y_true, y_pred)

    accuracy = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 0.0
    recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": [[true_negative, false_positive], [false_negative, true_positive]],
    }
    return metrics


def expected_calibration_error(probabilities: Sequence[float], labels: Sequence[int], num_bins: int = 10) -> float:
    probs = np.asarray(list(probabilities), dtype=float)
    targets = np.asarray(list(labels), dtype=int)

    if probs.size == 0:
        return 0.0

    probs = np.clip(probs, 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_indices = np.digitize(probs, bins, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    total = float(probs.size)
    ece = 0.0
    for bin_index in range(num_bins):
        mask = bin_indices == bin_index
        if not np.any(mask):
            continue
        bin_prob = float(np.mean(probs[mask]))
        bin_acc = float(np.mean(targets[mask]))
        ece += (np.sum(mask) / total) * abs(bin_acc - bin_prob)

    return float(ece)


def binary_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    targets = np.asarray(list(labels), dtype=int)
    probs = np.asarray(list(probabilities), dtype=float)
    if targets.size == 0 or len(np.unique(targets)) < 2:
        return None

    positive = targets == 1
    negative = targets == 0
    n_pos = int(np.sum(positive))
    n_neg = int(np.sum(negative))
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(probs, kind="mergesort")
    sorted_probs = probs[order]
    ranks = np.empty_like(sorted_probs, dtype=float)

    start = 0
    while start < sorted_probs.size:
        end = start + 1
        while end < sorted_probs.size and sorted_probs[end] == sorted_probs[start]:
            end += 1
        average_rank = (start + end + 1) / 2.0
        ranks[start:end] = average_rank
        start = end

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    ranked = ranks[inverse_order]
    rank_sum = float(np.sum(ranked[positive]))
    auc = (rank_sum - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
    return float(auc)
