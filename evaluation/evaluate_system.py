from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.pipeline.orchestrator import VerificationOrchestrator

from evaluation.common import (
    append_history,
    decision_to_binary,
    fake_probability_from_score,
    load_samples,
    summarize_latencies,
    timestamp_utc,
)
from evaluation.metrics import binary_auc, compute_metrics, expected_calibration_error


def _default_output_json() -> Path:
    return PROJECT_ROOT / "evaluation" / "latest_benchmark.json"


def _default_history_json() -> Path:
    return PROJECT_ROOT / "evaluation" / "benchmark_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VeriSight end-to-end benchmark")
    parser.add_argument("--dataset-root", type=Path, default=None, help="Root folder containing benchmark images")
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--output-json", type=Path, default=_default_output_json())
    parser.add_argument("--history-json", type=Path, default=_default_history_json())
    parser.add_argument("--append-history", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of samples to evaluate")
    return parser.parse_args()


def _serialize_sample_result(sample, result: Dict[str, Any], predicted_label: int, correct: bool) -> Dict[str, Any]:
    return {
        "path": str(sample.path),
        "dataset_name": sample.dataset_name,
        "group_id": sample.group_id,
        "label": int(sample.label),
        "decision": result["decision"],
        "predicted_label": int(predicted_label),
        "correct": bool(correct),
        "authenticity_score": int(result["authenticity_score"]),
        "confidence": float(result.get("confidence", 0.0)),
        "abstained": bool(result.get("abstained", False)),
        "available_layers": list(result.get("available_layers", [])),
        "layer_scores": dict(result.get("layer_scores", {})),
        "layer_reliabilities": dict(result.get("layer_reliabilities", {})),
        "effective_weights": dict(result.get("effective_weights", {})),
        "layer_status": dict(result.get("layer_status", {})),
        "processing_time_ms": int(result.get("processing_time_ms", 0)),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root or PROJECT_ROOT / "Data"
    samples = load_samples(dataset_root, split=args.split)
    if args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        raise RuntimeError(f"No benchmark samples were found under {dataset_root} (split={args.split}).")

    orchestrator = VerificationOrchestrator(project_root=PROJECT_ROOT)
    orchestrator.load_models()

    per_sample: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    fake_probabilities: List[float] = []
    latencies_ms: List[float] = []
    decision_counts: Counter[str] = Counter()
    abstain_count = 0

    for sample in samples:
        result = orchestrator.run_sync(str(sample.path), metadata={})
        decision = str(result.get("decision", "REJECT"))
        predicted_label = decision_to_binary(decision)
        correct = predicted_label == int(sample.label)

        y_true.append(int(sample.label))
        y_pred.append(predicted_label)
        fake_probabilities.append(fake_probability_from_score(float(result.get("authenticity_score", 50.0))))
        latencies_ms.append(float(result.get("processing_time_ms", 0)))
        decision_counts[decision] += 1
        if result.get("abstained"):
            abstain_count += 1

        per_sample.append(_serialize_sample_result(sample, result, predicted_label, correct))

    y_true_array = np.asarray(y_true, dtype=int)
    y_pred_array = np.asarray(y_pred, dtype=int)

    summary_metrics = compute_metrics(y_true_array, y_pred_array)
    summary_metrics["auc"] = binary_auc(y_true, fake_probabilities)
    summary_metrics["ece"] = expected_calibration_error(fake_probabilities, y_true)
    summary_metrics["abstain_rate"] = abstain_count / max(len(samples), 1)
    summary_metrics["coverage"] = 1.0 - summary_metrics["abstain_rate"]
    summary_metrics["latency"] = summarize_latencies(latencies_ms)

    summary = {
        "timestamp_utc": timestamp_utc(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": args.split,
        "sample_count": len(samples),
        "decision_counts": dict(decision_counts),
        "metrics": summary_metrics,
    }

    report = {
        "summary": summary,
        "per_sample": per_sample,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.append_history:
        append_history(
            args.history_json,
            {
                "timestamp_utc": summary["timestamp_utc"],
                "dataset_root": summary["dataset_root"],
                "split": summary["split"],
                "sample_count": summary["sample_count"],
                "metrics": summary_metrics,
                "output_json": str(args.output_json),
            },
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
