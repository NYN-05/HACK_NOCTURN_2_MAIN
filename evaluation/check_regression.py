from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.weights import BENCHMARK_BUDGET
from evaluation.common import PROJECT_ROOT


def _default_history_json() -> Path:
    return PROJECT_ROOT / "evaluation" / "benchmark_history.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check benchmark history for regressions")
    parser.add_argument("--history-json", type=Path, default=_default_history_json())
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"History JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("runs"), list):
        return [entry for entry in payload["runs"] if isinstance(entry, dict)]
    raise RuntimeError("Unsupported benchmark history format.")


def main() -> None:
    args = parse_args()
    history = _load_history(args.history_json)
    if not history:
        raise RuntimeError("Benchmark history is empty.")

    latest = history[-1]
    baseline = history[-2] if len(history) > 1 else latest
    latest_metrics = latest.get("metrics", {}) if isinstance(latest, dict) else {}
    baseline_metrics = baseline.get("metrics", {}) if isinstance(baseline, dict) else {}

    latest_accuracy = float(latest_metrics.get("accuracy", 0.0))
    baseline_accuracy = float(baseline_metrics.get("accuracy", 0.0))
    latest_p90 = float(latest_metrics.get("latency", {}).get("p90_ms", 0.0))
    baseline_p90 = float(baseline_metrics.get("latency", {}).get("p90_ms", 0.0))
    latest_p99 = float(latest_metrics.get("latency", {}).get("p99_ms", 0.0))
    baseline_p99 = float(baseline_metrics.get("latency", {}).get("p99_ms", 0.0))
    latest_ece = float(latest_metrics.get("ece", 0.0) or 0.0)
    baseline_ece = float(baseline_metrics.get("ece", 0.0) or 0.0)

    deltas = {
        "accuracy": latest_accuracy - baseline_accuracy,
        "p90_latency_ms": latest_p90 - baseline_p90,
        "p99_latency_ms": latest_p99 - baseline_p99,
        "ece": latest_ece - baseline_ece,
    }

    regressions: List[str] = []
    if latest_accuracy < float(BENCHMARK_BUDGET["min_accuracy"]):
        regressions.append("accuracy_below_budget")
    if latest_p90 > float(BENCHMARK_BUDGET["max_p90_latency_ms"]):
        regressions.append("p90_latency_above_budget")
    if latest_p99 > float(BENCHMARK_BUDGET["max_p99_latency_ms"]):
        regressions.append("p99_latency_above_budget")
    if len(history) > 1:
        if deltas["accuracy"] < -0.01:
            regressions.append("accuracy_regressed")
        if deltas["p90_latency_ms"] > 500.0:
            regressions.append("p90_latency_regressed")
        if deltas["p99_latency_ms"] > 1000.0:
            regressions.append("p99_latency_regressed")
        if deltas["ece"] > 0.02:
            regressions.append("calibration_regressed")

    report = {
        "history_json": str(args.history_json.resolve()),
        "history_size": len(history),
        "baseline_timestamp": baseline.get("timestamp_utc"),
        "latest_timestamp": latest.get("timestamp_utc"),
        "deltas": deltas,
        "regressions": regressions,
        "passed": not regressions,
        "budget": BENCHMARK_BUDGET,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
