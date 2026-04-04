from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.pipeline.orchestrator import VerificationOrchestrator

from evaluation.common import load_samples, summarize_latencies, timestamp_utc


def _default_output_json() -> Path:
    return PROJECT_ROOT / "evaluation" / "latency_benchmark.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VeriSight end-to-end latency")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--split", choices=["all", "train", "val", "test"], default="test")
    parser.add_argument("--output-json", type=Path, default=_default_output_json())
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


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

    latencies_ms: List[float] = []
    decisions: Dict[str, int] = {}
    for sample in samples:
        result = orchestrator.run_sync(str(sample.path), metadata={})
        latencies_ms.append(float(result.get("processing_time_ms", 0)))
        decision = str(result.get("decision", "REJECT"))
        decisions[decision] = decisions.get(decision, 0) + 1

    report = {
        "timestamp_utc": timestamp_utc(),
        "dataset_root": str(Path(dataset_root).resolve()),
        "split": args.split,
        "sample_count": len(samples),
        "decision_counts": decisions,
        "latency": summarize_latencies(latencies_ms),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
