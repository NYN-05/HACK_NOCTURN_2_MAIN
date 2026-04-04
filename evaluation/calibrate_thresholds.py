from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.weights import CALIBRATED_DECISION_THRESHOLDS
from evaluation.common import PROJECT_ROOT


def _default_input_json() -> Path:
    return PROJECT_ROOT / "evaluation" / "latest_benchmark.json"


def _default_output_json() -> Path:
    return PROJECT_ROOT / "evaluation" / "calibrated_thresholds.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate decision thresholds from a benchmark report")
    parser.add_argument("--input-json", type=Path, default=_default_input_json())
    parser.add_argument("--output-json", type=Path, default=_default_output_json())
    return parser.parse_args()


def _clamp_score(value: float) -> int:
    return int(max(0, min(100, round(value))))


def _safe_quantile(values: Sequence[float], quantile: float, fallback: float) -> float:
    if not values:
        return fallback
    return float(np.quantile(np.asarray(list(values), dtype=float), quantile))


def _enforce_descending_thresholds(auto_approve: int, fast_track: int, suspicious: int) -> List[Tuple[int, str]]:
    auto_approve = _clamp_score(auto_approve)
    fast_track = _clamp_score(fast_track)
    suspicious = _clamp_score(suspicious)

    auto_approve = max(auto_approve, fast_track + 1)
    fast_track = max(fast_track, suspicious + 1)
    auto_approve = min(99, auto_approve)
    fast_track = min(auto_approve - 1, fast_track)
    suspicious = min(fast_track - 1, suspicious)

    if fast_track <= suspicious:
        suspicious = max(1, fast_track - 1)
    if auto_approve <= fast_track:
        auto_approve = min(99, fast_track + 1)
    if auto_approve <= fast_track:
        fast_track = max(1, auto_approve - 1)

    return [
        (auto_approve, "AUTO_APPROVE"),
        (fast_track, "FAST_TRACK"),
        (suspicious, "SUSPICIOUS"),
        (0, "REJECT"),
    ]


def main() -> None:
    args = parse_args()
    if not args.input_json.exists():
        raise FileNotFoundError(f"Benchmark JSON not found: {args.input_json}")

    report = json.loads(args.input_json.read_text(encoding="utf-8"))
    per_sample = report.get("per_sample", []) if isinstance(report, dict) else []
    if not isinstance(per_sample, list) or not per_sample:
        raise RuntimeError("Benchmark report does not contain per-sample data required for calibration.")

    genuine_scores: List[float] = []
    fake_scores: List[float] = []
    all_scores: List[float] = []
    for sample in per_sample:
        if not isinstance(sample, dict):
            continue
        score = float(sample.get("authenticity_score", 50.0))
        label = int(sample.get("label", 1))
        all_scores.append(score)
        if label == 0:
            genuine_scores.append(score)
        else:
            fake_scores.append(score)

    if not genuine_scores or not fake_scores:
        raise RuntimeError("Calibration requires both genuine and fake samples.")

    genuine_median = _safe_quantile(genuine_scores, 0.50, 75.0)
    genuine_upper = _safe_quantile(genuine_scores, 0.80, 88.0)
    fake_upper = _safe_quantile(fake_scores, 0.90, 55.0)
    fake_middle = _safe_quantile(fake_scores, 0.60, 45.0)

    auto_approve = max(genuine_upper, fake_upper + 1.0, genuine_median + 8.0)
    fast_track = max(genuine_median, fake_middle + 1.0)
    suspicious = max(fake_middle, _safe_quantile(fake_scores, 0.35, 35.0))

    thresholds = _enforce_descending_thresholds(auto_approve, fast_track, suspicious)

    report_out: Dict[str, Any] = {
        "source_json": str(args.input_json.resolve()),
        "current_thresholds": CALIBRATED_DECISION_THRESHOLDS,
        "suggested_thresholds": thresholds,
        "statistics": {
            "sample_count": len(all_scores),
            "genuine_count": len(genuine_scores),
            "fake_count": len(fake_scores),
            "genuine_median": float(np.median(genuine_scores)),
            "fake_median": float(np.median(fake_scores)),
            "genuine_mean": float(np.mean(genuine_scores)),
            "fake_mean": float(np.mean(fake_scores)),
            "score_gap": float(np.median(genuine_scores) - np.median(fake_scores)),
        },
        "rationale": [
            "AUTO_APPROVE is placed above the upper fake tail and genuine median.",
            "FAST_TRACK stays above the fake middle while preserving a wider approved band for genuine cases.",
            "SUSPICIOUS captures the upper half of fake scores before the hard reject zone.",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report_out, indent=2), encoding="utf-8")
    print(json.dumps(report_out, indent=2))


if __name__ == "__main__":
    main()
