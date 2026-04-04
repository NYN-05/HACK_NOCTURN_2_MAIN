from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract hard negatives from evaluate_system JSON samples.")
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--score-window-min", type=float, default=35.0)
    parser.add_argument("--score-window-max", type=float, default=75.0)
    args = parser.parse_args()

    payload = json.loads(args.input_json.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])

    hard = []
    for sample in samples:
        true_label = str(sample.get("label_true", ""))
        pred_label = str(sample.get("label_pred", ""))
        score = float(sample.get("authenticity_score", 50.0))

        is_fp = true_label == "genuine" and pred_label == "manipulated"
        in_borderline = args.score_window_min <= score <= args.score_window_max
        if is_fp or in_borderline:
            hard.append(sample)

    out = {
        "input_report": str(args.input_json),
        "count": len(hard),
        "hard_negatives": hard,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved hard negatives: {args.output_json} ({len(hard)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
