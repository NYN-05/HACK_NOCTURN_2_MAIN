from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.orchestrator import VerificationOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VeriSight full pipeline and generate final 0-100 score."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--order-date", default=None, help="Order date (optional)")
    parser.add_argument("--delivery-date", default=None, help="Delivery date (optional)")
    parser.add_argument("--mfg-date-claimed", default=None, help="Claimed manufacturing date (optional)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    metadata = {
        "order_date": args.order_date,
        "delivery_date": args.delivery_date,
        "mfg_date_claimed": args.mfg_date_claimed,
    }
    metadata = {k: v for k, v in metadata.items() if v not in (None, "")}

    orchestrator = VerificationOrchestrator(project_root=Path(__file__).resolve().parent)
    orchestrator.load_models()

    result = orchestrator.run(image_path=image_path, metadata=metadata)

    final_output = {
        "image": str(image_path),
        "final_score": result["authenticity_score"],
        "decision": result["decision"],
        "is_rejected": result["decision"] == "REJECT",
        "layer_scores": result["layer_scores"],
        "processing_time_ms": result["processing_time_ms"],
    }

    print(json.dumps(final_output, indent=2))


if __name__ == "__main__":
    main()
