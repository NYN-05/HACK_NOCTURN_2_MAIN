from __future__ import annotations

from typing import Any, Dict
from engine.core.config import VeriSightConfig


def compute_reliability(output: Dict[str, Any], layer_name: str = "cnn") -> float:
    """Compute reliability with weight-based scaling for UI display.
    
    Higher weight models display higher reliability, lower weight models display lower reliability.
    This reflects the importance of each model in the final decision.
    """
    if not isinstance(output, dict):
        return 0.0

    if output.get("available") is False:
        return 0.0

    score = 1.0
    raw = output.get("raw", {})
    if not isinstance(raw, dict):
        raw = {}

    if raw.get("available") is False:
        return 0.0

    uncertainty = output.get("uncertainty", raw.get("uncertainty"))
    if isinstance(uncertainty, (int, float)):
        score = max(0.0, 1.0 - max(0.0, min(1.0, float(uncertainty))))

    fallback_flag = raw.get("fallback")
    if fallback_flag is not None:
        score -= 0.45

    flags = raw.get("flags", [])
    if isinstance(flags, list) and any("fail" in str(flag).lower() for flag in flags):
        score -= 0.25

    details = raw.get("details", {})
    if isinstance(details, dict) and details.get("ocr_engine_unavailable"):
        score -= 0.2

    # Base reliability before weight scaling
    base_reliability = max(0.15, min(1.0, score))
    
    # Apply weight-based scaling for UI display
    # This makes high-weight models appear more reliable and low-weight models appear less reliable
    layer_weights = VeriSightConfig.LAYER_WEIGHTS
    layer_weight = layer_weights.get(layer_name, 0.25)
    
    # Normalize weights to 0-1 range (CNN=0.45 -> scale factor ~1.5, OCR=0.125 -> scale factor ~0.4)
    weight_scale = layer_weight / 0.3  # 0.3 is a reference point between max (0.45) and min (0.125)
    weight_scale = max(0.35, min(1.5, weight_scale))  # Clamp to reasonable range
    
    # Apply weight scaling while maintaining reliability floor
    weighted_reliability = base_reliability * weight_scale
    
    return max(0.15, min(1.0, weighted_reliability))