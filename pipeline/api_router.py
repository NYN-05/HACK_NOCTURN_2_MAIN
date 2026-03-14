from __future__ import annotations

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from configs.weights import VERIFY_ENDPOINT
from pipeline.orchestrator import VerificationOrchestrator

LOGGER = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

_orchestrator: VerificationOrchestrator | None = None


def get_orchestrator() -> VerificationOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = VerificationOrchestrator()
        _orchestrator.load_models()
    return _orchestrator


@router.post(VERIFY_ENDPOINT)
async def verify_image(
    image: UploadFile = File(...),
    order_date: str | None = Form(None),
    delivery_date: str | None = Form(None),
    mfg_date_claimed: str | None = Form(None),
):
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a valid image.")

    suffix = Path(image.filename or "input.jpg").suffix or ".jpg"
    metadata = {
        "order_date": order_date,
        "delivery_date": delivery_date,
        "mfg_date_claimed": mfg_date_claimed,
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "")}

    try:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)

        result = get_orchestrator().run(temp_path, metadata=metadata)

        return {
            "authenticity_score": result["authenticity_score"],
            "decision": result["decision"],
            "layer_scores": result["layer_scores"],
            "processing_time_ms": result["processing_time_ms"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Verification pipeline failed")
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc
    finally:
        try:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except Exception:
            LOGGER.warning("Failed to cleanup temporary file", exc_info=True)
