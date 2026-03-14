import logging
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, HTTPException, UploadFile

from inference.onnx_inference import ONNXDetector

LOGGER = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

_detector: ONNXDetector | None = None


def get_detector() -> ONNXDetector:
    global _detector
    if _detector is None:
        _detector = ONNXDetector()
    return _detector


@router.post("/api/v1/transformer-detect")
async def transformer_detect(image: UploadFile = File(...)):
    start = time.perf_counter()

    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload a valid image.")

    suffix = Path(image.filename or "input.jpg").suffix or ".jpg"

    try:
        data = await image.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(data)
            temp_path = Path(temp_file.name)

        detector = get_detector()
        result = detector.predict(temp_path)

        processing_time_ms = int((time.perf_counter() - start) * 1000)

        response = {
            "vit_score": result["vit_score"],
            "label": result["label"],
            "processing_time_ms": processing_time_ms,
        }
        return response

    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Inference failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
    finally:
        try:
            if "temp_path" in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
        except Exception:
            LOGGER.warning("Failed to cleanup temporary file", exc_info=True)
