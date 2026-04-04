from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from configs.weights import ENABLE_REQUEST_CACHE, REQUEST_CACHE_MAX_ITEMS, RESPONSE_SCHEMA_VERSION, VERIFY_ENDPOINT
from engine.pipeline.orchestrator import VerificationOrchestrator

LOGGER = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

_orchestrator: VerificationOrchestrator | None = None
_MAX_CONCURRENT_REQUESTS = int(os.getenv("VERISIGHT_MAX_CONCURRENT_REQUESTS", "4"))
_REQUEST_TIMEOUT_MS = int(os.getenv("VERISIGHT_REQUEST_TIMEOUT_MS", "15000"))
_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)


@dataclass
class _CacheClaim:
    state: str
    value: Dict[str, Any] | None = None
    future: asyncio.Future | None = None


class _RequestCache:
    def __init__(self, max_items: int) -> None:
        self._max_items = max_items
        self._lock = asyncio.Lock()
        self._entries: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._inflight: Dict[str, asyncio.Future] = {}

    async def claim(self, key: str) -> _CacheClaim:
        loop = asyncio.get_running_loop()
        async with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                return _CacheClaim(state="hit", value=copy.deepcopy(self._entries[key]))

            future = self._inflight.get(key)
            if future is None:
                future = loop.create_future()
                self._inflight[key] = future
                return _CacheClaim(state="owner", future=future)

            return _CacheClaim(state="wait", future=future)

    async def settle(self, key: str, value: Dict[str, Any] | None = None, exc: Exception | None = None) -> None:
        async with self._lock:
            future = self._inflight.pop(key, None)
            if exc is None and value is not None:
                self._entries[key] = copy.deepcopy(value)
                self._entries.move_to_end(key)
                while len(self._entries) > self._max_items:
                    self._entries.popitem(last=False)

        if future is not None and not future.done():
            if exc is None and value is not None:
                future.set_result(copy.deepcopy(value))
            elif exc is not None:
                future.set_exception(exc)


_request_cache = _RequestCache(max(1, REQUEST_CACHE_MAX_ITEMS)) if ENABLE_REQUEST_CACHE else None


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

    metadata = {
        "order_date": order_date,
        "delivery_date": delivery_date,
        "mfg_date_claimed": mfg_date_claimed,
    }
    metadata = {key: value for key, value in metadata.items() if value not in (None, "")}
    cache_key: str | None = None

    try:
        file_bytes = await image.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        metadata_key = json.dumps(metadata, sort_keys=True, separators=(",", ":")) if metadata else "{}"
        cache_key = hashlib.sha256(file_bytes).hexdigest()
        cache_key = f"{cache_key}:{hashlib.sha256(metadata_key.encode('utf-8')).hexdigest()}"

        if _request_cache is not None:
            claim = await _request_cache.claim(cache_key)
            if claim.state == "hit":
                return claim.value or {}

            if claim.state == "wait" and claim.future is not None:
                return copy.deepcopy(await claim.future)

        image_object = Image.open(BytesIO(file_bytes)).convert("RGB")
        image_object.load()

        async with _semaphore:
            try:
                result = await asyncio.wait_for(
                    get_orchestrator().run(image_object, metadata=metadata),
                    timeout=max(1.0, _REQUEST_TIMEOUT_MS / 1000.0),
                )
            except asyncio.TimeoutError as exc:
                if _request_cache is not None:
                    await _request_cache.settle(cache_key, exc=exc)
                raise HTTPException(status_code=503, detail="Verification timed out. Please retry.") from exc

        response = {
            "schema_version": RESPONSE_SCHEMA_VERSION,
            "authenticity_score": result["authenticity_score"],
            "decision": result["decision"],
            "layer_scores": result["layer_scores"],
            "layer_reliabilities": result.get("layer_reliabilities", {}),
            "effective_weights": result.get("effective_weights", {}),
            "confidence": result.get("confidence", 0.0),
            "layer_status": result.get("layer_status", {}),
            "layer_outputs": result.get("layer_outputs", {}),
            "available_layers": result.get("available_layers", []),
            "abstained": result.get("abstained", False),
            "fusion_strategy": result.get("fusion_strategy", "weighted_average"),
            "meta_model_used": result.get("meta_model_used", False),
            "early_exit_triggered": result.get("early_exit_triggered", False),
            "processing_time_ms": result["processing_time_ms"],
        }

        if _request_cache is not None:
            await _request_cache.settle(cache_key, response)

        return response
    except HTTPException:
        raise
    except Exception as exc:
        if _request_cache is not None and cache_key is not None:
            await _request_cache.settle(cache_key, exc=exc)
        LOGGER.exception("Verification pipeline failed")
        raise HTTPException(status_code=500, detail=f"Verification failed: {exc}") from exc
