from __future__ import annotations

from fastapi import FastAPI

from engine.pipeline.api_router import get_orchestrator, router

app = FastAPI(title="VeriSight Orchestration API", version="1.0.0")
app.include_router(router)


@app.on_event("startup")
def preload_models() -> None:
    # Preload models once so first API request avoids cold-start latency.
    get_orchestrator().load_models()
