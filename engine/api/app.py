from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router
from .routes.verification import get_orchestrator

app = FastAPI(title="VeriSight Orchestration API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/health")
@app.get("/api/v1/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "VeriSight Orchestration API",
        "version": "1.0.0",
    }


@app.on_event("startup")
def preload_models() -> None:
    get_orchestrator().load_models()