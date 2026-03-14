from __future__ import annotations

from fastapi import FastAPI

from pipeline.api_router import router

app = FastAPI(title="VeriSight Orchestration API", version="1.0.0")
app.include_router(router)
