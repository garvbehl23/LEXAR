from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

DATA_DIR = Path("data")
REQUIRED_FILES = [
    DATA_DIR / "faiss_index" / "ipc.index",
    DATA_DIR / "models" / "lexar_query_encoder_v1" / "config.json",
]
OPTIONAL_CHUNKS = [
    DATA_DIR / "processed_docs" / "ipc_chunks.json",
    DATA_DIR / "processed_docs" / "lexar_medium_chunks.json",
]


class HealthResponse(BaseModel):
    status: str
    version: str
    data_ready: bool
    chunks_ready: bool
    details: dict


@router.get("/", response_model=HealthResponse, summary="System health check")
def health():
    from backend.app.config import get_settings
    settings = get_settings()

    missing_data = [str(p) for p in REQUIRED_FILES if not p.exists()]
    missing_chunks = [str(p) for p in OPTIONAL_CHUNKS if not p.exists()]

    data_ready = len(missing_data) == 0
    chunks_ready = len(missing_chunks) == 0

    overall = "ok" if data_ready else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        data_ready=data_ready,
        chunks_ready=chunks_ready,
        details={
            "missing_required": missing_data,
            "missing_chunks": missing_chunks,
        },
    )
