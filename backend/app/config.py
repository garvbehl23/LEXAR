from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Server ─────────────────────────────────────────────────────────────
    app_name: str = "LEXAR Legal AI"
    app_version: str = "1.1.1"
    debug: bool = False
    log_level: str = "INFO"

    # ── CORS ───────────────────────────────────────────────────────────────
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501", "*"]

    # ── Data paths ─────────────────────────────────────────────────────────
    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"

    upload_dir: str = "data/raw_docs"
    processed_dir: str = "data/processed_docs"
    max_upload_mb: int = 10

    # ── Retrieval ──────────────────────────────────────────────────────────
    default_index: str = "ipc"
    retrieval_top_k: int = 10
    reranking_top_k: int = 3

    # ── Model ──────────────────────────────────────────────────────────────
    generator_model: str = "google/flan-t5-base"
    evidence_threshold: float = 0.5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
