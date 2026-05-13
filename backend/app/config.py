from __future__ import annotations

import json
import re
from pathlib import Path
from functools import lru_cache
from typing import Any

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_cors(raw: Any) -> list[str]:
    """Handle JSON array, bare bracket list, comma-separated, or list."""
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str):
        return ["*"]
    raw = raw.strip()
    if not raw:
        return ["*"]
    if raw.startswith("["):
        # Try valid JSON first
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # [url1,url2] without quotes — strip brackets and split
            inner = raw[1:-1]
            return [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
    # Plain comma-separated
    return [x.strip() for x in raw.split(",") if x.strip()]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "LEXAR Legal AI"
    app_version: str = "1.1.1"
    debug: bool = False
    log_level: str = "INFO"

    # Accept Any so pydantic-settings does not try to JSON-decode it;
    # _coerce_cors normalises after all sources are merged.
    cors_origins: Any = ["http://localhost:8501", "http://127.0.0.1:8501", "*"]

    project_root: Path = Path(__file__).resolve().parents[2]
    data_dir: Path = project_root / "data"

    upload_dir: str = "data/raw_docs"
    processed_dir: str = "data/processed_docs"
    max_upload_mb: int = 10

    default_index: str = "ipc"
    retrieval_top_k: int = 10
    reranking_top_k: int = 3

    generator_model: str = "google/flan-t5-base"
    evidence_threshold: float = 0.5

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    @model_validator(mode="after")
    def _coerce_cors(self) -> Settings:
        self.cors_origins = _parse_cors(self.cors_origins)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
