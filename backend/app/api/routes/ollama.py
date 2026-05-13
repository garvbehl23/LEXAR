from __future__ import annotations

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class OllamaStatusResponse(BaseModel):
    available: bool
    models: list[str]
    selected: str | None


@router.get("/models", response_model=OllamaStatusResponse)
async def get_ollama_models():
    """
    Return available Ollama models from the local instance.
    Always returns 200 — `available: false` when Ollama is not running.
    """
    try:
        from lexar.generation.ollama_generator import get_available_models

        models = await asyncio.to_thread(get_available_models)
        return OllamaStatusResponse(
            available=len(models) > 0,
            models=models,
            selected=models[0] if models else None,
        )
    except Exception:
        return OllamaStatusResponse(available=False, models=[], selected=None)
