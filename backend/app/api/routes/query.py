from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("lexar.backend.query")

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Legal question to answer")
    index_name: str = Field(default="ipc", description="Index to query: ipc | ipc_crpc | ipc_crpc_iea | lexar_medium")
    top_k: int = Field(default=10, ge=1, le=50)
    rerank_k: int = Field(default=3, ge=1, le=10)
    debug_mode: bool = False
    return_provenance: bool = False
    has_user_docs: bool = False


class QueryResponse(BaseModel):
    answer: str
    status: str
    evidence_count: int
    confidence: float
    evidence_ids: list[str]
    debug: Optional[dict] = None
    provenance: Optional[dict] = None


INDEX_MAP: dict[str, tuple[str, str]] = {
    "ipc": (
        "data/processed_docs/ipc_chunks.json",
        "data/faiss_index/ipc.index",
    ),
    "ipc_crpc": (
        "data/processed_docs/ipc_crpc_chunks.json",
        "data/faiss_index/ipc_crpc.index",
    ),
    "ipc_crpc_iea": (
        "data/processed_docs/ipc_crpc_iea_chunks.json",
        "data/faiss_index/ipc_crpc_iea.index",
    ),
    "lexar_medium": (
        "data/processed_docs/lexar_medium_chunks.json",
        "data/faiss_index/lexar_medium.index",
    ),
}

_pipeline_cache: dict[str, object] = {}


def _get_pipeline(index_name: str):
    """Load and cache a LexarPipeline for the requested index."""
    if index_name in _pipeline_cache:
        return _pipeline_cache[index_name]

    from pathlib import Path
    import json

    if index_name not in INDEX_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown index '{index_name}'. Valid: {list(INDEX_MAP)}")

    chunks_path, index_path = INDEX_MAP[index_name]

    if not Path(chunks_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"Chunks file not found: {chunks_path}. Run data preparation first.",
        )
    if not Path(index_path).exists():
        raise HTTPException(
            status_code=503,
            detail=f"FAISS index not found: {index_path}",
        )

    try:
        from lexar.retrieval.ipc_retriever import IPCRetriever
        from lexar.lexar_pipeline import LexarPipeline

        ipc = IPCRetriever(chunks_path=chunks_path, index_path=index_path)
        pipeline = LexarPipeline(ipc=ipc)
        _pipeline_cache[index_name] = pipeline
        logger.info("Pipeline loaded for index '%s'", index_name)
        return pipeline
    except Exception as exc:
        logger.exception("Failed to load pipeline for index '%s'", index_name)
        raise HTTPException(status_code=500, detail=f"Pipeline load failed: {exc}") from exc


@router.post("/", response_model=QueryResponse, summary="Answer a legal question")
async def answer_query(req: QueryRequest):
    """
    Run the full LEXAR pipeline:
    Query → Retrieval → Reranking → Evidence-Constrained Generation → Citations
    """
    logger.info("Query [%s]: %.80s", req.index_name, req.query)

    pipeline = _get_pipeline(req.index_name)

    try:
        result = pipeline.answer(
            query=req.query,
            has_user_docs=req.has_user_docs,
            top_k=req.top_k,
            return_provenance=req.return_provenance,
            debug_mode=req.debug_mode,
        )
    except Exception as exc:
        logger.exception("Pipeline error for query: %.80s", req.query)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    return QueryResponse(
        answer=result.get("answer", ""),
        status=result.get("status", "unknown"),
        evidence_count=result.get("evidence_count", 0),
        confidence=result.get("confidence", 0.0),
        evidence_ids=[str(e) for e in result.get("evidence_ids", [])],
        debug=result.get("debug") if req.debug_mode else None,
        provenance=result.get("provenance") if req.return_provenance else None,
    )
