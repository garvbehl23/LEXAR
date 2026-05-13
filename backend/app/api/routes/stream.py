from __future__ import annotations

import asyncio
import json
import logging
import math
import threading
from typing import AsyncGenerator, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("lexar.backend.stream")
router = APIRouter()

INDEX_MAP: dict[str, tuple[str, str]] = {
    "ipc":          ("data/processed_docs/ipc_chunks.json",         "data/faiss_index/ipc.index"),
    "ipc_crpc":     ("data/processed_docs/ipc_crpc_chunks.json",     "data/faiss_index/ipc_crpc.index"),
    "ipc_crpc_iea": ("data/processed_docs/ipc_crpc_iea_chunks.json", "data/faiss_index/ipc_crpc_iea.index"),
    "lexar_medium": ("data/processed_docs/lexar_medium_chunks.json", "data/faiss_index/lexar_medium.index"),
}

_retriever_cache: dict[str, object] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Keyword boost map
# ─────────────────────────────────────────────────────────────────────────────

_KEYWORD_BOOST: list[tuple[list[str], list[str]]] = [
    (["theft", "stolen", "steal"],           ["378", "379", "380", "381", "382"]),
    (["murder", "homicide", "killed"],       ["299", "300", "302", "303", "304"]),
    (["assault", "hurt", "grievous"],        ["319", "320", "321", "322", "323", "324", "325"]),
    (["rape", "sexual assault"],             ["375", "376"]),
    (["kidnap", "abduction"],               ["359", "360", "361", "362", "363"]),
    (["fraud", "cheating", "deceive"],       ["415", "416", "417", "418", "420"]),
    (["defamation", "libel", "slander"],     ["499", "500"]),
    (["dowry", "harassment"],               ["304B", "498A"]),
    (["criminal conspiracy"],               ["120A", "120B"]),
    (["robbery", "dacoity"],               ["390", "391", "392", "393", "394", "395", "396"]),
    (["bribery", "corruption"],            ["161", "162", "163", "164", "165"]),
    (["bail", "arrest"],                   ["436", "437", "438", "439", "440"]),
    (["evidence", "witness", "testimony"], ["118", "119", "120", "121", "122"]),
]


def _boost_scores(query: str, chunks: list[dict]) -> list[dict]:
    """Bump rerank_score for chunks whose section number matches query keywords."""
    ql = query.lower()
    boosted_sections: set[str] = set()

    for keywords, sections in _KEYWORD_BOOST:
        if any(kw in ql for kw in keywords):
            boosted_sections.update(sections)

    if not boosted_sections:
        return chunks

    for c in chunks:
        meta = c.get("metadata") or c.get("meta") or {}
        sec = str(meta.get("section_number", ""))
        chunk_id = str(c.get("chunk_id", ""))
        if sec in boosted_sections or any(s in chunk_id for s in boosted_sections):
            c["rerank_score"] = c.get("rerank_score", 0.0) + 3.0  # boost by 3 logit units

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Request model
# ─────────────────────────────────────────────────────────────────────────────

class StreamRequest(BaseModel):
    query:        str  = Field(..., min_length=3, max_length=2000)
    index_name:   str  = Field(default="ipc")
    top_k:        int  = Field(default=20, ge=1, le=50)   # increased from 10
    rerank_k:     int  = Field(default=5,  ge=1, le=10)
    has_user_docs: bool = False
    model:        str  = Field(default="gemini")
    ollama_model: str  = Field(default="")


# ─────────────────────────────────────────────────────────────────────────────
# Error helpers
# ─────────────────────────────────────────────────────────────────────────────

def _friendly(exc: Exception | str) -> str:
    msg = str(exc).lower()
    if "quota" in msg or "429" in msg or "resource_exhausted" in msg:
        return "Gemini quota exceeded. Switching to fallback model."
    if "api_key" in msg or "invalid_api_key" in msg:
        return "Invalid API key. Check your GEMINI_API_KEY in .env."
    if "ollama" in msg or "connection refused" in msg:
        return "Local model unavailable. Starting fallback..."
    if "timeout" in msg or "timed out" in msg:
        return "Generation timed out. Please try again."
    if "not ready" in msg or "index" in msg:
        return "Knowledge base not ready. Run: python scripts/rebuild_indices.py"
    return "Something went wrong. Please try again."


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_retriever(index_name: str):
    if index_name in _retriever_cache:
        return _retriever_cache[index_name]
    from pathlib import Path
    if index_name not in INDEX_MAP:
        raise ValueError(f"Unknown index '{index_name}'")
    chunks_path, index_path = INDEX_MAP[index_name]
    if not Path(chunks_path).exists() or not Path(index_path).exists():
        raise FileNotFoundError(
            f"Index '{index_name}' not found. Run: python scripts/rebuild_indices.py"
        )
    from lexar.retrieval.ipc_retriever import IPCRetriever
    r = IPCRetriever(chunks_path=chunks_path, index_path=index_path)
    _retriever_cache[index_name] = r
    return r


def _build_generator(model: str, ollama_model: str = ""):
    if model == "gemini":
        import os
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise ValueError("GEMINI_API_KEY not set.")
        from lexar.generation.gemini_generator import GeminiGenerator
        return GeminiGenerator(api_key=key)

    if model == "ollama":
        from lexar.generation.ollama_generator import OllamaGenerator
        return OllamaGenerator(model=ollama_model or None)

    from lexar.generation.lexar_generator import LexarGenerator
    return LexarGenerator()


def _fallback_chain(model: str) -> list[str]:
    return {
        "ollama":  ["ollama",  "gemini", "flan-t5"],
        "gemini":  ["gemini",  "ollama", "flan-t5"],
        "flan-t5": ["flan-t5"],
    }.get(model, ["flan-t5"])


# ─────────────────────────────────────────────────────────────────────────────
# Async streaming bridge for sync generators
# ─────────────────────────────────────────────────────────────────────────────

async def _async_stream_tokens(
    generator, query: str, evidence: list
) -> AsyncGenerator[str, None]:
    loop = asyncio.get_running_loop()
    q: asyncio.Queue[tuple[str, Optional[str]]] = asyncio.Queue(maxsize=512)

    def _produce() -> None:
        try:
            for token in generator.stream_with_evidence(query, evidence):
                if token:
                    asyncio.run_coroutine_threadsafe(
                        q.put(("tok", token)), loop
                    ).result(timeout=30)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                q.put(("err", str(exc))), loop
            ).result(timeout=10)
        finally:
            asyncio.run_coroutine_threadsafe(
                q.put(("end", None)), loop
            ).result(timeout=10)

    threading.Thread(target=_produce, daemon=True).start()

    while True:
        try:
            # 120 s — generous for slow Ollama models
            kind, value = await asyncio.wait_for(q.get(), timeout=120.0)
        except asyncio.TimeoutError:
            raise TimeoutError("Token generation timed out after 120 s")
        if kind == "end":
            break
        if kind == "err":
            raise RuntimeError(value or "Generation failed")
        yield value  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────────────
# SSE helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def _phase(phase: str, message: str) -> str:
    return _sse({"type": "phase", "phase": phase, "message": message})


def _normalize_confidence(raw_logit: float) -> float:
    """Sigmoid-normalize a cross-encoder logit to [0, 1]."""
    return round(1.0 / (1.0 + math.exp(-raw_logit)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Route
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/", summary="Stream legal query answer via SSE")
async def stream_query(req: StreamRequest):
    """
    Always returns HTTP 200 text/event-stream.
    All errors are SSE events — never raw HTTP errors.
    """

    async def generate():
        yield _phase("thinking", "Analyzing legal context...")

        try:
            retriever = await asyncio.to_thread(_load_retriever, req.index_name)
        except (ValueError, FileNotFoundError) as exc:
            yield _sse({"type": "error", "message": str(exc)})
            yield _sse({"type": "done"})
            return
        except Exception as exc:
            logger.exception("Retriever load failed")
            yield _sse({"type": "error", "message": _friendly(exc)})
            yield _sse({"type": "done"})
            return

        # ── Retrieve ─────────────────────────────────────────────────────────
        yield _phase("retrieving", "Retrieving relevant laws...")

        try:
            retrieved = await asyncio.to_thread(
                retriever.retrieve, req.query, req.top_k
            )
        except Exception as exc:
            logger.exception("Retrieval failed")
            yield _sse({"type": "error", "message": "Retrieval failed. Please try again."})
            yield _sse({"type": "done"})
            return

        if not retrieved:
            yield _sse({"type": "token", "text": "No relevant legal material found. Try rephrasing or selecting a broader knowledge base."})
            yield _sse({"type": "done"})
            return

        # ── Keyword boost + rerank ────────────────────────────────────────────
        try:
            from lexar.reranking.cross_encoder import LegalCrossEncoderReranker
            reranker = LegalCrossEncoderReranker()
            evidence = await asyncio.to_thread(
                reranker.rerank, req.query, retrieved, req.rerank_k
            )
            # Keyword boost: re-sort after boosting scores
            evidence = _boost_scores(req.query, evidence)
            evidence = sorted(evidence, key=lambda c: c.get("rerank_score", 0.0), reverse=True)[: req.rerank_k]
        except Exception as exc:
            logger.exception("Reranking failed")
            yield _sse({"type": "error", "message": "Reranking failed. Please try again."})
            yield _sse({"type": "done"})
            return

        # Validate relevance: top chunk must be above a minimum threshold
        top_score = evidence[0].get("rerank_score", 0.0) if evidence else 0.0
        if top_score < -5.0:  # cross-encoder logit < -5 means very low relevance
            yield _sse({"type": "token", "text": "Insufficient legal evidence found for this query. Please try a more specific question."})
            yield _sse({"type": "done"})
            return

        # Compute normalized confidence
        scores = [c.get("rerank_score", 0.0) for c in evidence]
        avg_logit = sum(scores) / len(scores) if scores else 0.0
        confidence = _normalize_confidence(avg_logit)

        # ── Meta event ────────────────────────────────────────────────────────
        evidence_detail = []
        for c in evidence[:6]:
            meta = c.get("metadata") or c.get("meta") or {}
            evidence_detail.append({
                "chunk_id": c.get("chunk_id", ""),
                "text":     c.get("text", "")[:500],
                "section":  meta.get("section", ""),
                "statute":  meta.get("statute", ""),
                "score":    round(float(c.get("rerank_score", 0)), 3),
            })

        yield _sse({
            "type":           "meta",
            "evidence_count": len(evidence),
            "confidence":     confidence,
            "evidence_ids":   [c.get("chunk_id", "") for c in evidence],
            "evidence":       evidence_detail,
        })

        # ── Generate with fallback chain ──────────────────────────────────────
        yield _phase("generating", "Constructing answer...")

        chain    = _fallback_chain(req.model)
        last_exc: Optional[Exception] = None

        for try_model in chain:
            if try_model != req.model:
                yield _phase("generating", f"Retrying with {try_model}…")

            try:
                gen = await asyncio.to_thread(
                    _build_generator, try_model,
                    req.ollama_model if try_model == "ollama" else ""
                )

                if hasattr(gen, "stream_with_evidence"):
                    token_count = 0
                    async for token in _async_stream_tokens(gen, req.query, evidence):
                        yield _sse({"type": "token", "text": token})
                        token_count += 1
                    if token_count > 0:
                        last_exc = None
                        break
                    raise RuntimeError("No tokens produced")
                else:
                    result = await asyncio.to_thread(
                        gen.generate_with_evidence, req.query, evidence
                    )
                    answer = result.get("answer", "")
                    if answer:
                        words = answer.split()
                        for i, w in enumerate(words):
                            yield _sse({"type": "token", "text": w + " "})
                            if i % 12 == 0:
                                await asyncio.sleep(0.01)
                        last_exc = None
                        break
                    raise RuntimeError("Empty answer")

            except Exception as exc:
                last_exc = exc
                logger.warning("Model '%s' failed: %s", try_model, exc)
                continue

        if last_exc is not None:
            yield _sse({"type": "error", "message": _friendly(last_exc)})

    async def safe_generate():
        try:
            async for event in generate():
                yield event
        except Exception as exc:
            logger.exception("Unhandled SSE error")
            yield _sse({"type": "error", "message": "Unexpected error. Please try again."})
        finally:
            yield _sse({"type": "done"})

    return StreamingResponse(
        safe_generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )
