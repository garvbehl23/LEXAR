"""
LEXAR Ollama Generator

Connects to a locally running Ollama instance.
- Dynamically discovers available models (never hardcodes names)
- Falls back gracefully if Ollama is not running
- Exposes the same interface as GeminiGenerator
"""
from __future__ import annotations

import json
import logging
from typing import Optional
from urllib import request as urllib_request
from urllib.error import URLError

logger = logging.getLogger("lexar.generation.ollama")

OLLAMA_BASE = "http://localhost:11434"

_SYSTEM_PROMPT = """\
You are LEXAR, an expert Indian legal AI assistant. Answer questions strictly \
based on the legal evidence provided. Do not invent facts or statutes not present \
in the evidence. Cite sections by number. Be precise and accessible.\
"""

_ANSWER_TEMPLATE = """\
Evidence from Indian statutes:
{evidence_block}

Question: {query}

Instructions:
- Answer ONLY using the evidence above.
- Cite specific sections (e.g., "Section 302 IPC").
- Structure: (1) direct answer, (2) applicable sections, (3) brief explanation.
- Be concise (3-5 sentences unless more detail is essential).

Answer:\
"""


# ── Model discovery ───────────────────────────────────────────────────────────

def get_available_models() -> list[str]:
    """Return model names available in the local Ollama instance."""
    try:
        req = urllib_request.Request(
            f"{OLLAMA_BASE}/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib_request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read())
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except (URLError, json.JSONDecodeError, KeyError, OSError):
        return []


def is_ollama_running() -> bool:
    return len(get_available_models()) > 0


def resolve_model(requested: Optional[str] = None) -> str:
    """
    Pick the best available Ollama model.
    - If `requested` is in the available list, use it.
    - Otherwise use the first available model.
    - Raise ConnectionError if Ollama is not running.
    """
    available = get_available_models()
    if not available:
        raise ConnectionError(
            "Ollama is not running or has no models loaded. "
            "Start it with: ollama run llama3"
        )
    if requested and requested in available:
        return requested
    # Try prefix match (e.g. "llama3" matches "llama3:latest")
    if requested:
        prefix = requested.split(":")[0]
        for name in available:
            if name.split(":")[0] == prefix:
                return name
    return available[0]


# ── Generator ─────────────────────────────────────────────────────────────────

class OllamaGenerator:
    """
    Ollama-based answer generator for LEXAR.
    Identical public interface to GeminiGenerator.
    """

    def __init__(self, model: Optional[str] = None) -> None:
        self._requested_model = model
        self.base_url = OLLAMA_BASE

    def _resolve(self) -> str:
        return resolve_model(self._requested_model)

    def _build_evidence_block(self, chunks: list) -> str:
        lines = []
        for i, c in enumerate(chunks[:5], 1):
            # Support both 'metadata' (new chunks) and 'meta' (legacy chunks)
            meta = c.get("metadata") or c.get("meta") or {}
            statute = meta.get("statute", "")
            section = meta.get("section", "")
            label = f"[{i}] {statute} {section}".strip() if (statute or section) else f"[{i}] Source"
            lines.append(f"{label}:\n{c.get('text', '').strip()}")
        return "\n\n".join(lines)

    def _post(self, payload: dict, stream: bool = False):
        """Low-level POST to Ollama generate API."""
        data = json.dumps(payload).encode()
        req = urllib_request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return urllib_request.urlopen(req, timeout=120)

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_with_evidence(
        self,
        query: str,
        evidence_chunks: list,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ) -> dict:
        if not evidence_chunks:
            return {"answer": "No evidence provided.", "error": "empty_evidence", "provenance": {}}

        model = self._resolve()
        evidence_block = self._build_evidence_block(evidence_chunks)
        prompt = _ANSWER_TEMPLATE.format(evidence_block=evidence_block, query=query)

        try:
            resp = self._post({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "system": _SYSTEM_PROMPT,
            })
            raw = json.loads(resp.read())
            answer = raw.get("response", "").strip()
        except Exception as exc:
            return {"answer": "", "error": str(exc), "provenance": {}, "generator": "ollama"}

        return {
            "answer": answer,
            "error": None,
            "provenance": {},
            "generator": "ollama",
            "ollama_model": model,
        }

    def stream_with_evidence(
        self,
        query: str,
        evidence_chunks: list,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs,
    ):
        """Yield raw text tokens from Ollama's streaming API."""
        if not evidence_chunks:
            return

        model = self._resolve()
        evidence_block = self._build_evidence_block(evidence_chunks)
        prompt = _ANSWER_TEMPLATE.format(evidence_block=evidence_block, query=query)

        resp = self._post({
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "system": _SYSTEM_PROMPT,
        })

        for raw_line in resp:
            # urllib response yields bytes; decode before JSON-parsing
            line = (raw_line.decode("utf-8", errors="ignore") if isinstance(raw_line, bytes) else raw_line).strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                token = data.get("response", "")
                if token:
                    yield token
                if data.get("done"):
                    break
            except (json.JSONDecodeError, KeyError):
                continue
