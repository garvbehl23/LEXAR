"""
LEXAR Gemini Generator

Replaces flan-t5-base with Gemini API for high-quality legal reasoning.
Keeps the same generate_with_evidence() interface as LexarGenerator so
the pipeline can swap generators transparently.
"""
from __future__ import annotations

import os
import re
from typing import Optional


_SYSTEM_PROMPT = """\
You are LEXAR, an expert Indian legal AI assistant. You answer questions \
strictly based on the legal evidence provided. Do not invent facts, statutes, \
or case law not present in the evidence. Be precise, cite sections by number, \
and explain in plain language accessible to a non-lawyer.\
"""

_ANSWER_TEMPLATE = """\
Evidence from Indian statutes:
{evidence_block}

Question: {query}

Instructions:
- Answer only using the evidence above.
- Cite specific sections (e.g., "Section 302 IPC").
- Structure your answer: (1) direct answer, (2) applicable sections, (3) explanation.
- If the evidence does not cover the question, say so explicitly.
- Be concise (3-5 sentences unless detail is essential).
"""


class GeminiGenerator:
    """
    Gemini-powered answer generator for LEXAR.

    Identical interface to LexarGenerator.generate_with_evidence().
    Falls back gracefully if the API key is missing or the call fails.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        evidence_threshold: float = 0.5,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self.evidence_threshold = evidence_threshold
        self._client = None

        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Set it in .env or export GEMINI_API_KEY=<your-key>."
            )

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=self.api_key)
            self._client = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=_SYSTEM_PROMPT,
            )
        return self._client

    def _build_evidence_block(self, evidence_chunks: list) -> str:
        lines = []
        for i, chunk in enumerate(evidence_chunks, 1):
            # Support both 'metadata' (new chunks) and 'meta' (legacy chunks)
            meta = chunk.get("metadata") or chunk.get("meta") or {}
            statute = meta.get("statute", "")
            section = meta.get("section", "")
            label = f"[{i}] {statute} {section}".strip() if (statute or section) else f"[{i}] Source"
            text = chunk.get("text", "").strip()
            lines.append(f"{label}:\n{text}")
        return "\n\n".join(lines)

    def _extract_citations(self, text: str) -> list[str]:
        pattern = re.compile(
            r"\b(?:Section|Sec\.?|s\.)\s*\d+[A-Z]?(?:\(\d+\))?(?:\s+(?:of\s+)?(?:IPC|CrPC|IEA))?"
            r"|\b(?:IPC|CrPC|IEA)\s+(?:Section|Sec\.?|s\.)\s*\d+[A-Z]?(?:\(\d+\))?",
            re.IGNORECASE,
        )
        seen: list[str] = []
        for m in pattern.finditer(text):
            tag = m.group(0).strip()
            if tag and tag not in seen:
                seen.append(tag)
        return seen[:8]

    def generate_with_evidence(
        self,
        query: str,
        evidence_chunks: list,
        max_tokens: int = 512,
        temperature: float = 0.2,
        debug_mode: bool = False,
        enable_gating: bool = True,
        track_provenance: bool = True,
        provenance_multi_layer: bool = False,
        citation_mode: str = "inline",
    ) -> dict:
        if not evidence_chunks:
            return {
                "answer": "No evidence provided.",
                "provenance": [],
                "error": "empty_evidence",
                "evidence_token_count": 0,
                "query_token_count": 0,
                "attention_mask_shape": (0, 0),
            }

        evidence_block = self._build_evidence_block(evidence_chunks)
        prompt = _ANSWER_TEMPLATE.format(
            evidence_block=evidence_block,
            query=query,
        )

        try:
            import google.generativeai as genai  # type: ignore

            client = self._get_client()
            response = client.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                ),
            )
            answer = response.text.strip()
        except Exception as exc:
            return {
                "answer": "",
                "error": str(exc),
                "provenance": {},
                "evidence_token_count": len(evidence_block.split()),
                "query_token_count": len(query.split()),
                "attention_mask_shape": (0, 0),
            }

        citations = self._extract_citations(answer)

        result = {
            "answer": answer,
            "error": None,
            "provenance": {},
            "citations": [{"text": c} for c in citations],
            "citation_mode": citation_mode,
            "answer_with_citations": answer,
            "evidence_token_count": len(evidence_block.split()),
            "query_token_count": len(query.split()),
            "attention_mask_shape": (0, 0),
            "has_token_provenance": False,
            "token_provenances": [],
            "generator": "gemini",
            "gemini_model": self.model_name,
        }

        if debug_mode:
            result["debug"] = {
                "mode": "gemini_debug",
                "prompt_preview": prompt[:500],
                "evidence_chunks_count": len(evidence_chunks),
                "citations_extracted": citations,
            }

        return result

    def stream_with_evidence(
        self,
        query: str,
        evidence_chunks: list,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        """Yield raw text chunks using Gemini's native streaming API."""
        if not evidence_chunks:
            return

        evidence_block = self._build_evidence_block(evidence_chunks)
        prompt = _ANSWER_TEMPLATE.format(evidence_block=evidence_block, query=query)

        import google.generativeai as genai  # type: ignore

        client = self._get_client()
        response = client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
            ),
            stream=True,
        )
        for chunk in response:
            try:
                text = chunk.text
                if text:
                    yield text
            except Exception:
                continue
