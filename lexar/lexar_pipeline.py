"""
LEXAR End-to-End Pipeline

Implements the core LEXAR architecture with evidence-constrained generation:

    Query → Dense Retrieval → Evidence Re-ranking → 
    Evidence-Constrained Generation → Citation-Aware Output

KEY PRINCIPLES:
1. No generation without evidence (retrieval is mandatory)
2. Hard attention masking prevents parametric memory leakage
3. Evidence metadata flows through the pipeline
4. Generation is provably grounded in retrieved chunks
5. Failures are localized and transparent

PIPELINE STAGES:
1. ROUTING: Determine which indices to query (IPC, Judgment, User docs)
2. RETRIEVAL: Dense retrieval from selected indices
3. RERANKING: Cross-encoder ranking of retrieved chunks
4. GENERATION: Evidence-constrained decoder (hard attention masking)
5. CITATION: Attach citations based on generation provenance
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lexar.retrieval.multi_index_retriever import MultiIndexRetriever
from lexar.reranking.cross_encoder import LegalCrossEncoderReranker
from lexar.citation.citation_mapper import attach_citations

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_PROCESSED = _DATA_ROOT / "processed_docs"
_INDEX_DIR = _DATA_ROOT / "faiss_index"

# Maps index_name → (chunks_file, index_file, ids_file)
_INDEX_MAP: Dict[str, tuple] = {
    "ipc": (
        _PROCESSED / "ipc_chunks.json",
        _INDEX_DIR / "ipc.index",
        _INDEX_DIR / "ipc_chunk_ids.json",
    ),
    "ipc_crpc": (
        _PROCESSED / "ipc_crpc_chunks.json",
        _INDEX_DIR / "ipc_crpc.index",
        _INDEX_DIR / "ipc_crpc_chunk_ids.json",
    ),
    "ipc_crpc_iea": (
        _PROCESSED / "ipc_crpc_iea_chunks.json",
        _INDEX_DIR / "ipc_crpc_iea.index",
        _INDEX_DIR / "ipc_crpc_iea_chunk_ids.json",
    ),
    "lexar_medium": (
        _PROCESSED / "lexar_medium_chunks.json",
        _INDEX_DIR / "lexar_medium.index",
        _INDEX_DIR / "lexar_medium_chunk_ids.json",
    ),
}


def _build_generator():
    """Return GeminiGenerator if GEMINI_API_KEY is set, else LexarGenerator."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            from lexar.generation.gemini_generator import GeminiGenerator
            return GeminiGenerator(api_key=api_key)
        except Exception as exc:
            print(f"[WARN] GeminiGenerator init failed ({exc}). Falling back to flan-t5.")
    from lexar.generation.lexar_generator import LexarGenerator
    return LexarGenerator()


def _load_ipc_retriever(index_name: str):
    """Load an IPCRetriever for the given index name, or None if files are missing."""
    entry = _INDEX_MAP.get(index_name)
    if not entry:
        return None
    chunks_path, index_path, ids_path = entry
    if not chunks_path.exists() or not index_path.exists():
        return None
    try:
        from lexar.retrieval.ipc_retriever import IPCRetriever
        return IPCRetriever(
            chunks_path=str(chunks_path),
            index_path=str(index_path),
            chunk_ids_path=str(ids_path) if ids_path.exists() else None,
        )
    except Exception as exc:
        print(f"[WARN] Could not load retriever for '{index_name}': {exc}")
        return None


class LexarPipeline:
    """
    End-to-end LEXAR pipeline with evidence-constrained generation.

    Pass index_name to select which FAISS index to query.
    If GEMINI_API_KEY is in the environment, Gemini is used for generation;
    otherwise flan-t5-base is used with hard attention masking.
    """

    def __init__(self, ipc=None, judgment=None, user=None, index_name: str = "lexar_medium"):
        """
        Initialize LEXAR pipeline.

        Args:
            ipc: IPCRetriever instance or None (auto-loaded from index_name if None)
            judgment: JudgmentRetriever instance or None
            user: UserRetriever instance or None
            index_name: which FAISS index to use (ipc | ipc_crpc | ipc_crpc_iea | lexar_medium)
        """
        if ipc is None:
            ipc = _load_ipc_retriever(index_name)

        self.retriever = MultiIndexRetriever(
            ipc=ipc,
            judgment=judgment,
            user=user,
        )
        self.reranker = LegalCrossEncoderReranker()
        self.generator = _build_generator()
        self._index_name = index_name

        # Configuration
        self.retrieval_top_k = 20   # increased for better recall
        self.reranking_top_k = 5
        self.min_rerank_score = 0.0

    def answer(
        self,
        query: str,
        has_user_docs: bool = False,
        top_k: int = 10,
        rerank_k: int = 5,
        return_provenance: bool = False,
        debug_mode: bool = False,
    ) -> Dict:
        """
        Generate answer with explicit evidence grounding.

        This is the primary LEXAR generation method. It:
        1. Retrieves candidate evidence
        2. Reranks by relevance
        3. Applies hard attention masking to prevent parametric memory
        4. Returns answer + provenance metadata

        Args:
            query: str - user's legal question
            has_user_docs: bool - whether user uploaded documents
            top_k: int - initial retrieval depth
            return_provenance: bool - include token-level provenance

        Returns:
            {
                "answer": str - generated response
                "evidence_count": int - number of chunks used
                "confidence": float - rerank confidence (avg of top-k)
                "status": "success" | "no_evidence" | "low_confidence"
                "evidence_ids": list - chunk IDs for citation
                "provenance": dict (optional) - token-level tracing
            }
        """
        # ===== STAGE 1: RETRIEVAL =====
        retrieved = self._retrieve(query, has_user_docs, top_k)

        if not retrieved:
            return {
                "answer": "No relevant legal material found.",
                "evidence_count": 0,
                "confidence": 0.0,
                "status": "no_evidence",
                "evidence_ids": [],
            }

        # ===== STAGE 2: RERANKING =====
        evidence, confidence = self._rerank_and_score(query, retrieved, rerank_k or self.reranking_top_k)

        if not evidence:
            return {
                "answer": "Evidence retrieved but reranking returned no results.",
                "evidence_count": 0,
                "confidence": 0.0,
                "status": "no_evidence",
                "evidence_ids": [],
            }

        # ===== STAGE 3: EVIDENCE-CONSTRAINED GENERATION WITH GATING =====
        generation_result = self._generate_with_evidence(query, evidence, debug_mode)

        # Check if generation was rejected due to insufficient evidence
        if generation_result.get("status") == "insufficient_evidence":
            # Return the structured refusal from the gating mechanism
            return {
                "status": "insufficient_evidence",
                "reason": generation_result.get("reason"),
                "max_attention": generation_result.get("max_attention"),
                "required_threshold": generation_result.get("required_threshold"),
                "deficit": generation_result.get("deficit"),
                "evidence_count": len(evidence),
                "evidence_summary": generation_result.get("evidence_summary"),
                "suggestions": generation_result.get("suggestions"),
                "explanation": generation_result.get("explanation"),
                "query": query,
            }

        if generation_result.get("error"):
            return {
                "answer": "Generation failed: " + generation_result["error"],
                "evidence_count": len(evidence),
                "confidence": confidence,
                "status": "generation_error",
                "evidence_ids": [c.get("chunk_id") for c in evidence],
            }

        # ===== STAGE 4: CITATION MAPPING =====
        final_answer = attach_citations(
            generation_result["answer"],
            evidence
        )

        # ===== RETURN RESULT =====
        result = {
            "answer": final_answer,
            "evidence_count": len(evidence),
            "confidence": confidence,
            "status": "success",
            "evidence_ids": [c.get("chunk_id") for c in evidence],
            "evidence": evidence,
        }

        if return_provenance:
            result["provenance"] = generation_result.get("provenance", {})
            result["attention_mask_stats"] = generation_result.get("attention_mask_stats")

        if debug_mode:
            result["debug"] = generation_result.get("debug")
            # Include gating info if available
            if "gating" in generation_result:
                result["gating"] = generation_result["gating"]

        return result

    def _retrieve(
        self,
        query: str,
        has_user_docs: bool,
        top_k: int
    ) -> List[Dict]:
        """
        STAGE 1: Dense retrieval from indices.

        Args:
            query: str - user question
            has_user_docs: bool - include user documents
            top_k: int - number of candidates

        Returns:
            List of retrieved chunks with metadata
        """
        retrieved = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            has_user_docs=has_user_docs
        )
        return retrieved

    def _rerank_and_score(
        self,
        query: str,
        retrieved: List[Dict],
        top_k: int
    ) -> Tuple[List[Dict], float]:
        """
        STAGE 2: Rerank evidence by relevance + compute confidence.

        Args:
            query: str - user question
            retrieved: list - candidate chunks
            top_k: int - number to keep

        Returns:
            (evidence: list of reranked chunks, confidence: float)
        """
        evidence = self.reranker.rerank(query, retrieved, top_k)

        # Cross-encoder produces raw logits (can be negative).
        # Sigmoid-normalize to [0, 1] for display.
        if evidence:
            import math
            scores = [c.get("rerank_score", 0.0) for c in evidence]
            avg_logit = sum(scores) / len(scores)
            confidence = 1.0 / (1.0 + math.exp(-avg_logit))
        else:
            confidence = 0.0

        return evidence, confidence

    def _generate_with_evidence(
        self,
        query: str,
        evidence: List[Dict],
        debug_mode: bool = False
    ) -> Dict:
        """
        STAGE 3: Evidence-constrained generation.

        Uses hard attention masking to prevent generation from parametric memory.

        Args:
            query: str - user question
            evidence: list - reranked evidence chunks

        Returns:
            {
                "answer": str - generated text
                "error": str or None - error message if failed
                "provenance": dict - token-level metadata
                "attention_mask_stats": dict - debugging info
            }
        """
        if not evidence:
            return {
                "answer": "",
                "error": "no_evidence",
                "provenance": {}
            }

        try:
            result = self.generator.generate_with_evidence(
                query=query,
                evidence_chunks=evidence,
                max_tokens=200,
                temperature=0.0,  # Deterministic for reproducibility
                debug_mode=debug_mode,
            )

            return {
                "answer": result["answer"],
                "error": result.get("error"),
                "provenance": result.get("provenance", {}),
                "attention_mask_stats": {
                    "evidence_tokens": result.get("evidence_token_count"),
                    "query_tokens": result.get("query_token_count"),
                    "mask_shape": result.get("attention_mask_shape"),
                }
            }
        except Exception as e:
            return {
                "answer": "",
                "error": str(e),
                "provenance": {}
            }

    def answer_legacy(self, query: str, has_user_docs: bool = False, top_k: int = 10) -> str:
        """
        Legacy API for backward compatibility.
        
        WARNING: This method does NOT return full metadata.
        Use answer() for LEXAR-compliant results.

        Args:
            query: str - user question
            has_user_docs: bool - include user docs
            top_k: int - retrieval depth

        Returns:
            str - generated answer
        """
        result = self.answer(query, has_user_docs, top_k)
        return result["answer"]
