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

from typing import Dict, List, Optional, Tuple
from lexar.retrieval.multi_index_retriever import MultiIndexRetriever
from lexar.reranking.cross_encoder import LegalCrossEncoderReranker
from lexar.generation.lexar_generator import LexarGenerator
from lexar.citation.citation_mapper import attach_citations


class LexarPipeline:
    """
    End-to-end LEXAR pipeline with evidence-constrained generation.
    
    Enforces LEXAR principles at every stage:
    - Retrieval is mandatory
    - Evidence metadata is preserved
    - Generation is hard-masked to evidence
    - Failures are explicit and transparent
    """

    def __init__(self, ipc=None, judgment=None, user=None):
        """
        Initialize LEXAR pipeline.

        Args:
            ipc: IPCRetriever instance or None
            judgment: JudgmentRetriever instance or None
            user: UserRetriever instance or None
        """
        self.retriever = MultiIndexRetriever(
            ipc=ipc,
            judgment=judgment,
            user=user
        )
        self.reranker = LegalCrossEncoderReranker()
        self.generator = LexarGenerator()

        # Configuration
        self.retrieval_top_k = 10  # Initial retrieval depth
        self.reranking_top_k = 3   # Evidence count for generation
        self.min_rerank_score = 0.0  # Confidence threshold (can be tuned)

    def answer(
        self,
        query: str,
        has_user_docs: bool = False,
        top_k: int = 10,
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
        evidence, confidence = self._rerank_and_score(query, retrieved, self.reranking_top_k)

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

        # Compute confidence as average rerank score of selected evidence
        if evidence:
            scores = [c.get("rerank_score", 0.0) for c in evidence]
            confidence = sum(scores) / len(scores) if scores else 0.0
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
