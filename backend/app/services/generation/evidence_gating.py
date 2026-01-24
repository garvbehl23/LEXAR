"""
Evidence Sufficiency Gating

Enforces that answers are only finalized when there is sufficient evidential
support from retrieved legal chunks. This is a critical safety mechanism for
LEXAR to prevent hallucination and ensure legal reliability.

Core Principle:
    An answer must not be returned to the user unless at least one evidence
    chunk received sufficient attention during generation.

Definition:
    S = max_i A(c_i)  where A(c_i) is attention mass on chunk c_i
    
    If S < τ (threshold), the answer is rejected and a structured refusal
    is returned with explanation and suggestions.

This ensures LEXAR cannot bypass evidence constraints through generation.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EvidenceSufficiencyGate:
    """
    Gate that enforces minimum evidence support before answer finalization.
    
    This is the final safety check in LEXAR's evidence constraint pipeline:
    
    Evidence-Constrained Attention (prevents attending outside evidence)
        ↓
    Evidence Attribution (tracks which chunks were attended)
        ↓
    Evidence Sufficiency Gating (ensures sufficient attention) ← THIS MODULE
        ↓
    Answer Finalization or Structured Refusal
    
    Attributes:
        threshold (float): Minimum attention mass required (default 0.5)
        enable_gating (bool): Whether gating is active (can be disabled for testing)
        strict_mode (bool): If True, uses > instead of >= for threshold check
    """

    def __init__(
        self,
        threshold: float = 0.5,
        enable_gating: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize the evidence sufficiency gate.
        
        Args:
            threshold: Minimum attention mass required for answer finalization.
                      Range: [0.0, 1.0]. Default 0.5 means at least one chunk
                      must receive 50% of attention.
            enable_gating: If False, gating is bypassed (for testing/comparison).
                          Still tracks metrics but doesn't reject answers.
            strict_mode: If True, uses > instead of >= (stricter enforcement).
        
        Raises:
            ValueError: If threshold not in [0.0, 1.0]
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")
        
        self.threshold = threshold
        self.enable_gating = enable_gating
        self.strict_mode = strict_mode
        
        logger.info(
            f"Evidence Sufficiency Gate initialized: "
            f"threshold={threshold}, enabled={enable_gating}, strict={strict_mode}"
        )

    def evaluate(
        self,
        attention_distribution: Dict[str, float],
        evidence_chunks: List[Dict[str, Any]],
        query: str,
        answer: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate if evidence sufficiency is met for answer finalization.
        
        Args:
            attention_distribution: {chunk_id: attention_weight} from debug mode
            evidence_chunks: Retrieved evidence chunks with metadata
            query: User's original query
            answer: Generated answer from the model
        
        Returns:
            Tuple of (passes_gate: bool, gate_info: dict)
            
            If passes_gate=True:
                gate_info = {
                    "passes": True,
                    "max_attention": float,
                    "max_chunk_id": str,
                    "evidence_count": int,
                    "threshold": float,
                    "margin": float  # How much above threshold
                }
            
            If passes_gate=False:
                gate_info = {
                    "passes": False,
                    "max_attention": float,
                    "max_chunk_id": str,
                    "evidence_count": int,
                    "threshold": float,
                    "deficit": float,  # How much below threshold
                    "refusal": {...}  # Structured refusal message
                }
        
        Raises:
            ValueError: If attention_distribution doesn't sum to ~1.0
        """
        # Validate inputs
        if not attention_distribution:
            raise ValueError("Attention distribution cannot be empty")
        
        # Verify attention sums to 1.0 (allowing small floating point error)
        total_attention = sum(attention_distribution.values())
        if abs(total_attention - 1.0) > 1e-5:
            logger.warning(
                f"Attention distribution doesn't sum to 1.0: {total_attention}"
            )
        
        # Compute sufficiency metric
        max_attention = max(attention_distribution.values())
        max_chunk_id = max(attention_distribution, key=attention_distribution.get)
        
        # Determine if threshold is met
        if self.strict_mode:
            passes = max_attention > self.threshold
        else:
            passes = max_attention >= self.threshold
        
        # Build gate info
        gate_info = {
            "passes": passes,
            "max_attention": round(max_attention, 4),
            "max_chunk_id": max_chunk_id,
            "evidence_count": len(evidence_chunks),
            "threshold": self.threshold,
        }
        
        if passes:
            # Add margin info for successful evaluations
            gate_info["margin"] = round(max_attention - self.threshold, 4)
            gate_info["status"] = "evidence_sufficient"
            
            logger.info(
                f"Evidence gate PASSED: max_attention={max_attention:.2%}, "
                f"margin={gate_info['margin']:.2%}"
            )
        else:
            # Add deficit and refusal info for failed evaluations
            gate_info["deficit"] = round(self.threshold - max_attention, 4)
            gate_info["status"] = "evidence_insufficient"
            
            # Create structured refusal
            refusal = self._create_refusal(
                max_attention=max_attention,
                threshold=self.threshold,
                evidence_count=len(evidence_chunks),
                evidence_chunks=evidence_chunks,
                query=query,
                answer=answer
            )
            gate_info["refusal"] = refusal
            
            logger.warning(
                f"Evidence gate FAILED: max_attention={max_attention:.2%}, "
                f"deficit={gate_info['deficit']:.2%}"
            )
        
        # Apply gating decision
        if self.enable_gating:
            final_passes = passes
        else:
            # If gating disabled, always pass but still track metrics
            final_passes = True
            gate_info["gating_bypassed"] = True
            if not passes:
                logger.info(
                    "Evidence gate BYPASSED (gating disabled): "
                    f"max_attention={max_attention:.2%}"
                )
        
        return final_passes, gate_info

    def _create_refusal(
        self,
        max_attention: float,
        threshold: float,
        evidence_count: int,
        evidence_chunks: List[Dict[str, Any]],
        query: str,
        answer: str
    ) -> Dict[str, Any]:
        """
        Create a structured refusal message when evidence is insufficient.
        
        Args:
            max_attention: Highest attention weight achieved
            threshold: Required threshold
            evidence_count: Number of evidence chunks retrieved
            evidence_chunks: The actual chunks
            query: User's query
            answer: Generated answer (not returned to user)
        
        Returns:
            Structured refusal dict with explanation and suggestions
        """
        refusal = {
            "status": "insufficient_evidence",
            "reason": (
                f"No legal provision received sufficient attention support. "
                f"Highest attention was {max_attention:.1%} but threshold is {threshold:.1%}."
            ),
            "max_attention": round(max_attention, 4),
            "required_threshold": threshold,
            "deficit": round(threshold - max_attention, 4),
            "evidence_retrieved": evidence_count,
        }
        
        # Add evidence summary for transparency
        if evidence_chunks:
            evidence_summary = []
            for chunk in evidence_chunks[:3]:  # Top 3 chunks
                evidence_summary.append({
                    "chunk_id": chunk.get("chunk_id", "unknown"),
                    "statute": chunk.get("metadata", {}).get("statute", "unknown"),
                    "section": chunk.get("metadata", {}).get("section", "unknown"),
                })
            refusal["evidence_summary"] = evidence_summary
        
        # Add suggestions
        refusal["suggestions"] = [
            "Rephrase your query to be more specific",
            "Expand the legal corpus with more relevant statutes",
            "Break down complex questions into simpler sub-questions",
            "Provide additional context about jurisdiction or relevant laws"
        ]
        
        # Add explanatory note
        refusal["explanation"] = (
            "LEXAR safety mechanism: This refusal indicates that the retrieved "
            "evidence was not sufficiently focused on any single legal provision "
            "to provide a reliable answer. This prevents hallucination and ensures "
            "all answers are grounded in specific legal text."
        )
        
        return refusal

    def get_threshold(self) -> float:
        """Get current threshold value."""
        return self.threshold

    def set_threshold(self, threshold: float) -> None:
        """
        Set a new threshold value.
        
        Args:
            threshold: New threshold in [0.0, 1.0]
        
        Raises:
            ValueError: If threshold not in valid range
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")
        
        old_threshold = self.threshold
        self.threshold = threshold
        logger.info(
            f"Evidence gate threshold updated: {old_threshold:.2f} → {threshold:.2f}"
        )

    def enable(self) -> None:
        """Enable evidence gating."""
        self.enable_gating = True
        logger.info("Evidence gate ENABLED")

    def disable(self) -> None:
        """Disable evidence gating (for testing/comparison)."""
        self.enable_gating = False
        logger.warning("Evidence gate DISABLED - bypassing sufficiency checks")

    def is_enabled(self) -> bool:
        """Check if gating is currently enabled."""
        return self.enable_gating


class EvidenceGatingStats:
    """
    Track evidence gating statistics across queries.
    
    Useful for monitoring:
    - How often gate rejects answers
    - Distribution of max_attention values
    - Threshold effectiveness
    """

    def __init__(self):
        """Initialize statistics tracker."""
        self.total_evaluations = 0
        self.passed_evaluations = 0
        self.failed_evaluations = 0
        self.attention_values = []
        self.queries_rejected = []

    def record(self, passes: bool, max_attention: float, query: str) -> None:
        """Record an evaluation result."""
        self.total_evaluations += 1
        
        if passes:
            self.passed_evaluations += 1
        else:
            self.failed_evaluations += 1
            self.queries_rejected.append({
                "query": query,
                "max_attention": max_attention
            })
        
        self.attention_values.append(max_attention)

    def get_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if self.total_evaluations == 0:
            return {
                "total_evaluations": 0,
                "no_data": True
            }
        
        pass_rate = self.passed_evaluations / self.total_evaluations
        avg_attention = sum(self.attention_values) / len(self.attention_values)
        
        return {
            "total_evaluations": self.total_evaluations,
            "passed": self.passed_evaluations,
            "failed": self.failed_evaluations,
            "pass_rate": round(pass_rate, 4),
            "avg_max_attention": round(avg_attention, 4),
            "min_max_attention": round(min(self.attention_values), 4),
            "max_max_attention": round(max(self.attention_values), 4),
            "recently_rejected": self.queries_rejected[-5:]  # Last 5
        }

    def reset(self) -> None:
        """Reset all statistics."""
        self.total_evaluations = 0
        self.passed_evaluations = 0
        self.failed_evaluations = 0
        self.attention_values = []
        self.queries_rejected = []
