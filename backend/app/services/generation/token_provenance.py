"""
Token-Level Provenance Tracking for LEXAR

Maps each generated token to the evidence chunk(s) it attended to.

Definition: prov(y_t) = arg max_{c_i ∈ R̃(q)} Σ_{j∈c_i} α_{t,j}

Where:
- y_t: Generated token at position t
- α_{t,j}: Attention weight from y_t to input position j
- c_i: Evidence chunk i
- prov(y_t): Primary supporting chunk for token y_t
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TokenProvenance:
    """Provenance information for a single generated token."""
    
    token: str
    """The generated token text"""
    
    position: int
    """Position in the generated sequence (0-indexed)"""
    
    supporting_chunk: str
    """Primary chunk that this token attended to most (chunk_id)"""
    
    attention_mass: float
    """Attention mass assigned to the supporting chunk (0.0-1.0, 4 decimals)"""
    
    secondary_chunks: List[Tuple[str, float]] = field(default_factory=list)
    """Secondary chunks with significant attention [(chunk_id, attention_mass), ...]
    Includes chunks with attention > secondary_threshold (default: 0.05)"""
    
    layer_distributions: Optional[Dict[str, float]] = field(default=None)
    """Per-layer attention distribution if multi_layer=True
    Format: {"layer_0": {"chunk_id": mass}, "layer_1": {...}}"""
    
    confidence: float = 0.0
    """Confidence score: primary_attention / sum(all_chunk_attentions)
    High confidence (>0.7) = strong evidence support
    Low confidence (<0.3) = distributed attention across chunks"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "token": self.token,
            "position": self.position,
            "supporting_chunk": self.supporting_chunk,
            "attention_mass": round(self.attention_mass, 4),
            "confidence": round(self.confidence, 4),
        }
        
        if self.secondary_chunks:
            result["secondary_chunks"] = [
                {
                    "chunk_id": chunk_id,
                    "attention_mass": round(attention, 4)
                }
                for chunk_id, attention in self.secondary_chunks
            ]
        
        if self.layer_distributions:
            result["layer_distributions"] = {
                layer_name: {
                    chunk_id: round(mass, 4)
                    for chunk_id, mass in layer_dist.items()
                }
                for layer_name, layer_dist in self.layer_distributions.items()
            }
        
        return result


@dataclass
class ProvenanceStats:
    """Statistics about provenance tracking."""
    
    total_tokens: int = 0
    """Total tokens tracked"""
    
    high_confidence_tokens: int = 0
    """Tokens with confidence >= 0.7"""
    
    medium_confidence_tokens: int = 0
    """Tokens with 0.3 <= confidence < 0.7"""
    
    low_confidence_tokens: int = 0
    """Tokens with confidence < 0.3"""
    
    chunk_coverage: Dict[str, int] = field(default_factory=dict)
    """How many tokens were supported by each chunk"""
    
    avg_confidence: float = 0.0
    """Average confidence across all tokens"""
    
    primary_layer_used: Optional[int] = None
    """Which layer contributed most to provenance (if multi-layer tracking)"""
    
    def calculate(self, provenances: List[TokenProvenance]):
        """Calculate statistics from provenance list."""
        if not provenances:
            return
        
        self.total_tokens = len(provenances)
        self.high_confidence_tokens = 0
        self.medium_confidence_tokens = 0
        self.low_confidence_tokens = 0
        self.chunk_coverage = {}
        
        confidence_sum = 0.0
        
        for prov in provenances:
            confidence_sum += prov.confidence
            
            if prov.confidence >= 0.7:
                self.high_confidence_tokens += 1
            elif prov.confidence >= 0.3:
                self.medium_confidence_tokens += 1
            else:
                self.low_confidence_tokens += 1
            
            # Track chunk coverage
            chunk_id = prov.supporting_chunk
            self.chunk_coverage[chunk_id] = self.chunk_coverage.get(chunk_id, 0) + 1
        
        self.avg_confidence = confidence_sum / len(provenances) if provenances else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_tokens": self.total_tokens,
            "high_confidence_tokens": self.high_confidence_tokens,
            "medium_confidence_tokens": self.medium_confidence_tokens,
            "low_confidence_tokens": self.low_confidence_tokens,
            "confidence_distribution": {
                "high": f"{self.high_confidence_tokens}/{self.total_tokens}",
                "medium": f"{self.medium_confidence_tokens}/{self.total_tokens}",
                "low": f"{self.low_confidence_tokens}/{self.total_tokens}"
            },
            "chunk_coverage": self.chunk_coverage,
            "avg_confidence": round(self.avg_confidence, 4),
            "primary_layer": self.primary_layer_used,
        }


class TokenProvenanceTracker:
    """Tracks token-level provenance during decoding.
    
    For each generated token, maps it to the evidence chunk it attended to.
    Uses actual attention weights from the model (no string matching).
    """
    
    def __init__(
        self,
        token_ids_to_chunk_ids: Dict[int, str],
        secondary_threshold: float = 0.05,
        track_multi_layer: bool = False,
        enable_tracking: bool = True
    ):
        """Initialize provenance tracker.
        
        Args:
            token_ids_to_chunk_ids: Maps input token position to chunk ID
                {0: "IPC_302", 1: "IPC_302", 2: "IPC_34", ...}
            secondary_threshold: Minimum attention to include secondary chunks (default 0.05)
            track_multi_layer: If True, track per-layer distributions
            enable_tracking: If False, disable tracking (for comparison/testing)
        """
        self.token_ids_to_chunk_ids = token_ids_to_chunk_ids
        self.secondary_threshold = secondary_threshold
        self.track_multi_layer = track_multi_layer
        self.enable_tracking = enable_tracking
        
        # Storage for attention data
        self.layer_attention_weights: Dict[int, np.ndarray] = {}
        # Format: {layer_idx: array of shape (seq_len, input_len)}
        
        self.attention_heads: Dict[int, np.ndarray] = {}
        # Format: {layer_idx: array of shape (num_heads, seq_len, input_len)}
        
        self.generated_tokens: List[str] = []
        self.token_provenances: List[TokenProvenance] = []
        self.stats = ProvenanceStats()
        
        logger.info(
            f"TokenProvenanceTracker initialized with {len(token_ids_to_chunk_ids)} "
            f"input tokens, multi_layer={track_multi_layer}, enabled={enable_tracking}"
        )
    
    def track_layer_attention(
        self,
        layer_idx: int,
        attention_weights: np.ndarray,
        attention_heads: Optional[np.ndarray] = None
    ):
        """Record attention weights for a single layer.
        
        Args:
            layer_idx: Which decoder layer (0, 1, 2, ...)
            attention_weights: Array of shape (batch_size, seq_len, input_len)
                or (seq_len, input_len) for single batch
            attention_heads: Optional array of shape (num_heads, seq_len, input_len)
                For detailed per-head analysis
        """
        if not self.enable_tracking:
            return
        
        # Ensure 2D (remove batch dimension if needed)
        if attention_weights.ndim == 3:
            attention_weights = attention_weights[0]  # First in batch
        
        if attention_heads is not None and attention_heads.ndim == 4:
            attention_heads = attention_heads[0]  # First in batch
        
        self.layer_attention_weights[layer_idx] = attention_weights
        if attention_heads is not None:
            self.attention_heads[layer_idx] = attention_heads
        
        logger.debug(
            f"Layer {layer_idx}: tracked attention {attention_weights.shape}, "
            f"heads: {attention_heads.shape if attention_heads is not None else 'None'}"
        )
    
    def record_token(self, token: str):
        """Record a generated token for provenance mapping.
        
        Args:
            token: The generated token (or subword/BPE token)
        """
        if not self.enable_tracking:
            return
        
        self.generated_tokens.append(token)
    
    def compute_provenances(self) -> List[TokenProvenance]:
        """Compute provenance for all recorded tokens.
        
        Returns:
            List of TokenProvenance objects
        """
        if not self.enable_tracking:
            return []
        
        if not self.generated_tokens:
            logger.warning("No tokens recorded for provenance computation")
            return []
        
        if not self.layer_attention_weights:
            logger.warning("No attention weights recorded for provenance computation")
            # Fallback: assign all tokens to the first chunk in the mapping to preserve determinism
            chunk_order = []
            for cid in self.token_ids_to_chunk_ids.values():
                if cid not in chunk_order:
                    chunk_order.append(cid)
            fallback_chunk = chunk_order[0] if chunk_order else None
            if fallback_chunk:
                self.token_provenances = [
                    TokenProvenance(
                        token=tok,
                        position=idx,
                        supporting_chunk=fallback_chunk,
                        attention_mass=1.0,
                        confidence=1.0,
                        secondary_chunks=[],
                        layer_distributions=None,
                    )
                    for idx, tok in enumerate(self.generated_tokens)
                ]
                self.stats.calculate(self.token_provenances)
                return self.token_provenances
            return []
        
        self.token_provenances = []
        
        # Get average attention across layers (if multiple layers tracked)
        attention_weights = self._aggregate_layer_attention()
        
        num_generated_tokens = attention_weights.shape[0]
        num_input_tokens = attention_weights.shape[1]
        token_count = len(self.generated_tokens)
        
        if num_generated_tokens != token_count:
            logger.warning(
                f"Mismatch: {num_generated_tokens} attention positions "
                f"but {token_count} tokens"
            )
            seq_len = min(num_generated_tokens, token_count)
            attention_weights = attention_weights[:seq_len, :]
            self.generated_tokens = self.generated_tokens[:seq_len]
            num_generated_tokens = seq_len
            token_count = seq_len
        
        # For each generated token, compute provenance
        for token_idx, token in enumerate(self.generated_tokens):
            if token_idx >= num_generated_tokens:
                logger.warning(f"Token index {token_idx} exceeds attention shape")
                break
            
            # Get attention for this token (shape: [num_input_tokens])
            token_attention = attention_weights[token_idx, :]
            
            # Aggregate attention by chunk
            chunk_attention = self._aggregate_attention_by_chunk(token_attention)
            
            # Find primary supporting chunk
            primary_chunk, primary_attention = self._get_primary_chunk(chunk_attention)
            
            # Get secondary chunks
            secondary_chunks = self._get_secondary_chunks(
                chunk_attention, primary_chunk
            )
            
            # Compute confidence
            total_attention = sum(chunk_attention.values())
            confidence = (
                primary_attention / total_attention 
                if total_attention > 0 
                else 0.0
            )
            
            # Get per-layer distributions if tracking multi-layer
            layer_distributions = None
            if self.track_multi_layer:
                layer_distributions = self._get_layer_distributions(token_idx)
            
            # Create provenance record
            prov = TokenProvenance(
                token=token,
                position=token_idx,
                supporting_chunk=primary_chunk,
                attention_mass=primary_attention,
                secondary_chunks=secondary_chunks,
                layer_distributions=layer_distributions,
                confidence=confidence,
            )
            
            self.token_provenances.append(prov)
        
        # Calculate statistics
        self.stats.calculate(self.token_provenances)
        
        logger.info(
            f"Computed provenance for {len(self.token_provenances)} tokens. "
            f"Avg confidence: {self.stats.avg_confidence:.3f}"
        )
        
        return self.token_provenances
    
    def _aggregate_layer_attention(self) -> np.ndarray:
        """Average attention weights across all tracked layers.
        
        Returns:
            Array of shape (seq_len, input_len) with averaged attention
        """
        if len(self.layer_attention_weights) == 1:
            # Single layer - return as is
            return list(self.layer_attention_weights.values())[0]
        
        # Multiple layers - average them
        layer_indices = sorted(self.layer_attention_weights.keys())
        layers = [self.layer_attention_weights[idx] for idx in layer_indices]
        
        # Stack and mean
        stacked = np.stack(layers, axis=0)  # (num_layers, seq_len, input_len)
        averaged = np.mean(stacked, axis=0)  # (seq_len, input_len)
        
        # Re-normalize (softmax gave us normalized weights, averaging preserves this)
        return averaged
    
    def _aggregate_attention_by_chunk(self, token_attention: np.ndarray) -> Dict[str, float]:
        """Aggregate token attention weights by chunk.
        
        Args:
            token_attention: Attention weights for one token (shape: [num_input_tokens])
        
        Returns:
            Dictionary mapping chunk_id to aggregated attention mass
        """
        chunk_attention = {}
        
        for token_pos, attention_weight in enumerate(token_attention):
            # Find which chunk this input token belongs to
            chunk_id = self.token_ids_to_chunk_ids.get(token_pos, "unknown")
            
            # Add to chunk's total attention
            if chunk_id not in chunk_attention:
                chunk_attention[chunk_id] = 0.0
            chunk_attention[chunk_id] += float(attention_weight)
        
        return chunk_attention
    
    def _get_primary_chunk(
        self, chunk_attention: Dict[str, float]
    ) -> Tuple[str, float]:
        """Find the chunk with highest attention.
        
        Args:
            chunk_attention: Attention mass per chunk
        
        Returns:
            (primary_chunk_id, primary_attention_mass)
        """
        if not chunk_attention:
            return "unknown", 0.0
        
        primary = max(chunk_attention.items(), key=lambda x: x[1])
        return primary[0], primary[1]
    
    def _get_secondary_chunks(
        self,
        chunk_attention: Dict[str, float],
        primary_chunk: str
    ) -> List[Tuple[str, float]]:
        """Get secondary chunks with attention > threshold (excluding primary).
        
        Args:
            chunk_attention: Attention mass per chunk
            primary_chunk: The primary supporting chunk to exclude
        
        Returns:
            List of (chunk_id, attention_mass) tuples, sorted by attention (descending)
        """
        secondary = [
            (chunk_id, attention)
            for chunk_id, attention in chunk_attention.items()
            if chunk_id != primary_chunk and attention > self.secondary_threshold
        ]
        
        # Sort by attention (descending)
        secondary.sort(key=lambda x: x[1], reverse=True)
        
        return secondary
    
    def _get_layer_distributions(self, token_idx: int) -> Dict[str, Dict[str, float]]:
        """Get per-layer chunk attention distributions.
        
        Args:
            token_idx: Position in generated sequence
        
        Returns:
            Dict mapping layer names to chunk attention distributions
        """
        distributions = {}
        
        for layer_idx in sorted(self.layer_attention_weights.keys()):
            attention_weights = self.layer_attention_weights[layer_idx]
            
            if token_idx >= attention_weights.shape[0]:
                continue
            
            token_attention = attention_weights[token_idx, :]
            chunk_attention = self._aggregate_attention_by_chunk(token_attention)
            
            layer_name = f"layer_{layer_idx}"
            distributions[layer_name] = chunk_attention
        
        return distributions if distributions else None
    
    def get_provenances(self) -> List[TokenProvenance]:
        """Get computed provenances (compute if not already done)."""
        if not self.token_provenances and self.generated_tokens:
            self.compute_provenances()
        return self.token_provenances
    
    def get_stats(self) -> ProvenanceStats:
        """Get provenance statistics."""
        return self.stats
    
    def get_provenance_dict(self) -> Dict[str, Any]:
        """Get all provenance data as dictionary (for JSON serialization)."""
        provenances = self.get_provenances()
        
        return {
            "tokens": len(self.generated_tokens),
            "provenances": [p.to_dict() for p in provenances],
            "statistics": self.stats.to_dict(),
            "summary": {
                "total_tokens": len(provenances),
                "avg_confidence": self.stats.avg_confidence,
                "primary_chunks": list(self.stats.chunk_coverage.keys()),
                "tracking_enabled": self.enable_tracking,
                "multi_layer": self.track_multi_layer,
            }
        }
    
    def get_answer_with_provenance(self, answer_text: str) -> Dict[str, Any]:
        """Generate answer with inline provenance markers.
        
        Args:
            answer_text: The generated answer
        
        Returns:
            Dictionary with answer_with_provenance and token_provenances
        """
        provenances = self.get_provenances()
        
        # Build provenance lookup
        token_to_prov = {p.position: p for p in provenances}
        
        result = {
            "answer": answer_text,
            "token_provenances": [p.to_dict() for p in provenances],
            "has_provenance": len(provenances) > 0,
        }
        
        # Add summary of provenance quality
        if provenances:
            result["provenance_quality"] = {
                "avg_confidence": round(self.stats.avg_confidence, 4),
                "high_confidence_ratio": (
                    self.stats.high_confidence_tokens / len(provenances)
                    if provenances else 0.0
                ),
            }
        
        return result
    
    def reset(self):
        """Reset tracker for new generation."""
        self.layer_attention_weights.clear()
        self.attention_heads.clear()
        self.generated_tokens.clear()
        self.token_provenances.clear()
        self.stats = ProvenanceStats()
        logger.debug("TokenProvenanceTracker reset")


def create_token_to_chunk_mapping(
    evidence_chunks: List[Dict[str, Any]],
    tokenizer,
    query_text: Optional[str] = None,
) -> Dict[int, str]:
    """Create mapping from input token positions to chunk IDs.
    
    Mirrors the formatting used by EvidenceTokenizer + LexarGenerator:
    evidence_text = "\n\n".join(f"[{chunk_id}] {text}")
    combined_text = evidence_text + "\n\n[QUESTION]\n" + query
    
    Args:
        evidence_chunks: List of evidence chunks with 'chunk_id' and 'text'
        tokenizer: Tokenizer to count tokens per chunk
        query_text: Optional query string to map to chunk id "query"
    
    Returns:
        Dictionary mapping token position to chunk_id
    """
    token_to_chunk = {}
    current_pos = 0
    sep_tokens = tokenizer.encode("\n\n", add_special_tokens=False)

    for chunk in evidence_chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        text = chunk.get("text", "")
        formatted = f"[{chunk_id}] {text}"

        tokens = tokenizer.encode(formatted, add_special_tokens=False)
        for i in range(len(tokens)):
            token_to_chunk[current_pos + i] = chunk_id
        current_pos += len(tokens)

        # Map separator tokens to the same chunk to avoid unknown gaps
        for i in range(len(sep_tokens)):
            token_to_chunk[current_pos + i] = chunk_id
        current_pos += len(sep_tokens)

    # Map [QUESTION] prefix and query tokens to "query"
    question_prefix = tokenizer.encode("[QUESTION]\n", add_special_tokens=False)
    for i in range(len(question_prefix)):
        token_to_chunk[current_pos + i] = "query"
    current_pos += len(question_prefix)

    if query_text:
        query_tokens = tokenizer.encode(query_text, add_special_tokens=False)
        for i in range(len(query_tokens)):
            token_to_chunk[current_pos + i] = "query"
        current_pos += len(query_tokens)

    return token_to_chunk
