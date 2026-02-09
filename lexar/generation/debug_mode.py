"""
Evidence-Debug Mode: Attention Weight Analysis

This module provides utilities for understanding which evidence chunks contributed
to the generated answer by analyzing attention weights across decoder layers.

CORE FEATURE:
Extract and aggregate attention weights from the model to compute:
1. Per-chunk attention distribution (how much each chunk contributed)
2. Layer-wise attention patterns (which layers attended to which chunks)
3. Token-level attention traces (which tokens attended to which chunks)

This enables:
- Debugging: Why did the model generate that claim?
- Auditing: Which evidence supported this answer?
- Interpretability: What did each decoder layer focus on?
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class AttentionWeightExtractor:
    """
    Extract and process attention weights from decoder layers.
    """

    def __init__(self):
        self.layer_attentions = []  # List of attention tensors per layer
        self.token_count = 0
        self.chunk_boundaries = {}  # {chunk_id: (start_idx, end_idx)}

    def register_layer_attention(self, attention_weights: torch.Tensor, layer_idx: int):
        """
        Register attention weights from a decoder layer.

        Args:
            attention_weights: (batch_size, num_heads, seq_length, seq_length)
            layer_idx: Which decoder layer this came from
        """
        self.layer_attentions.append({
            "layer": layer_idx,
            "weights": attention_weights.detach().cpu().numpy()
        })

    def set_chunk_boundaries(self, chunk_id: str, start_idx: int, end_idx: int):
        """
        Record token boundaries for each evidence chunk.

        Args:
            chunk_id: Identifier for chunk (e.g., "IPC_302")
            start_idx: Token index where chunk starts
            end_idx: Token index where chunk ends
        """
        self.chunk_boundaries[chunk_id] = (start_idx, end_idx)

    def compute_chunk_attention_distribution(
        self,
        evidence_chunks: List[Dict],
        generated_start_idx: int
    ) -> Dict[str, float]:
        """
        Compute how much attention was paid to each evidence chunk during generation.

        Args:
            evidence_chunks: List of evidence dicts with chunk_id
            generated_start_idx: Where generated tokens start in sequence

        Returns:
            {chunk_id: attention_weight} normalized to sum to 1.0
        """
        if not self.layer_attentions:
            return {}

        chunk_attention = defaultdict(float)

        # Average attention weights across all layers
        for layer_data in self.layer_attentions:
            weights = layer_data["weights"]  # (batch, heads, seq, seq)

            if weights.size == 0:
                continue

            # For each generated token (query position >= generated_start_idx)
            # sum attention to evidence positions
            for gen_idx in range(generated_start_idx, weights.shape[2]):
                # Attention from this generated token to all other positions
                gen_attention = weights[0, :, gen_idx, :].mean(axis=0)  # Average over heads

                # Sum attention to each chunk's tokens
                for chunk_id, (start, end) in self.chunk_boundaries.items():
                    if start < end:
                        chunk_attention[chunk_id] += gen_attention[start:end].sum()

        # Normalize to [0, 1]
        total = sum(chunk_attention.values())
        if total > 0:
            chunk_attention = {k: v / total for k, v in chunk_attention.items()}

        return dict(chunk_attention)

    def compute_layer_wise_attention(
        self,
        evidence_chunks: List[Dict],
        generated_start_idx: int
    ) -> Dict[int, Dict[str, float]]:
        """
        Compute attention per chunk for each decoder layer.

        Returns:
            {layer_idx: {chunk_id: attention_weight}}
        """
        layer_attention = {}

        for layer_data in self.layer_attentions:
            layer_idx = layer_data["layer"]
            weights = layer_data["weights"]

            if weights.size == 0:
                continue

            chunk_attention = defaultdict(float)

            for gen_idx in range(generated_start_idx, weights.shape[2]):
                gen_attention = weights[0, :, gen_idx, :].mean(axis=0)

                for chunk_id, (start, end) in self.chunk_boundaries.items():
                    if start < end:
                        chunk_attention[chunk_id] += gen_attention[start:end].sum()

            # Normalize
            total = sum(chunk_attention.values())
            if total > 0:
                chunk_attention = {k: v / total for k, v in chunk_attention.items()}

            layer_attention[layer_idx] = dict(chunk_attention)

        return layer_attention

    def get_top_attended_chunks(
        self,
        chunk_attention: Dict[str, float],
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most attended chunks.

        Args:
            chunk_attention: {chunk_id: weight}
            top_k: Number of top chunks to return

        Returns:
            [(chunk_id, weight), ...] sorted by weight descending
        """
        return sorted(chunk_attention.items(), key=lambda x: x[1], reverse=True)[:top_k]


class DebugModeRenderer:
    """
    Render debug information in human-readable format.
    """

    @staticmethod
    def format_attention_distribution(chunk_attention: Dict[str, float]) -> str:
        """
        Format attention distribution for display.

        Args:
            chunk_attention: {chunk_id: weight}

        Returns:
            Formatted string showing attention percentages
        """
        lines = ["Attention Distribution:"]
        for chunk_id, weight in sorted(chunk_attention.items(), key=lambda x: x[1], reverse=True):
            percentage = weight * 100
            bar_length = int(percentage / 5)  # Scale to ~20 char width
            bar = "█" * bar_length + "░" * (20 - bar_length)
            lines.append(f"  {chunk_id:20s} │{bar}│ {percentage:5.1f}%")
        return "\n".join(lines)

    @staticmethod
    def format_layer_attention(layer_attention: Dict[int, Dict[str, float]]) -> str:
        """
        Format layer-wise attention for display.

        Args:
            layer_attention: {layer_idx: {chunk_id: weight}}

        Returns:
            Formatted string showing per-layer attention
        """
        lines = ["Layer-Wise Attention:"]
        for layer_idx in sorted(layer_attention.keys()):
            chunk_attn = layer_attention[layer_idx]
            lines.append(f"  Layer {layer_idx}:")
            for chunk_id, weight in sorted(chunk_attn.items(), key=lambda x: x[1], reverse=True):
                percentage = weight * 100
                lines.append(f"    {chunk_id:20s} {percentage:5.1f}%")
        return "\n".join(lines)

    @staticmethod
    def format_supporting_chunks(
        chunks: List[Dict],
        chunk_attention: Dict[str, float],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Format supporting chunks ranked by attention.

        Args:
            chunks: List of evidence chunks
            chunk_attention: {chunk_id: weight}
            top_k: How many to include

        Returns:
            List of top chunks with metadata and attention weight
        """
        chunk_map = {c.get("chunk_id"): c for c in chunks}
        top_chunk_ids = sorted(
            chunk_attention.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        supporting = []
        for chunk_id, weight in top_chunk_ids:
            if chunk_id in chunk_map:
                chunk = chunk_map[chunk_id].copy()
                chunk["attention_weight"] = float(weight)
                chunk["attention_percentage"] = float(weight * 100)
                supporting.append(chunk)

        return supporting


class DebugModeTracer:
    """
    Trace token generation with attention to evidence.
    """

    def __init__(self, chunk_boundaries: Dict[str, Tuple[int, int]], tokenizer):
        self.chunk_boundaries = chunk_boundaries
        self.tokenizer = tokenizer
        self.chunk_map = {v: k for k, v in chunk_boundaries.items()}  # idx → chunk_id

    def trace_token_attention(
        self,
        token_idx: int,
        attention_weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Trace which chunks a generated token attended to.

        Args:
            token_idx: Position of generated token in sequence
            attention_weights: (seq_length,) attention weights for this token

        Returns:
            {chunk_id: attention_weight}
        """
        attention_weights = attention_weights.detach().cpu().numpy()
        chunk_attention = defaultdict(float)

        for chunk_id, (start, end) in self.chunk_boundaries.items():
            if start < end:
                chunk_attention[chunk_id] = float(
                    attention_weights[start:end].sum()
                )

        # Normalize
        total = sum(chunk_attention.values())
        if total > 0:
            chunk_attention = {k: v / total for k, v in chunk_attention.items()}

        return dict(chunk_attention)

    def generate_token_trace(
        self,
        generated_token_ids: List[int],
        all_attention_weights: List[torch.Tensor]
    ) -> List[Dict]:
        """
        Generate trace for all generated tokens.

        Args:
            generated_token_ids: Token IDs that were generated
            all_attention_weights: List of attention weight tensors per token

        Returns:
            List of trace dicts with token, provenance, attention
        """
        trace = []

        for token_idx, token_id in enumerate(generated_token_ids):
            token_str = self.tokenizer.decode([token_id])

            if token_idx < len(all_attention_weights):
                chunk_attn = self.trace_token_attention(token_idx, all_attention_weights[token_idx])
                top_chunk = max(chunk_attn.items(), key=lambda x: x[1])[0] if chunk_attn else "UNKNOWN"
            else:
                chunk_attn = {}
                top_chunk = "UNKNOWN"

            trace.append({
                "token": token_str,
                "token_id": int(token_id),
                "primary_chunk": top_chunk,
                "chunk_attention": chunk_attn
            })

        return trace


def create_debug_result(
    answer: str,
    evidence_chunks: List[Dict],
    chunk_attention: Dict[str, float],
    layer_attention: Dict[int, Dict[str, float]],
    provenance: Dict = None
) -> Dict:
    """
    Create a comprehensive debug result dict.

    Args:
        answer: Generated answer string
        evidence_chunks: List of evidence chunks used
        chunk_attention: {chunk_id: attention_weight}
        layer_attention: {layer_idx: {chunk_id: weight}}
        provenance: Optional token-level provenance

    Returns:
        Complete debug result dict
    """
    renderer = DebugModeRenderer()

    return {
        "answer": answer,
        "debug": {
            "attention_distribution": chunk_attention,
            "layer_wise_attention": layer_attention,
            "supporting_chunks": renderer.format_supporting_chunks(
                evidence_chunks,
                chunk_attention,
                top_k=3
            ),
            "attention_visualization": renderer.format_attention_distribution(chunk_attention),
            "layer_visualization": renderer.format_layer_attention(layer_attention),
        },
        "evidence_count": len(evidence_chunks),
        "top_attended_chunk": max(chunk_attention.items(), key=lambda x: x[1])[0] if chunk_attention else None,
        "provenance": provenance or {}
    }
