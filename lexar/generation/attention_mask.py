"""
Evidence-Constrained Attention Mask Builder

This module constructs hard binary attention masks that enforce the LEXAR principle:
"The decoder may attend ONLY to retrieved legal chunks and the user query."

The mask operates at the token level:
- Evidence tokens (from retrieved chunks): mask value = 0.0
- Query tokens: mask value = 0.0
- All other tokens (parametric memory, padding): mask value = -∞

This mask is ADDED to the attention logits BEFORE softmax, ensuring impossible
(zero probability) attention to non-evidence tokens.

Mathematical Definition:
    E_ij = {
        0       if token j ∈ R(q) ∪ {query}
        -∞      otherwise
    }

Where:
    E_ij = evidence mask at position (i, j)
    R(q) = retrieved evidence set for query q
    i = query position
    j = key/value position

The mask is combined with causal mask to prevent future-token attention.
"""

import torch
from typing import List, Dict, Tuple


class EvidenceTokenizer:
    """
    Maps chunk provenance to token positions.
    Tracks which tokens in the concatenated sequence came from evidence.
    """

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: HuggingFace tokenizer (e.g., T5Tokenizer)
        """
        self.tokenizer = tokenizer

    def tokenize_evidence(self, evidence_chunks: List[Dict]) -> Tuple[str, torch.Tensor]:
        """
        Tokenize evidence chunks and build token-to-chunk mapping.

        Args:
            evidence_chunks: List of dicts with keys:
                - "text": str - the evidence text
                - "chunk_id": str - unique identifier (for debugging)
                - "metadata": dict - statute, section, jurisdiction, etc.

        Returns:
            evidence_text: str - concatenated evidence
            evidence_token_mask: torch.Tensor of shape (num_tokens,)
                - True if token is from evidence
                - False if token is padding/separator
        """
        if not evidence_chunks:
            return "", torch.tensor([], dtype=torch.bool)

        evidence_texts = []
        token_boundaries = []  # Track where each chunk starts/ends

        for chunk in evidence_chunks:
            text = chunk.get("text", "")
            if text.strip():
                # Format: [CHUNK_ID] text
                chunk_id = chunk.get("chunk_id", "unknown")
                formatted = f"[{chunk_id}] {text}"
                evidence_texts.append(formatted)
                token_boundaries.append((len(evidence_texts) - 1, formatted))

        evidence_text = "\n\n".join(evidence_texts)

        # Tokenize to get exact token count
        tokens = self.tokenizer(evidence_text, return_tensors=None)["input_ids"]

        # All tokens from evidence are valid
        evidence_token_mask = torch.ones(len(tokens), dtype=torch.bool)

        return evidence_text, evidence_token_mask

    def tokenize_query(self, query: str) -> Tuple[str, torch.Tensor]:
        """
        Tokenize query and mark all tokens as evidence-allowed.

        Args:
            query: str - the user's question

        Returns:
            query_text: str
            query_token_mask: torch.Tensor of shape (num_tokens,)
                - True for all tokens (query is always in evidence set)
        """
        tokens = self.tokenizer(query, return_tensors=None)["input_ids"]
        query_token_mask = torch.ones(len(tokens), dtype=torch.bool)
        return query, query_token_mask


class AttentionMaskBuilder:
    """
    Constructs hard binary attention masks that enforce evidence constraints.
    """

    def __init__(self, pad_token_id: int = 0):
        """
        Args:
            pad_token_id: ID of padding token (assumed to be non-evidence)
        """
        self.pad_token_id = pad_token_id

    def build_evidence_mask(
        self,
        evidence_token_mask: torch.Tensor,
        query_token_mask: torch.Tensor,
        seq_length: int,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Build hard binary evidence mask.

        CORE LOGIC:
        - Evidence tokens (from chunks): can attend to other evidence + query
        - Query tokens: can attend to themselves + evidence
        - Generated tokens: can attend to evidence + query + previous generated
        - Non-evidence tokens: -∞ (impossible attention)

        Args:
            evidence_token_mask: torch.Tensor of shape (num_evidence_tokens,)
                - True if token is evidence, False otherwise
            query_token_mask: torch.Tensor of shape (num_query_tokens,)
                - True for all tokens (query is always allowed)
            seq_length: int - total sequence length in generation (evidence + query + generated)
            device: str - device for mask tensor

        Returns:
            evidence_mask: torch.Tensor of shape (seq_length, seq_length)
                - 0.0 if position j can attend to position i
                - -∞ if position j cannot attend to position i
        """
        evidence_mask = torch.full(
            (seq_length, seq_length),
            float("-inf"),
            device=device,
            dtype=torch.float32
        )

        # Indices of evidence + query tokens
        num_evidence = evidence_token_mask.shape[0]
        num_query = query_token_mask.shape[0]

        evidence_indices = torch.arange(num_evidence, device=device)
        query_indices = torch.arange(num_query, device=device) + num_evidence

        # Evidence tokens can attend to evidence + query
        evidence_mask[evidence_indices, :][:, evidence_indices] = 0.0
        evidence_mask[evidence_indices, :][:, query_indices] = 0.0

        # Query tokens can attend to evidence + query + themselves
        evidence_mask[query_indices, :][:, evidence_indices] = 0.0
        evidence_mask[query_indices, :][:, query_indices] = 0.0

        # Generated tokens (after evidence + query) can ONLY attend to evidence + query
        generated_start = num_evidence + num_query
        if seq_length > generated_start:
            generated_indices = torch.arange(
                generated_start, seq_length, device=device
            )
            # Generated tokens can attend to evidence + query
            evidence_mask[generated_indices, :][:, evidence_indices] = 0.0
            evidence_mask[generated_indices, :][:, query_indices] = 0.0

        return evidence_mask

    def combine_with_causal_mask(
        self,
        evidence_mask: torch.Tensor,
        seq_length: int,
        device: str = "cpu"
    ) -> torch.Tensor:
        """
        Combine evidence mask with causal mask (future tokens hidden).

        Standard causal (triangular) mask prevents attending to future tokens.
        We apply this in addition to evidence mask via element-wise minimum:
            combined = min(evidence_mask, causal_mask)

        Since -∞ < 0, if EITHER mask forbids attention, the combined mask forbids it.

        Args:
            evidence_mask: torch.Tensor of shape (seq_length, seq_length)
            seq_length: int - total sequence length
            device: str - device for mask tensor

        Returns:
            combined_mask: torch.Tensor of shape (seq_length, seq_length)
                - Combines evidence + causal constraints
        """
        # Build causal mask: lower triangular matrix (can attend to past/present)
        causal_mask = torch.tril(
            torch.zeros(seq_length, seq_length, device=device, dtype=torch.float32)
        ).fill_diagonal_(0.0)

        # Set forbidden positions (upper triangular) to -∞
        causal_mask = causal_mask.masked_fill(
            torch.triu(torch.ones(seq_length, seq_length, device=device, dtype=torch.bool), diagonal=1),
            float("-inf")
        )

        # Combine: if either mask forbids, combined forbids
        # Since -∞ is absorbing: max(evidence_mask, causal_mask)
        combined_mask = torch.maximum(evidence_mask, causal_mask)

        return combined_mask

    def build_full_mask(
        self,
        evidence_token_mask: torch.Tensor,
        query_token_mask: torch.Tensor,
        generated_seq_length: int = 1,
        device: str = "cpu",
        use_causal: bool = True
    ) -> torch.Tensor:
        """
        Build the complete attention mask for decoder forward pass.

        This is the primary API for mask construction.

        Args:
            evidence_token_mask: torch.Tensor of shape (num_evidence_tokens,)
            query_token_mask: torch.Tensor of shape (num_query_tokens,)
            generated_seq_length: int - length of generated sequence so far (default 1 for first token)
            device: str - device for mask tensor
            use_causal: bool - whether to apply causal masking (default True)

        Returns:
            attn_mask: torch.Tensor of shape (total_seq_length, total_seq_length)
                Ready to add to attention logits before softmax.

        Example:
            evidence_chunks = [{"text": "Section 302...", "chunk_id": "IPC_302"}]
            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            evidence_text, evidence_mask = tokenize_evidence(evidence_chunks)
            
            query = "What is murder?"
            query_text, query_mask = tokenize_query(query)
            
            attn_mask = mask_builder.build_full_mask(
                evidence_mask, query_mask, generated_seq_length=1, device="cuda"
            )
            
            # In forward pass:
            # logits = scale(Q @ K.T)
            # logits = logits + attn_mask  # Add -∞ to forbidden positions
            # attn = softmax(logits)  # Softmax ignores -∞
        """
        num_evidence = evidence_token_mask.shape[0]
        num_query = query_token_mask.shape[0]
        total_seq_length = num_evidence + num_query + generated_seq_length

        evidence_mask = self.build_evidence_mask(
            evidence_token_mask, query_token_mask, total_seq_length, device
        )

        if use_causal:
            full_mask = self.combine_with_causal_mask(evidence_mask, total_seq_length, device)
        else:
            full_mask = evidence_mask

        return full_mask


class ProvenanceTracker:
    """
    Tracks token provenance for interpretability and debugging.
    Maps generated tokens back to their source evidence.
    """

    def __init__(self):
        self.provenance_map = {}  # token_position -> (chunk_id, metadata)

    def record_evidence_tokens(
        self,
        evidence_chunks: List[Dict],
        tokenizer,
        start_idx: int = 0
    ) -> int:
        """
        Record which tokens came from which evidence chunks.

        Args:
            evidence_chunks: List of evidence dicts
            tokenizer: HuggingFace tokenizer
            start_idx: Starting token index in sequence

        Returns:
            next_idx: The token index where this evidence ends
        """
        current_idx = start_idx
        for chunk in evidence_chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            metadata = chunk.get("metadata", {})
            text = chunk.get("text", "")

            tokens = tokenizer(text, return_tensors=None)["input_ids"]
            for i, token_id in enumerate(tokens):
                self.provenance_map[current_idx + i] = {
                    "chunk_id": chunk_id,
                    "metadata": metadata,
                    "token_id": token_id
                }
            current_idx += len(tokens)

        return current_idx

    def record_query_tokens(
        self,
        query: str,
        tokenizer,
        start_idx: int = 0
    ) -> int:
        """
        Record query tokens as special provenance.

        Args:
            query: Query string
            tokenizer: HuggingFace tokenizer
            start_idx: Starting token index

        Returns:
            next_idx: The token index where query ends
        """
        tokens = tokenizer(query, return_tensors=None)["input_ids"]
        for i, token_id in enumerate(tokens):
            self.provenance_map[start_idx + i] = {
                "chunk_id": "QUERY",
                "metadata": {"type": "user_query"},
                "token_id": token_id
            }
        return start_idx + len(tokens)

    def get_provenance(self, token_idx: int) -> Dict:
        """
        Look up which chunk/query a token came from.

        Args:
            token_idx: Token position in sequence

        Returns:
            Provenance dict or {"chunk_id": "GENERATED"} if outside tracked range
        """
        return self.provenance_map.get(
            token_idx,
            {"chunk_id": "GENERATED", "metadata": {}}
        )

    def trace_generation(self, generated_token_ids: List[int], tokenizer) -> List[Dict]:
        """
        Trace generated tokens back to evidence or identify as parametric.

        Args:
            generated_token_ids: List of token IDs from generation
            tokenizer: HuggingFace tokenizer

        Returns:
            List of dicts showing token -> provenance mapping
        """
        trace = []
        current_idx = len(self.provenance_map)  # Start after known tokens

        for token_id in generated_token_ids:
            provenance = self.get_provenance(current_idx)
            token_str = tokenizer.decode([token_id])

            trace.append({
                "token": token_str,
                "token_id": token_id,
                "provenance": provenance
            })
            current_idx += 1

        return trace
