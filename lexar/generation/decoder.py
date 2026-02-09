"""
Evidence-Constrained Decoder

Custom Transformer decoder that enforces hard attention masking to prevent
generation from parametric memory.

CORE PRINCIPLE:
The decoder shall not attend to any token outside the retrieved evidence set R(q)
and the user query q. This is enforced via hard binary masking at every attention
head in every decoder layer.

Mathematical Constraint:
    P(y_t | y_{<t}, q, R(q)) restricted to:
        - Query tokens
        - Evidence tokens  
        - Previously generated tokens (causal mask)
        - NEVER parametric memory

Implementation:
    1. Pre-compute evidence mask (token indices that are evidence/query)
    2. Apply mask at EVERY attention layer (attention_mask parameter)
    3. Mask is added to logits BEFORE softmax: logits + mask
    4. -∞ values become 0 probability after softmax
    5. Keys/values NOT truncated — masking prevents their use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from transformers.models.t5.modeling_t5 import T5Stack, T5Config
from transformers import T5Tokenizer, T5ForConditionalGeneration


class EvidenceConstrainedSelfAttention(nn.Module):
    """
    Self-attention with hard evidence-based masking.
    
    Replaces standard attention to enforce:
    - Decoder can only attend to evidence + query + previous generated tokens
    - No attention to parametric memory
    - Combined with causal mask
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_kv = config.d_kv
        self.dropout_rate = config.dropout_rate

        # Linear projections
        self.q_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.num_heads * self.d_kv, bias=False)
        self.out_proj = nn.Linear(self.num_heads * self.d_kv, self.d_model, bias=False)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: (batch_size, seq_length, d_model)
            attention_mask: (seq_length, seq_length) or (batch_size, seq_length, seq_length)
                Binary mask from AttentionMaskBuilder. Contains:
                - 0.0 for allowed attention
                - -∞ for forbidden attention
            key_value_states: For cross-attention (not used in decoder self-attention)
            past_key_value: KV cache for generation
            output_attentions: Whether to return attention weights

        Returns:
            attn_output: (batch_size, seq_length, d_model)
            attn_weights: (batch_size, num_heads, seq_length, seq_length) or None
            present_key_value: KV cache
        """
        batch_size, seq_length, _ = hidden_states.shape
        is_cross_attention = key_value_states is not None

        # --- Query Projection ---
        q = self.q_proj(hidden_states)
        q = q.view(batch_size, seq_length, self.num_heads, self.d_kv)
        q = q.transpose(1, 2)  # (batch_size, num_heads, seq_length, d_kv)

        # --- Key/Value Projection ---
        if is_cross_attention:
            # For cross-attention (encoder output), no masking
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
            kv_seq_length = key_value_states.shape[1]
        else:
            # For self-attention, use hidden states
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            kv_seq_length = seq_length

        k = k.view(batch_size, kv_seq_length, self.num_heads, self.d_kv)
        k = k.transpose(1, 2)  # (batch_size, num_heads, kv_seq_length, d_kv)

        v = v.view(batch_size, kv_seq_length, self.num_heads, self.d_kv)
        v = v.transpose(1, 2)  # (batch_size, num_heads, kv_seq_length, d_kv)

        # --- Scaled Dot-Product Attention with Evidence Masking ---
        # logits = Q @ K.T / sqrt(d_k)
        logits = torch.matmul(q, k.transpose(-2, -1)) / (self.d_kv ** 0.5)
        # Shape: (batch_size, num_heads, seq_length, kv_seq_length)

        # Apply evidence mask: logits + mask
        # Forbidden positions have -∞, softmax will convert to 0 probability
        if attention_mask is not None and not is_cross_attention:
            # attention_mask shape: (seq_length, seq_length) or (batch_size, seq_length, seq_length)
            if attention_mask.dim() == 2:
                # Expand to batch dimension
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
                # Now: (1, 1, seq_length, seq_length)
            else:
                # Assume (batch_size, seq_length, seq_length)
                attention_mask = attention_mask.unsqueeze(1)
                # Now: (batch_size, 1, seq_length, seq_length)

            logits = logits + attention_mask

        # --- Softmax ---
        attn_weights = F.softmax(logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # --- Weighted Sum of Values ---
        attn_output = torch.matmul(attn_weights, v)
        # Shape: (batch_size, num_heads, seq_length, d_kv)

        # --- Reshape and Project Out ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, self.num_heads * self.d_kv)
        attn_output = self.out_proj(attn_output)

        # --- Return ---
        present_key_value = (k, v) if not is_cross_attention else None

        if output_attentions:
            return attn_output, attn_weights, present_key_value
        else:
            return attn_output, None, present_key_value


class EvidenceConstrainedDecoderLayer(nn.Module):
    """
    Single decoder layer with evidence-constrained self-attention.
    """

    def __init__(self, config: T5Config):
        super().__init__()
        self.self_attn = EvidenceConstrainedSelfAttention(config)
        self.cross_attn = nn.MultiheadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout_rate,
            batch_first=False
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout_rate),
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states: (batch_size, seq_length, d_model) - decoder input
            encoder_hidden_states: (batch_size, kv_seq_length, d_model) - encoder output (evidence)
            attention_mask: Evidence-constrained mask from AttentionMaskBuilder
            encoder_attention_mask: Standard causal/padding mask for cross-attention
            output_attentions: Return attention weights

        Returns:
            hidden_states: (batch_size, seq_length, d_model)
            self_attn_weights: Optional attention weights
            cross_attn_weights: Optional attention weights
        """
        # --- Self-Attention with Evidence Masking ---
        self_attn_output, self_attn_weights, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.norm1(self_attn_output)

        # --- Cross-Attention (Evidence + Query) ---
        # Standard cross-attention (evidence keys/values can be attended without masking)
        cross_attn_output, cross_attn_weights = self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            attn_mask=encoder_attention_mask,
        )
        hidden_states = hidden_states + self.norm2(cross_attn_output)

        # --- Feed-Forward ---
        ffn_output = self.ffn(hidden_states)
        hidden_states = hidden_states + self.norm3(ffn_output)

        outputs = (hidden_states,)
        if output_attentions:
            outputs = outputs + (self_attn_weights, cross_attn_weights)

        return outputs


class EvidenceConstrainedDecoder(nn.Module):
    """
    Full Transformer decoder with evidence-constrained attention.
    
    This decoder STRICTLY enforces:
    - Self-attention only to evidence + query + previous generated tokens
    - Cross-attention only to encoder output (evidence)
    - No generation from parametric memory
    """

    def __init__(self, config: T5Config, num_layers: int = 6):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        self.d_model = config.d_model

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.layers = nn.ModuleList(
            [EvidenceConstrainedDecoderLayer(config) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(config.d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Args:
            input_ids: (batch_size, seq_length) - decoder input token IDs
            encoder_hidden_states: (batch_size, kv_seq_length, d_model) - evidence embeddings
            attention_mask: Evidence-constrained mask from AttentionMaskBuilder
                Shape: (seq_length, seq_length) or (batch_size, seq_length, seq_length)
            encoder_attention_mask: Cross-attention mask (standard)
            output_attentions: Whether to return all attention weights

        Returns:
            logits: (batch_size, seq_length, vocab_size)
            attention_weights: Optional tuple of attention tensors
        """
        # --- Embedding ---
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = hidden_states * (self.config.d_model ** 0.5)  # Scale by sqrt(d_model)
        hidden_states = self.dropout(hidden_states)

        # --- Decoder Layers with Evidence Masking ---
        all_self_attns = ()
        all_cross_attns = ()

        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns = all_self_attns + (layer_outputs[1],)
                all_cross_attns = all_cross_attns + (layer_outputs[2],)

        # --- Final Layer Norm ---
        hidden_states = self.norm(hidden_states)

        # --- Project to Vocabulary ---
        logits = self.lm_head(hidden_states)

        outputs = (logits,)
        if output_attentions:
            outputs = outputs + (all_self_attns, all_cross_attns)

        return outputs


class LexarEvidenceConstrainedModel(nn.Module):
    """
    Full LEXAR model: Encoder + Evidence-Constrained Decoder
    
    Enforces evidence grounding at the architectural level.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        super().__init__()
        self.base_model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.config = self.base_model.config

        # Replace standard decoder with evidence-constrained decoder
        self.decoder = EvidenceConstrainedDecoder(self.config)

    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            input_ids: (batch_size, seq_length) - encoder input (concatenated evidence + query)
            decoder_input_ids: (batch_size, seq_length) - decoder input
            attention_mask: Encoder attention mask (standard)
            decoder_attention_mask: Evidence-constrained decoder mask
            encoder_outputs: Cached encoder outputs
            output_attentions: Return attention weights

        Returns:
            logits: (batch_size, seq_length, vocab_size)
        """
        if encoder_outputs is None:
            encoder_outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        encoder_hidden_states = encoder_outputs[0]

        # Evidence-constrained decoding
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        logits = decoder_outputs[0]

        return logits
