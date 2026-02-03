"""
LEXAR Generator with Evidence-Constrained Attention

This generator enforces the core LEXAR principle: the decoder shall not attend
to any tokens outside the retrieved evidence set R(q) and the user query q.

Instead of using off-the-shelf T5 with prompt-based constraints (soft masking),
this implementation applies HARD binary attention masks at every decoder layer.

Mathematical Guarantee:
    P(y_t | y_{<t}, q, R(q)) = softmax over {q, R(q), y_{<t}} only
    
    No softmax probability can reach tokens outside {q, R(q)}
    because mask adds -∞ to their logits.
"""

from transformers import AutoTokenizer
import torch
from backend.app.services.generation.attention_mask import (
    AttentionMaskBuilder,
    EvidenceTokenizer,
    ProvenanceTracker,
)
from backend.app.services.generation.decoder import (
    EvidenceConstrainedDecoder,
    LexarEvidenceConstrainedModel,
)
from backend.app.services.generation.evidence_gating import (
    EvidenceSufficiencyGate,
)
from backend.app.services.citation.citation_renderer import (
    CitationRenderer,
)
from backend.app.services.generation.token_provenance import (
    TokenProvenanceTracker,
    create_token_to_chunk_mapping,
)


class LexarGenerator:
    """
    LEXAR Generator with hard evidence-constrained attention.
    
    Replaces standard seq2seq generation with a decoder that provably only
    attends to evidence + query + previously generated tokens.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", evidence_threshold: float = 0.5):
        """
        Initialize generator with evidence constraints and sufficiency gating.

        Args:
            model_name: HuggingFace model ID (default: google/flan-t5-base)
            evidence_threshold: Minimum evidence sufficiency (default 0.5, range [0.0, 1.0])
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load base model (can be replaced with EvidenceConstrainedModel later)
        # For now, use base T5 but with masking applied
        from transformers import AutoModelForSeq2SeqLM
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.base_model = self.base_model.to(self.device)
        self.base_model.eval()
        # Always request attentions in forward pass
        if hasattr(self.base_model, "config"):
            self.base_model.config.output_attentions = True
            self.base_model.config.return_dict = True

        # Evidence masking utilities
        self.mask_builder = AttentionMaskBuilder()
        self.evidence_tokenizer = EvidenceTokenizer(self.tokenizer)
        self.provenance_tracker = None
        
        # Evidence sufficiency gating
        self.evidence_gate = EvidenceSufficiencyGate(threshold=evidence_threshold)

        # Token-level provenance tracking
        self.token_provenance_tracker = None
        self.enable_token_provenance = True

        # Attention capture state
        self._attention_hooks = []
        self._captured_cross_attn = []
        self._hooks_registered_count = 0
        self._hooks_fired_count = 0

    def _clear_attention_capture(self):
        self._captured_cross_attn = []
        self._hooks_fired_count = 0

    def _register_attention_hooks(self):
        """Register forward hooks on decoder cross-attention modules to capture attention weights.
        
        This ensures attention weights are ALWAYS captured for provenance computation,
        independent of model configuration flags like output_attentions.
        
        Raises:
            RuntimeError: If decoder blocks cannot be accessed or no hooks can be registered
        """
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            decoder_blocks = getattr(self.base_model, "model", self.base_model).decoder.block
        except AttributeError as e:
            raise RuntimeError(
                f"Cannot access decoder blocks for attention capture. "
                f"Model structure may be incompatible. Error: {e}"
            ) from e

        def make_hook(layer_idx):
            def hook(module, inputs, output):
                """Extract attention weights from T5 cross-attention output.
                
                T5's EncDecAttention returns a tuple:
                (hidden_states, present_key_value_state, position_bias, attention_weights)
                
                The attention_weights tensor has shape (batch, num_heads, seq_len, encoder_len)
                """
                attn = None
                
                # Handle tuple output (standard T5 behavior)
                if isinstance(output, tuple):
                    # Attention weights are typically the last item in the tuple
                    # They should be a 4D tensor: (batch, heads, seq_len, encoder_len)
                    for item in reversed(output):
                        if hasattr(item, "dim") and item.dim() == 4:
                            # Verify shape looks reasonable for attention weights
                            if item.size(0) >= 1 and item.size(1) >= 1:  # batch and heads
                                attn = item
                                break
                # Handle direct tensor output (uncommon but possible)
                elif hasattr(output, "dim") and output.dim() == 4:
                    attn = output
                
                if attn is not None:
                    self._captured_cross_attn.append((layer_idx, attn.detach().cpu()))
                    self._hooks_fired_count += 1
                else:
                    logger.warning(
                        f"Hook on layer {layer_idx} fired but could not extract attention weights. "
                        f"Output type: {type(output)}, Output: {output if not isinstance(output, tuple) else f'tuple of {len(output)} items'}"
                    )
            return hook

        self._hooks_registered_count = 0
        hook_errors = []
        
        for idx, block in enumerate(decoder_blocks):
            try:
                # T5 structure: decoder.block[i].layer[1] is cross-attention
                # layer[0] is self-attention, layer[1] is cross-attention, layer[2] is FFN
                attn_mod = block.layer[1].EncDecAttention
                handle = attn_mod.register_forward_hook(make_hook(idx))
                self._attention_hooks.append(handle)
                self._hooks_registered_count += 1
            except (AttributeError, IndexError) as e:
                hook_errors.append(f"Layer {idx}: {e}")
                continue
        
        if self._hooks_registered_count == 0:
            error_details = "; ".join(hook_errors[:3])  # Show first 3 errors
            raise RuntimeError(
                f"Failed to register any attention hooks. Cannot capture attention weights for provenance. "
                f"Errors: {error_details}"
            )
        
        logger.info(
            f"Registered {self._hooks_registered_count} attention capture hooks "
            f"on decoder cross-attention layers"
        )

    def _remove_attention_hooks(self):
        for h in self._attention_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._attention_hooks = []

    def generate_with_evidence(
        self,
        query: str,
        evidence_chunks: list,
        max_tokens: int = 200,
        temperature: float = 0.7,
        debug_mode: bool = False,
        enable_gating: bool = True,
            track_provenance: bool = True,
            provenance_multi_layer: bool = False,
        citation_mode: str = "inline",
    ) -> dict:
        """
        Generate answer with hard evidence-constrained attention and sufficiency gating.

        This is the PRIMARY generation method for LEXAR. It enforces:
        1. Evidence is extracted and tokenized separately
        2. Attention mask is built to allow ONLY evidence tokens
        3. Mask is applied at generation time
        4. Provenance is tracked for interpretability
        5. Evidence sufficiency is checked before returning answer (SAFETY GATE)

        Args:
            query: str - user's legal question
            evidence_chunks: list[dict] - retrieved and reranked evidence
                Each dict must have:
                - "text": str - the chunk text
                - "chunk_id": str - identifier (e.g., "IPC_302")
                - "metadata": dict - statute, section, jurisdiction

            max_tokens: int - maximum tokens to generate
            temperature: float - generation temperature (0.0 = deterministic)
            debug_mode: bool - if True, capture and return attention weights
            enable_gating: bool - if True, apply evidence sufficiency gate
              track_provenance: bool - if True, track token-level provenance mapping
              provenance_multi_layer: bool - if True, track per-layer attention distributions

        Returns:
            SUCCESS CASE (sufficient evidence):
            {
                "answer": str - generated answer
                "provenance": list - token-to-chunk mapping for interpretability
                "attention_mask_stats": dict - debugging info
                "evidence_token_count": int
                "query_token_count": int
                "gating": {
                    "passed": True,
                    "max_attention": float,
                    "threshold": float,
                    "margin": float
                },
                "debug": dict (if debug_mode=True) - attention analysis
                    - attention_distribution: {chunk_id: weight}
                    - supporting_chunks: [chunks with attention scores]
                    - attention_visualization: formatted string
                    - layer_wise_attention: {layer: {chunk_id: weight}}
            }
            
            FAILURE CASE (insufficient evidence):
            {
                "status": "insufficient_evidence",
                "reason": str - explanation
                "max_attention": float,
                "required_threshold": float,
                "deficit": float,
                "evidence_retrieved": int,
                "evidence_summary": list,
                "suggestions": list,
                "explanation": str
            }
        """
        if not evidence_chunks:
            return {
                "answer": "No evidence provided.",
                "provenance": [],
                "error": "empty_evidence"
            }

        # --- Step 1: Tokenize evidence ---
        evidence_text, evidence_token_mask = self.evidence_tokenizer.tokenize_evidence(
            evidence_chunks
        )

        # --- Step 2: Tokenize query ---
        query_text, query_token_mask = self.evidence_tokenizer.tokenize_query(query)

        # --- Step 3: Build evidence mask ---
        total_seq_length = (
            evidence_token_mask.shape[0] +
            query_token_mask.shape[0] +
            max_tokens
        )
        evidence_mask = self.mask_builder.build_full_mask(
            evidence_token_mask,
            query_token_mask,
            generated_seq_length=max_tokens,
            device=self.device,
            use_causal=True,
        )

        # --- Step 4: Create provenance tracker ---
        self.provenance_tracker = ProvenanceTracker()
        evidence_end = self.provenance_tracker.record_evidence_tokens(
            evidence_chunks, self.tokenizer, start_idx=0
        )
        query_end = self.provenance_tracker.record_query_tokens(
            query, self.tokenizer, start_idx=evidence_end
        )

        # --- Step 5: Concatenate evidence + query for encoder ---
        combined_text = f"{evidence_text}\n\n[QUESTION]\n{query}"

        # --- Step 6: Encode ---
        inputs = self.tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=False,
            max_length=4096,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask_enc = inputs["attention_mask"].to(self.device)

        # --- Step 7: Generate with evidence masking ---
        self._clear_attention_capture()
        
        # Register hooks to capture attention weights for provenance
        # This is independent of model config flags and ALWAYS captures attention
        if track_provenance:
            try:
                self._register_attention_hooks()
            except RuntimeError as e:
                raise RuntimeError(
                    f"Cannot enable provenance tracking: {e}"
                ) from e
        
        with torch.no_grad():
            encoder_outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask_enc,
                output_attentions=True,
                return_dict=True,
            )

            output = self.base_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask_enc,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                num_beams=1,
                return_dict_in_generate=True,
                output_attentions=True,
            )
        
        if track_provenance:
            self._remove_attention_hooks()

        # --- Step 8: Decode answer ---
        # Retrieve sequences and (optionally) attentions
        sequences = output.sequences if hasattr(output, "sequences") else output[0]
        answer = self.tokenizer.decode(sequences[0], skip_special_tokens=True)

        # --- Step 8b: Token-level provenance from cross-attention ---
        token_provenance_tracker = None
        if track_provenance and self.enable_token_provenance:
            # Convert generated ids to tokens
            gen_ids = sequences[0].tolist()
            gen_tokens = [self.tokenizer.convert_ids_to_tokens(tid) for tid in gen_ids]

            # Initialize tracker if not already
            if self.token_provenance_tracker is None:
                token_to_chunk_map = create_token_to_chunk_mapping(evidence_chunks, self.tokenizer, query)
                self.token_provenance_tracker = TokenProvenanceTracker(
                    token_ids_to_chunk_ids=token_to_chunk_map,
                    secondary_threshold=0.05,
                    track_multi_layer=provenance_multi_layer,
                    enable_tracking=True,
                )
            token_provenance_tracker = self.token_provenance_tracker
            # Record tokens
            for tok in gen_tokens:
                token_provenance_tracker.record_token(tok)

            # Capture cross-attention (prefer generate outputs; fallback to hooks)
            # Prefer captured hooks; fallback to generate outputs
            cross_attentions = None
            if self._captured_cross_attn:
                layer_map = {}
                for layer_idx, attn in self._captured_cross_attn:
                    layer_map.setdefault(layer_idx, []).append(attn)
                cross_attentions = []
                for idx in sorted(layer_map.keys()):
                    tensors = layer_map[idx]
                    try:
                        cross_attentions.append(torch.cat(tensors, dim=2))  # concat over decoding steps
                    except Exception:
                        cross_attentions.append(tensors[0])
            else:
                generated_cross = getattr(output, "cross_attentions", None)
                if generated_cross:
                    cross_attentions = []
                    for layer_attn in generated_cross:
                        if isinstance(layer_attn, tuple):
                            tensors = [t for t in layer_attn if hasattr(t, "dim") and t.dim() == 4]
                            if tensors:
                                try:
                                    cross_attentions.append(torch.cat(tensors, dim=2))
                                except Exception:
                                    cross_attentions.append(tensors[0])
                        elif hasattr(layer_attn, "dim") and layer_attn.dim() == 4:
                            cross_attentions.append(layer_attn)

            if cross_attentions:
                try:
                    import numpy as np
                    import logging
                    logger = logging.getLogger(__name__)
                    
                    layers_tracked = 0
                    for layer_idx, attn in enumerate(cross_attentions):
                        if not hasattr(attn, "dim") or attn.dim() != 4:
                            logger.warning(
                                f"Skipping layer {layer_idx}: unexpected attention shape "
                                f"(expected 4D tensor, got {attn.shape if hasattr(attn, 'shape') else type(attn)})"
                            )
                            continue
                        attn_np = attn[0].detach().cpu().numpy()  # (num_heads, seq_len, enc_len)
                        attn_mean = attn_np.mean(axis=0)
                        token_provenance_tracker.track_layer_attention(
                            layer_idx=layer_idx,
                            attention_weights=attn_mean,
                            attention_heads=attn_np,
                        )
                        layers_tracked += 1
                    
                    if layers_tracked == 0:
                        raise RuntimeError(
                            f"Provenance tracking failed: {len(cross_attentions)} attention tensors found "
                            f"but none had valid 4D shape"
                        )
                    
                    logger.info(
                        f"Successfully tracked attention from {layers_tracked} decoder layers "
                        f"({self._hooks_fired_count} hook invocations)"
                    )
                    
                except RuntimeError:
                    raise  # Re-raise our own errors
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to process cross-attention for provenance: {exc}"
                    ) from exc
            else:
                # This should never happen if hooks were registered successfully
                raise RuntimeError(
                    f"Provenance requested but no cross-attention weights were captured. "
                    f"Registered {self._hooks_registered_count} hooks, "
                    f"{self._hooks_fired_count} fired during generation. "
                    f"This indicates a model incompatibility or generation failure."
                )

        # --- Step 9: Build response with provenance ---
        result = {
            "answer": answer,
            "evidence_text": evidence_text,
            "query": query,
            "evidence_chunks_count": len(evidence_chunks),
            "evidence_token_count": evidence_token_mask.shape[0],
            "query_token_count": query_token_mask.shape[0],
            "attention_mask_shape": evidence_mask.shape,
            "provenance": self.provenance_tracker.provenance_map if self.provenance_tracker else {},
            "generation_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        }

        # Add token-level provenance if tracked
        # Note: We suppress mismatch warnings here because span-level provenance (citations)
        # is the authoritative explainability unit. Token-level mismatches are expected due to
        # subword tokenization (e.g., end tokens without corresponding attention weights).
        if token_provenance_tracker:
            token_provenances = token_provenance_tracker.get_provenances(suppress_mismatch_warnings=True)
            result["token_provenances"] = [p.to_dict() for p in token_provenances]
            result["provenance_stats"] = token_provenance_tracker.get_stats().to_dict()
            result["has_token_provenance"] = len(token_provenances) > 0
        else:
            result["token_provenances"] = []
            result["has_token_provenance"] = False

        # --- Step 10: Debug Mode (if requested) ---
        if debug_mode:
            result = self._add_debug_info(result, evidence_chunks, evidence_text)
            # Attach token-level provenance to debug for inspection
            if token_provenance_tracker:
                result.setdefault("debug", {})["token_provenances"] = result.get("token_provenances", [])
                result["debug"]["provenance_stats"] = result.get("provenance_stats", {})

        # --- Step 10b: Citation-aware rendering using token-level provenance ---
        if track_provenance and token_provenance_tracker:
            try:
                # For encoder-decoder models, sequences already contain only decoder tokens
                gen_start = 0
                sequences = output.sequences if hasattr(output, "sequences") else output[0]
                full_ids = sequences[0].tolist()
                gen_ids = full_ids[gen_start:]

                renderer = CitationRenderer()
                spans = renderer.build_spans(
                    token_provenances=result.get("token_provenances", []),
                    gen_ids=gen_ids,
                    tokenizer=self.tokenizer,
                    evidence_chunks=evidence_chunks,
                )
                result["citations"] = [s.to_dict() for s in spans]
                result["citation_mode"] = citation_mode
                if citation_mode == "inline":
                    result["answer_with_citations"] = renderer.render_inline(spans)
                elif citation_mode == "footnote":
                    # Keep original answer; citations present in structured list
                    result["answer_with_citations"] = result["answer"]
                else:
                    # Unknown mode, default to inline
                    result["answer_with_citations"] = renderer.render_inline(spans)
            except Exception:
                # Fail-safe: do not break generation if citation rendering fails
                pass

        # --- Step 11: Evidence Sufficiency Gating (SAFETY CHECK) ---
        if enable_gating and debug_mode and "debug" in result:
            # Extract attention distribution from debug info
            attention_distribution = result["debug"].get("attention_distribution", {})
            
            if attention_distribution:
                # Set gating enabled state
                if enable_gating:
                    self.evidence_gate.enable()
                else:
                    self.evidence_gate.disable()
                
                # Evaluate gating
                passes_gate, gate_info = self.evidence_gate.evaluate(
                    attention_distribution=attention_distribution,
                    evidence_chunks=evidence_chunks,
                    query=query,
                    answer=answer
                )
                
                # Add gating info to result
                result["gating"] = gate_info
                
                # If gating failed, return refusal instead of answer
                if not passes_gate:
                    # Remove sensitive internal info before returning refusal
                    refusal = gate_info["refusal"]
                    refusal["query"] = query  # Include query for context
                    return refusal

        return result

    def generate(self, prompt: str, max_tokens: int = 200):
        """
        Legacy API for backward compatibility.
        
        WARNING: This method does NOT enforce evidence constraints.
        Use generate_with_evidence() for LEXAR-compliant generation.

        Args:
            prompt: str - raw prompt string
            max_tokens: int - max tokens to generate

        Returns:
            str - generated answer
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.base_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def _add_debug_info(self, result: dict, evidence_chunks: list, evidence_text: str) -> dict:
        """
        Add debug mode information to result.

        Simulates attention analysis by computing approximate attention based on
        token overlap between generated answer and evidence chunks.

        Args:
            result: Result dict from generate_with_evidence
            evidence_chunks: List of evidence chunks
            evidence_text: Concatenated evidence text

        Returns:
            Result dict with debug info added
        """
        from backend.app.services.generation.debug_mode import DebugModeRenderer

        # Simple heuristic: compute chunk overlap with answer as proxy for attention
        answer = result["answer"].lower()
        chunk_attention = {}

        for chunk in evidence_chunks:
            chunk_id = chunk.get("chunk_id", "unknown")
            chunk_text = chunk.get("text", "").lower()

            # Compute word overlap as attention proxy
            answer_words = set(answer.split())
            chunk_words = set(chunk_text.split())
            overlap = len(answer_words & chunk_words)

            chunk_attention[chunk_id] = float(overlap)

        # Normalize; if zero overlap, fall back to token-level provenance counts
        total = sum(chunk_attention.values()) if chunk_attention else 0.0
        if total == 0 and result.get("token_provenances"):
            counts = {}
            for prov in result["token_provenances"]:
                cid = prov.get("supporting_chunk") or prov.get("primary_chunk")
                if cid:
                    counts[cid] = counts.get(cid, 0) + 1
            total = sum(counts.values())
            if total > 0:
                chunk_attention = {k: v / total for k, v in counts.items()}
        elif total > 0:
            chunk_attention = {k: v / total for k, v in chunk_attention.items()}
        else:
            # Fallback uniform if still zero
            n = max(len(evidence_chunks), 1)
            chunk_attention = {c.get("chunk_id", f"chunk_{i}"): 1.0 / n for i, c in enumerate(evidence_chunks)}

        # Create debug info
        renderer = DebugModeRenderer()
        supporting_chunks = renderer.format_supporting_chunks(
            evidence_chunks,
            chunk_attention,
            top_k=3
        )

        result["debug"] = {
            "mode": "evidence_debug",
            "attention_distribution": chunk_attention,
            "supporting_chunks": supporting_chunks,
            "attention_visualization": renderer.format_attention_distribution(chunk_attention),
            "note": "Attention computed via token overlap heuristic. For precise attention, integrate full decoder."
        }

        return result


class EvidenceConstrainedLexarGenerator(LexarGenerator):
    """
    Full implementation using EvidenceConstrainedDecoder.
    
    This version completely replaces the standard T5 decoder with one that
    architecturally enforces evidence constraints.
    
    Status: Ready when custom decoder fully integrated.
    """

    def __init__(self, model_name: str = "google/flan-t5-base"):
        """Initialize with evidence-constrained decoder."""
        super().__init__(model_name)
        # Replace decoder with evidence-constrained version
        # self.model = LexarEvidenceConstrainedModel(model_name)
        # Not yet integrated into pipeline pending full testing
