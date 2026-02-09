"""
Citation-Aware Output Rendering using Token-Level Provenance

Groups consecutive generated tokens that share the same supporting chunk
into citation spans and renders inline or footnote-style citations.

Constraints:
- Do NOT guess citations; derive strictly from provenance + evidence metadata
- Do NOT merge spans across different chunks
- If provenance ambiguous, mark span as uncertain
"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CitationSpan:
    start_idx: int
    end_idx: int
    chunk_id: str
    text_span: str
    statute: Optional[str] = None
    section: Optional[str] = None
    confidence: float = 0.0
    uncertain: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text_span": self.text_span,
            "statute": self.statute,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "confidence": round(self.confidence, 4),
            "uncertain": self.uncertain,
            "start": self.start_idx,
            "end": self.end_idx,
        }


class CitationRenderer:
    def __init__(self, confidence_threshold: float = 0.3):
        # Below this confidence, mark spans as uncertain
        self.confidence_threshold = confidence_threshold

    def _chunk_meta_map(self, evidence_chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        return {
            c.get("chunk_id", "unknown"): c.get("metadata", {}) for c in evidence_chunks
        }

    def build_spans(
        self,
        token_provenances: List[Dict[str, Any]],
        gen_ids: List[int],
        tokenizer,
        evidence_chunks: List[Dict[str, Any]],
    ) -> List[CitationSpan]:
        """Aggregate token provenance into contiguous citation spans.
        - Break spans when supporting chunk changes
        - Attach statute/section strictly from evidence metadata
        - Mark uncertain if confidence is low or chunk_id unknown
        """
        if not token_provenances:
            return []

        chunk_meta = self._chunk_meta_map(evidence_chunks)

        spans: List[CitationSpan] = []
        # Helper to decode span
        def decode_span(start: int, end: int) -> str:
            # gen_ids correspond to generated token ids (decoder positions)
            slice_ids = gen_ids[start : end + 1]
            return tokenizer.decode(slice_ids, skip_special_tokens=True).strip()

        # Initialize first span
        current_chunk = token_provenances[0]["supporting_chunk"]
        current_start = 0
        conf_accum = [token_provenances[0].get("confidence", 0.0)]

        for i in range(1, len(token_provenances)):
            prov = token_provenances[i]
            chunk_id = prov.get("supporting_chunk", "unknown")
            conf_accum.append(prov.get("confidence", 0.0))

            if chunk_id != current_chunk:
                # Close previous span
                end_idx = i - 1
                text = decode_span(current_start, end_idx)
                meta = chunk_meta.get(current_chunk, {})
                statute = meta.get("statute") or meta.get("code")
                section = meta.get("section")
                avg_conf = sum(conf_accum[:-1]) / max(len(conf_accum[:-1]), 1)
                uncertain = (
                    avg_conf < self.confidence_threshold or current_chunk == "unknown"
                )
                spans.append(
                    CitationSpan(
                        start_idx=current_start,
                        end_idx=end_idx,
                        chunk_id=current_chunk,
                        text_span=text,
                        statute=statute,
                        section=section,
                        confidence=avg_conf,
                        uncertain=uncertain,
                    )
                )
                # Start new span
                current_chunk = chunk_id
                current_start = i
                conf_accum = [prov.get("confidence", 0.0)]

        # Close final span
        end_idx = len(token_provenances) - 1
        text = decode_span(current_start, end_idx)
        meta = chunk_meta.get(current_chunk, {})
        statute = meta.get("statute") or meta.get("code")
        section = meta.get("section")
        avg_conf = sum(conf_accum) / max(len(conf_accum), 1)
        uncertain = (avg_conf < self.confidence_threshold or current_chunk == "unknown")
        spans.append(
            CitationSpan(
                start_idx=current_start,
                end_idx=end_idx,
                chunk_id=current_chunk,
                text_span=text,
                statute=statute,
                section=section,
                confidence=avg_conf,
                uncertain=uncertain,
            )
        )

        return spans

    def render_inline(self, spans: List[CitationSpan]) -> str:
        """Construct an annotated answer string by concatenating spans and appending inline citations.
        Example: "... text ... (IPC §302) ..."
        If span is uncertain or missing statute/section, annotate as "(uncertain: chunk_id)".
        """
        parts: List[str] = []
        for s in spans:
            citation = None
            if s.statute and s.section and not s.uncertain:
                citation = f"({s.statute} §{s.section})"
            elif s.statute and s.section and s.uncertain:
                citation = f"({s.statute} §{s.section}, uncertain)"
            else:
                citation = f"(uncertain: {s.chunk_id})"
            parts.append(f"{s.text_span} {citation}".strip())
        # Add spaces between parts
        return " ".join(p.strip() for p in parts).strip()

    def render_footnotes(self, spans: List[CitationSpan]) -> List[Dict[str, Any]]:
        """Return structured footnote-style citations list."""
        return [s.to_dict() for s in spans]
