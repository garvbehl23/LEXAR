def attach_citations(answer: str, evidence_chunks: list[dict]) -> str:
    """Append primary and supporting citation tags to the answer."""
    if not evidence_chunks:
        return answer

    def _meta(c: dict) -> dict:
        return c.get("metadata") or c.get("meta") or {}

    primary_meta = _meta(evidence_chunks[0])
    primary = primary_meta.get("section", "")
    statute = primary_meta.get("statute", "IPC")

    supporting = {
        _meta(c).get("section", "")
        for c in evidence_chunks[1:]
        if _meta(c).get("section")
    }
    supporting.discard(primary)

    if not primary:
        return answer

    citation_text = f"[Primary: {statute} {primary}]"
    if supporting:
        citation_text += " [Supporting: " + ", ".join(sorted(supporting)) + "]"

    return f"{answer}\n\n{citation_text}"
