def attach_citations(answer: str, evidence_chunks: list[dict]) -> str:
    """
    Attach primary and supporting legal citations.
    """

    if not evidence_chunks:
        return answer

    primary = evidence_chunks[0]["metadata"].get("section")
    supporting = {
        c["metadata"].get("section")
        for c in evidence_chunks[1:]
        if c.get("metadata", {}).get("section")
    }

    citation_text = f"[Primary: IPC §{primary}]"
    if supporting:
        citation_text += f" [Supporting: {', '.join(f'IPC §{s}' for s in sorted(supporting))}]"

    return f"{answer}\n\n{citation_text}"
