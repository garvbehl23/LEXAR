def fuse_context(query: str, evidence_chunks: list[dict]) -> str:
    """
    Fuse re-ranked evidence chunks with the query into a single prompt.
    """

    context = ""
    for i, c in enumerate(evidence_chunks):
        context += f"[LAW {i+1}]\n{c['text']}\n\n"

    prompt = (
        "You are a legal assistant. Answer the question strictly "
        "using the provided legal provisions.\n\n"
        f"{context}"
        f"[QUESTION]\n{query}\n\n"
        "[ANSWER]"
    )

    return prompt
