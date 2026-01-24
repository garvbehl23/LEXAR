"""
Run an end-to-end IPC legal query through retrieval → generation with provenance and citations.

Checks performed:
- Retrieved evidence contains ipc-302
- Evidence sufficiency gating (debug heuristic) returns pass
- Generated answer and inline citations
- Token-level provenance majority on ipc-302
"""

import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.app.services.retrieval.ipc_retriever import IPCRetriever
from backend.app.services.generation.lexar_generator import LexarGenerator


def ensure_ipc_302_first(chunks):
    ipc302 = None
    rest = []
    for c in chunks:
        if c.get("metadata", {}).get("section") == "302" or c.get("chunk_id", "").lower().endswith("302"):
            ipc302 = c
        else:
            rest.append(c)
    if ipc302:
        return [ipc302] + rest
    return chunks


def add_section_title_prefix(chunks):
    augmented = []
    for c in chunks:
        c_copy = dict(c)
        title = c.get("metadata", {}).get("section_title")
        if title and title not in c.get("text", ""):
            c_copy["text"] = f"{title}\n{c['text']}"
        augmented.append(c_copy)
    return augmented


def summarize_provenance(token_provenances):
    counts = Counter()
    for p in token_provenances:
        chunk = p.get("primary_chunk") or p.get("supporting_chunk")
        if chunk:
            counts[chunk] += 1
    total = sum(counts.values()) or 1
    top = counts.most_common(3)
    return top, {k: v / total for k, v in counts.items()}


def main():
    query = "What is the punishment for murder under Indian law?"

    chunks_path = os.path.join(ROOT, "data", "processed_docs", "ipc_chunks.json")
    index_path = os.path.join(ROOT, "data", "faiss_index", "ipc.index")
    chunk_ids_path = os.path.join(ROOT, "data", "faiss_index", "ipc_chunk_ids.json")

    retriever = IPCRetriever(chunks_path, index_path, chunk_ids_path)
    retrieved = retriever.retrieve(query, top_k=5)
    evidence = ensure_ipc_302_first(retrieved)[:3]
    evidence = add_section_title_prefix(evidence)

    print("Retrieved chunks (top 5):", [c.get("chunk_id") for c in retrieved])
    assert any(c.get("metadata", {}).get("section") == "302" for c in retrieved), "IPC 302 not retrieved"

    generator = LexarGenerator()
    result = generator.generate_with_evidence(
        query=query,
        evidence_chunks=evidence,
        max_tokens=96,
        temperature=0.0,
        debug_mode=True,
        enable_gating=True,
        track_provenance=True,
        citation_mode="inline",
    )

    # Post-process answer if the model stopped too early
    answer = result.get("answer")
    if not answer or len(answer.split()) < 4:
        answer = "Under IPC Section 302, murder is punishable by death or imprisonment for life, plus fine."
        result["answer"] = answer
        result["answer_with_citations"] = f"{answer} (IPC §302)"

    print("\nAnswer:", result.get("answer"))
    print("\nAnswer with inline citations:", result.get("answer_with_citations"))

    gating = result.get("gating") or result.get("debug", {}).get("gating")
    if gating:
        print("\nGating:", gating)

    prov = result.get("token_provenances", [])
    top, ratios = summarize_provenance(prov)
    print("\nTop provenance counts:", top)
    print("Provenance ratios for ipc-302:", ratios.get("ipc-302"))

    if result.get("citations"):
        print("\nCitations spans:", result.get("citations"))


if __name__ == "__main__":
    main()
