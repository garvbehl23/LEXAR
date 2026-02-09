"""
Build a combined FAISS index over IPC + CrPC + IEA statute chunks.

Requirements satisfied:
- Load ipc_chunks.json, crpc_chunks.json, and iea_1872_chunks.json.
- Concatenate chunks without modifying text.
- Use existing encoder g_ψ (LegalEmbedder wraps SentenceTransformer all-MiniLM-L6-v2).
- Build deterministic FAISS IndexFlatIP (no randomness).
- Preserve statute metadata and chunk IDs (by keeping chunks intact and saving chunk_id order).
"""

import json
import os
import sys
from typing import List

import faiss
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.app.services.retrieval.embedder import LegalEmbedder

IPC_CHUNKS_PATH = os.path.join(ROOT, "data", "processed_docs", "ipc_chunks.json")
CRPC_CHUNKS_PATH = os.path.join(ROOT, "data", "processed_docs", "crpc_chunks.json")
IEA_CHUNKS_PATH = os.path.join(ROOT, "data", "processed_docs", "iea_1872_chunks.json")

INDEX_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc_iea.index")
CHUNK_IDS_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc_iea_chunk_ids.json")


def load_chunks(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_chunks(chunks: List[dict], embedder: LegalEmbedder) -> np.ndarray:
    # Do NOT modify text. Use raw chunk text as-is.
    texts = [c["text"] for c in chunks]
    return embedder.embed_texts(texts).astype("float32")


def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_chunk_ids(chunks: List[dict], path: str):
    chunk_ids = [c.get("chunk_id") for c in chunks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False, indent=2)


def test_retrieval(chunks: List[dict], index: faiss.IndexFlatIP, embedder: LegalEmbedder):
    """Test retrieval with a sample query."""
    query = "Is a confession to police admissible?"
    print(f"\n{'='*80}")
    print(f"VALIDATION: Testing retrieval for query: '{query}'")
    print(f"{'='*80}")
    
    # Encode query using base model (matching the index)
    q_emb = embedder.embed_query(query).astype("float32")
    q_emb = np.expand_dims(q_emb, axis=0)
    
    # Search
    scores, ids = index.search(q_emb, 5)
    
    print("\nTop-5 results:")
    for rank, (idx, score) in enumerate(zip(ids[0], scores[0]), 1):
        if idx == -1 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        chunk_id = chunk.get("chunk_id", "?")
        text_preview = chunk.get("text", "")[:120].replace("\n", " ")
        print(f"  {rank}. [{score:.4f}] {chunk_id}: {text_preview}...")
    
    # Check if expected IEA sections appear in top-5
    top_chunk_ids = [chunks[idx].get("chunk_id", "") for idx in ids[0] if idx != -1 and idx < len(chunks)]
    iea_sections = [cid for cid in top_chunk_ids if cid.startswith("iea-")]
    
    print(f"\nIEA sections in top-5: {iea_sections}")
    if any(cid.startswith("iea-") for cid in top_chunk_ids[:5]):
        print("✓ Validation passed: IEA sections found in top-5 results")
    else:
        print("⚠ Warning: No IEA sections in top-5 (may indicate retrieval issue)")


def main():
    os.makedirs(os.path.join(ROOT, "data", "faiss_index"), exist_ok=True)

    print(f"Loading IPC chunks from {IPC_CHUNKS_PATH} ...")
    ipc_chunks = load_chunks(IPC_CHUNKS_PATH)
    print(f"Loaded {len(ipc_chunks)} IPC chunks")

    print(f"Loading CrPC chunks from {CRPC_CHUNKS_PATH} ...")
    crpc_chunks = load_chunks(CRPC_CHUNKS_PATH)
    print(f"Loaded {len(crpc_chunks)} CrPC chunks")

    print(f"Loading IEA chunks from {IEA_CHUNKS_PATH} ...")
    iea_chunks = load_chunks(IEA_CHUNKS_PATH)
    print(f"Loaded {len(iea_chunks)} IEA chunks")

    # Concatenate without modifying chunk contents
    chunks = ipc_chunks + crpc_chunks + iea_chunks
    print(f"\nCombined total chunks: {len(chunks)}")
    print(f"  IPC:  {len(ipc_chunks)}")
    print(f"  CrPC: {len(crpc_chunks)}")
    print(f"  IEA:  {len(iea_chunks)}")

    # Initialize embedder (base model for chunk encoding)
    embedder = LegalEmbedder(use_finetuned_query_encoder=False)
    print("\nEncoding chunks with g_ψ (SentenceTransformer all-MiniLM-L6-v2, normalized) ...")
    vectors = encode_chunks(chunks, embedder)

    print("Building FAISS IndexFlatIP (deterministic)...")
    index = build_index(vectors)

    print(f"\nWriting index to {INDEX_PATH} ...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Writing chunk_id mapping to {CHUNK_IDS_PATH} ...")
    save_chunk_ids(chunks, CHUNK_IDS_PATH)

    print(f"\n✓ Index build complete. Index entries: {index.ntotal}")

    # Run validation test
    test_retrieval(chunks, index, embedder)

    print(f"\n{'='*80}")
    print("Build and validation complete.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
