"""
Build a FAISS index over IPC statute chunks produced by ingest_india_statutes.py.

Requirements satisfied:
- Load chunks from data/processed_docs/ipc_chunks.json without mutating text.
- Use existing encoder g_ψ (LegalEmbedder wraps SentenceTransformer all-MiniLM-L6-v2) with
  normalized embeddings for cosine/IP search.
- Build deterministic FAISS IndexFlatIP (no randomness) and persist to disk.
- Persist chunk_id order mapping to reload without re-encoding.
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


CHUNKS_PATH = os.path.join(ROOT, "data", "processed_docs", "ipc_chunks.json")
INDEX_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc.index")
CHUNK_IDS_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc_chunk_ids.json")


def load_chunks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_chunks(chunks: List[dict], embedder: LegalEmbedder) -> np.ndarray:
    # Include section titles (when present) to keep semantic hints like "Punishment for murder" while
    # preserving the original chunk text as-is.
    texts = []
    for c in chunks:
        title = c.get("metadata", {}).get("section_title")
        if title:
            texts.append(f"{title}\n{c['text']}")
        else:
            texts.append(c["text"])
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


def main():
    os.makedirs(os.path.join(ROOT, "data", "faiss_index"), exist_ok=True)

    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks")

    embedder = LegalEmbedder()
    print("Encoding chunks with g_ψ (SentenceTransformer all-MiniLM-L6-v2, normalized) ...")
    vectors = encode_chunks(chunks, embedder)

    print("Building FAISS IndexFlatIP (deterministic)...")
    index = build_index(vectors)

    print(f"Writing index to {INDEX_PATH} ...")
    faiss.write_index(index, INDEX_PATH)

    print(f"Writing chunk_id mapping to {CHUNK_IDS_PATH} ...")
    save_chunk_ids(chunks, CHUNK_IDS_PATH)

    print(f"Done. Index entries: {index.ntotal}")


if __name__ == "__main__":
    main()
