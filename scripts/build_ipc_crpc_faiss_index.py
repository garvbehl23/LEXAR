"""
Build a combined FAISS index over IPC + CrPC statute chunks.

Requirements satisfied:
- Load ipc_chunks.json and crpc_chunks.json.
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

INDEX_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc.index")
CHUNK_IDS_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc_chunk_ids.json")


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


def main():
    os.makedirs(os.path.join(ROOT, "data", "faiss_index"), exist_ok=True)

    print(f"Loading IPC chunks from {IPC_CHUNKS_PATH} ...")
    ipc_chunks = load_chunks(IPC_CHUNKS_PATH)
    print(f"Loaded {len(ipc_chunks)} IPC chunks")

    print(f"Loading CrPC chunks from {CRPC_CHUNKS_PATH} ...")
    crpc_chunks = load_chunks(CRPC_CHUNKS_PATH)
    print(f"Loaded {len(crpc_chunks)} CrPC chunks")

    # Concatenate without modifying chunk contents
    chunks = ipc_chunks + crpc_chunks
    print(f"Combined total chunks: {len(chunks)}")

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
