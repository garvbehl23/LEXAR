from __future__ import annotations

import faiss
import numpy as np

from lexar.retrieval.embedder import LegalEmbedder


class UserRetriever:
    """Retriever for user-uploaded documents using in-memory FAISS index."""

    def __init__(self, chunks: list[dict]):
        self.embedder = LegalEmbedder()
        self.chunks = chunks

        if not chunks:
            self.index = None
            return

        texts = [c.get("text", "") for c in chunks]
        embeddings = self.embedder.embed_texts(texts).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # IP = cosine on normalized vectors
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or not self.chunks:
            return []

        q_emb = self.embedder.embed_query(query).astype("float32")
        q_emb = np.expand_dims(q_emb, axis=0)

        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(q_emb, k)

        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
