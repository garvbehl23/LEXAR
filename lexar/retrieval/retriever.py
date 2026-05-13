"""Legacy DenseRetriever — used by evaluation scripts and frontend eval tab."""
from __future__ import annotations

import os
import faiss
import numpy as np

from lexar.retrieval.embedder import LegalEmbedder


class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict],
        index_path: str = "data/faiss_index/ipc.index",
    ):
        self.chunks = chunks
        self.index_path = index_path
        self.embedder = LegalEmbedder()
        self.index = None

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self._build_index()

    def _build_index(self):
        texts = [c.get("text", "") for c in self.chunks]
        embeddings = self.embedder.embed_texts(texts).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None:
            return []

        q_emb = self.embedder.embed_query(query).astype("float32")
        q_emb = np.expand_dims(q_emb, axis=0)

        k = min(top_k, len(self.chunks))
        scores, ids = self.index.search(q_emb, k)

        results = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)
        return results
