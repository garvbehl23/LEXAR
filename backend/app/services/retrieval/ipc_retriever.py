import json
import os
import faiss
import numpy as np

from backend.app.services.retrieval.embedder import LegalEmbedder


class IPCRetriever:
    def __init__(self, chunks_path: str, index_path: str, chunk_ids_path: str | None = None):
        """
        IPC Retriever
        Assumes 1-to-1 alignment between chunks file and FAISS index.
        """
        self.embedder = LegalEmbedder()

        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

        # Optional chunk_id order mapping check
        if chunk_ids_path and os.path.exists(chunk_ids_path):
            with open(chunk_ids_path, "r", encoding="utf-8") as f:
                chunk_ids = json.load(f)
            if len(chunk_ids) != len(self.chunks):
                print(
                    f"[WARN] chunk_id mapping size ({len(chunk_ids)}) "
                    f"!= chunks size ({len(self.chunks)})"
                )

        # Safety check (VERY IMPORTANT)
        if self.index.ntotal != len(self.chunks):
            print(
                f"[WARN] IPC index size ({self.index.ntotal}) "
                f"!= chunks size ({len(self.chunks)})"
            )

    def retrieve(self, query: str, top_k: int = 5):
        """
        Retrieve top_k IPC chunks relevant to the query.
        """

        # Encode query
        q_emb = self.embedder.embed_query(query).astype("float32")
        q_emb = np.expand_dims(q_emb, axis=0)

        # FAISS search
        scores, ids = self.index.search(q_emb, top_k)

        results = []
        max_idx = len(self.chunks)

        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            if idx >= max_idx:
                # This should not happen if index & chunks are aligned
                continue

            chunk = dict(self.chunks[idx])  # shallow copy to avoid mutating source
            chunk["score"] = float(score)
            results.append(chunk)

        return results
