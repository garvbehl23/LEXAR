import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class IPCRetriever:
    def __init__(self, chunks_path: str, index_path: str):
        """
        IPC Retriever
        Assumes 1-to-1 alignment between chunks file and FAISS index.
        """

        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Load chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load FAISS index
        self.index = faiss.read_index(index_path)

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
        q_emb = self.model.encode([query])
        q_emb = np.asarray(q_emb, dtype="float32")

        # FAISS search
        scores, ids = self.index.search(q_emb, top_k)

        results = []
        max_idx = len(self.chunks)

        for idx in ids[0]:
            if idx == -1:
                continue
            if idx >= max_idx:
                # This should not happen if index & chunks are aligned
                continue

            results.append(self.chunks[idx])

        return results
