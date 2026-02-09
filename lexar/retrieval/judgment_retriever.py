import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class JudgmentRetriever:
    def __init__(self, chunks_path, index_path):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        with open(chunks_path) as f:
            self.chunks = json.load(f)

        self.index = faiss.read_index(index_path)

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query]).astype("float32")
        scores, ids = self.index.search(q_emb, top_k)

        results = []
        max_idx = len(self.chunks)

        for i in ids[0]:
            if i == -1:
                continue
            if i >= max_idx:
                continue
            results.append(self.chunks[i])

        return results
