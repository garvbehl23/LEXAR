import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class UserRetriever:
    def __init__(self, chunks):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.chunks = chunks

        embeddings = self.model.encode(
            [c["text"] for c in chunks]
        ).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query]).astype("float32")
        scores, ids = self.index.search(q_emb, top_k)

        return [self.chunks[i] for i in ids[0] if i != -1]
