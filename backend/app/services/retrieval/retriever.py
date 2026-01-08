import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(
        self,
        chunks: list[dict],
        index_path: str = "data/faiss_index/ipc.index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.chunks = chunks
        self.index_path = index_path
        self.model = SentenceTransformer(model_name)

        self.embeddings = None
        self.index = None

        if os.path.exists(self.index_path):
            self._load_index()
        else:
            self._build_index()

    def _build_index(self):
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)

        faiss.write_index(self.index, self.index_path)

    def _load_index(self):
        self.index = faiss.read_index(self.index_path)

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            chunk = self.chunks[idx]
            chunk["score"] = float(-dist)
            results.append(chunk)

        return results
