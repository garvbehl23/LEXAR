import faiss
import numpy as np

class FaissIndex:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray):
        self.index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int = 5):
        query_vector = np.expand_dims(query_vector, axis=0)
        scores, indices = self.index.search(query_vector, k)
        return scores[0], indices[0]
