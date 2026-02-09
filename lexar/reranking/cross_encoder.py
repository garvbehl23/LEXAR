from sentence_transformers import CrossEncoder

class LegalCrossEncoderReranker:
    """
    Cross-encoder based evidence re-ranker φω
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, retrieved_chunks: list[dict], top_k: int = 5):
        """
        Args:
            query: legal question
            retrieved_chunks: output of dense retriever
            top_k: number of chunks to keep after re-ranking
        """

        pairs = [(query, c["text"]) for c in retrieved_chunks]
        scores = self.model.predict(pairs)

        for c, score in zip(retrieved_chunks, scores):
            c["rerank_score"] = float(score)

        reranked = sorted(
            retrieved_chunks,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]
