import json

from backend.app.services.retrieval.retriever import DenseRetriever
from backend.app.services.reranking.cross_encoder import LegalCrossEncoderReranker
from backend.app.services.generation.context_fusion import fuse_context
from backend.app.services.generation.lexar_generator import LexarGenerator
from backend.app.services.generation.answer_cleaner import clean_answer
from backend.app.services.citation.citation_mapper import attach_citations

class LexarPipeline:
    def __init__(self, chunks_path: str):
        with open(chunks_path) as f:
            self.chunks = json.load(f)

        self.retriever = DenseRetriever(self.chunks)
        self.reranker = LegalCrossEncoderReranker()
        self.generator = LexarGenerator()

    def answer(self, query: str):
        retrieved = self.retriever.retrieve(query, top_k=10)
        evidence = self.reranker.rerank(query, retrieved, top_k=3)

        prompt = fuse_context(query, evidence)
        answer = self.generator.generate(prompt)
        answer = clean_answer(answer)

        return attach_citations(answer, evidence)
