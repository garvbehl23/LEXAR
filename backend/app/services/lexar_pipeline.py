from backend.app.services.retrieval.multi_index_retriever import MultiIndexRetriever
from backend.app.services.reranking.cross_encoder import LegalCrossEncoderReranker
from backend.app.services.generation.context_fusion import fuse_context
from backend.app.services.generation.lexar_generator import LexarGenerator
from backend.app.services.citation.citation_mapper import attach_citations


class LexarPipeline:
    """
    End-to-end LEXAR pipeline:
    routing → retrieval → reranking → generation → citation
    """

    def __init__(self, ipc=None, judgment=None, user=None):
        self.retriever = MultiIndexRetriever(
            ipc=ipc,
            judgment=judgment,
            user=user
        )
        self.reranker = LegalCrossEncoderReranker()
        self.generator = LexarGenerator()

    def answer(self, query: str, has_user_docs: bool = False, top_k: int = 10):
        # 1. Retrieve
        retrieved = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            has_user_docs=has_user_docs
        )

        if not retrieved:
            return "No relevant legal material found."

        # 2. Rerank
        evidence = self.reranker.rerank(
            query,
            retrieved,
            top_k=3
        )

        # 3. Fuse context
        prompt = fuse_context(query, evidence)

        # 4. Generate answer
        answer = self.generator.generate(prompt)

        # 5. Attach citations
        final_answer = attach_citations(answer, evidence)

        return final_answer
