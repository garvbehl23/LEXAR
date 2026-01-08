import json

from backend.app.services.retrieval.retriever import DenseRetriever
from backend.app.services.reranking.cross_encoder import LegalCrossEncoderReranker
from backend.app.services.generation.context_fusion import fuse_context
from backend.app.services.generation.lexar_generator import LexarGenerator
from backend.app.services.citation.citation_mapper import attach_citations
from backend.app.services.generation.answer_cleaner import clean_answer

# Load IPC chunks
with open("data/processed_docs/ipc2_chunks.json") as f:
    chunks = json.load(f)

query = "What is the punishment for murder under IPC?"

# 1. Retrieve
retriever = DenseRetriever(chunks)
retrieved = retriever.retrieve(query, top_k=10)

# 2. Re-rank
reranker = LegalCrossEncoderReranker()
evidence = reranker.rerank(query, retrieved, top_k=3)

# 3. Fuse context
prompt = fuse_context(query, evidence)

# 4. Generate answer
generator = LexarGenerator()
answer = generator.generate(prompt)
answer = clean_answer(answer)
# 5. Attach citations
final_answer = attach_citations(answer, evidence)

print("\n--- FINAL ANSWER WITH CITATIONS ---\n")
print(final_answer)
