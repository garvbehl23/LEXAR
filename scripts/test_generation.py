import json
from backend.app.services.retrieval.retriever import DenseRetriever
from backend.app.services.reranking.cross_encoder import LegalCrossEncoderReranker
from backend.app.services.generation.context_fusion import fuse_context
from backend.app.services.generation.lexar_generator import LexarGenerator

with open("data/processed_docs/ipc2_chunks.json") as f:
    chunks = json.load(f)

query = "What is the punishment for murder under IPC?"

retriever = DenseRetriever(chunks)
retrieved = retriever.retrieve(query, top_k=10)

reranker = LegalCrossEncoderReranker()
evidence = reranker.rerank(query, retrieved, top_k=3)

prompt = fuse_context(query, evidence)

generator = LexarGenerator()
answer = generator.generate(prompt)

print("\n--- GENERATED ANSWER ---\n")
print(answer)
