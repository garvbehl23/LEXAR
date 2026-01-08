import json
from backend.app.services.retrieval.retriever import DenseRetriever
from backend.app.services.reranking.cross_encoder import LegalCrossEncoderReranker

with open("data/processed_docs/ipc2_chunks.json") as f:
    chunks = json.load(f)

query = "What is the punishment for murder under IPC?"

# Step 1: Dense retrieval
retriever = DenseRetriever(chunks)
retrieved = retriever.retrieve(query, top_k=10)

print("\n--- BEFORE RE-RANKING ---")
for r in retrieved[:3]:
    print(r["score"], r["text"][:80])

# Step 2: Re-ranking
reranker = LegalCrossEncoderReranker()
reranked = reranker.rerank(query, retrieved, top_k=5)

print("\n--- AFTER RE-RANKING ---")
for r in reranked:
    print(r["rerank_score"], r["text"][:80])
