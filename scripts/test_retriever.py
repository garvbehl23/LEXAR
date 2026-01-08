import json
from backend.app.services.retrieval.retriever import DenseRetriever

with open("data/processed_docs/ipc_chunks.json") as f:
    chunks = json.load(f)

retriever = DenseRetriever(chunks)

query = "What is the punishment for murder under IPC?"
results = retriever.retrieve(query, top_k=3)

for r in results:
    print("\n---")
    print("Score:", r["score"])
    print("Text:", r["text"][:300])
