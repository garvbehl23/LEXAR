import time
import json
from backend.app.services.retrieval.retriever import DenseRetriever

with open("data/processed_docs/lexar_medium_chunks.json") as f:
    chunks = json.load(f)

start = time.time()
retriever = DenseRetriever(
    chunks,
    index_path="data/faiss_index/lexar_medium.index"
)
print("Index load time:", time.time() - start)

queries = [
    "What is the punishment for murder under IPC?",
    "Explain culpable homicide under IPC",
    "Can intent be inferred from circumstances in murder cases?"
]

for q in queries:
    t0 = time.time()
    retriever.retrieve(q, top_k=5)
    print(f"Query time ({q[:30]}...):", time.time() - t0)
