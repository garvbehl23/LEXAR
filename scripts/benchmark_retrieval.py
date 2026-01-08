import time
import json
from backend.app.services.retrieval.retriever import DenseRetriever

with open("data/processed_docs/ipc2_chunks.json") as f:
    chunks = json.load(f)

# First run (build)
start = time.time()
retriever = DenseRetriever(chunks)
print("Index build time:", time.time() - start)

# Second run (load)
start = time.time()
retriever = DenseRetriever(chunks)
print("Index load time:", time.time() - start)

# Query time
start = time.time()
retriever.retrieve("What is the punishment for murder under IPC?")
print("Query time:", time.time() - start)
