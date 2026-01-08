import json
from backend.app.services.retrieval.retriever import DenseRetriever

with open("data/processed_docs/lexar_medium_chunks.json") as f:
    chunks = json.load(f)

retriever = DenseRetriever(
    chunks,
    index_path="data/faiss_index/lexar_medium.index"
)

print("Medium-scale FAISS index built.")
