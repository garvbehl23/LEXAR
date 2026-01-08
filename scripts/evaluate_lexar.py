import json
from backend.app.services.retrieval.retriever import DenseRetriever
from backend.app.services.reranking.cross_encoder import LegalCrossEncoderReranker

# Load chunks
with open("data/processed_docs/ipc2_chunks.json") as f:
    chunks = json.load(f)

retriever = DenseRetriever(chunks)
reranker = LegalCrossEncoderReranker()

# Load gold queries
with open("evaluation/gold_queries.json") as f:
    gold_queries = json.load(f)

def extract_sections(results):
    sections = []
    for r in results:
        sec = r.get("metadata", {}).get("section")
        if sec:
            sections.append(sec)
    return sections

precision_at_5 = []
recall_at_5 = []
mrr_scores = []
precision_at_3 =[]
for item in gold_queries:
    query = item["query"]
    relevant = set(item["relevant_sections"])

    retrieved = retriever.retrieve(query, top_k=10)
    reranked = reranker.rerank(query, retrieved, top_k=5)

    retrieved_sections = extract_sections(reranked)
    retrieved_sections_3 = extract_sections(reranked[:3])
    hits_3 = len(relevant.intersection(retrieved_sections_3))
    precision_at_3.append(hits_3 / 3)
    # Precision@5
    hits = len(relevant.intersection(retrieved_sections))
    precision_at_5.append(hits / 5)

    # Recall@5
    recall_at_5.append(hits / len(relevant))

    # MRR
    rank = 0
    for i, sec in enumerate(retrieved_sections):
        if sec in relevant:
            rank = i + 1
            break
    mrr_scores.append(1 / rank if rank > 0 else 0)

print("\n--- LEXAR METRICS ---")
print("Precision@5:", sum(precision_at_5) / len(precision_at_5))
print("Recall@5:", sum(recall_at_5) / len(recall_at_5))
print("MRR:", sum(mrr_scores) / len(mrr_scores))
print("Precision@3:", sum(precision_at_3) / len(precision_at_3))
