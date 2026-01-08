import time
import json

from backend.app.services.retrieval.retriever import DenseRetriever
from backend.app.services.reranking.cross_encoder import LegalCrossEncoderReranker
from backend.app.services.generation.context_fusion import fuse_context
from backend.app.services.generation.lexar_generator import LexarGenerator
from backend.app.services.generation.answer_cleaner import clean_answer
from backend.app.services.citation.citation_mapper import attach_citations

# Load medium-scale corpus
with open("data/processed_docs/lexar_medium_chunks.json") as f:
    chunks = json.load(f)

query = "What is the punishment for murder under IPC?"

# Initialize components
retriever = DenseRetriever(
    chunks,
    index_path="data/faiss_index/lexar_medium.index"
)
reranker = LegalCrossEncoderReranker()
generator = LexarGenerator()

timings = {}

# --- Retrieval ---
t0 = time.time()
retrieved = retriever.retrieve(query, top_k=10)
timings["retrieval"] = time.time() - t0

# --- Re-ranking ---
t0 = time.time()
evidence = reranker.rerank(query, retrieved, top_k=3)
timings["reranking"] = time.time() - t0

# --- Prompt fusion ---
t0 = time.time()
prompt = fuse_context(query, evidence)
timings["fusion"] = time.time() - t0

# --- Generation ---
t0 = time.time()
answer = generator.generate(prompt)
answer = clean_answer(answer)
timings["generation"] = time.time() - t0

# --- Citation attachment ---
t0 = time.time()
final_answer = attach_citations(answer, evidence)
timings["citation"] = time.time() - t0

timings["total"] = sum(timings.values())

print("\n--- END-TO-END LEXAR LATENCY (seconds) ---")
for k, v in timings.items():
    print(f"{k:12s}: {v:.4f}")

print("\n--- FINAL ANSWER ---")
print(final_answer)
