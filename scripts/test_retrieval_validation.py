"""
Test retrieval with fine-tuned query encoder on validation queries.

Queries:
1. "Evidence law" → Expect IEA §§24-26
2. "Is a confession to police admissible?" → Expect IEA §§24-26  
3. "What evidence is required to prove murder?" → Expect IPC + IEA mix
4. "When can police arrest without warrant?" → Expect CrPC §41
"""

import json
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
INDEX_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc_iea.index")
CHUNK_IDS_PATH = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc_iea_chunk_ids.json")
QUERY_ENCODER_PATH = os.path.join(ROOT, "data", "models", "lexar_query_encoder_v1")

# Check if fine-tuned encoder exists, otherwise use base
if os.path.exists(QUERY_ENCODER_PATH):
    print(f"Loading fine-tuned query encoder from {QUERY_ENCODER_PATH}")
    query_encoder = SentenceTransformer(QUERY_ENCODER_PATH)
    encoder_type = "FINE-TUNED"
else:
    print("Fine-tuned encoder not found, using base model")
    query_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    encoder_type = "BASE"

# Load FAISS index
print(f"Loading FAISS index from {INDEX_PATH}")
index = faiss.read_index(INDEX_PATH)
print(f"Index has {index.ntotal} vectors")

# Load chunk IDs
with open(CHUNK_IDS_PATH, "r") as f:
    chunk_ids = json.load(f)

print(f"Chunk IDs: {len(chunk_ids)}")
print()

# Test queries
queries = [
    ("Evidence law", "IEA §§24-26"),
    ("Is a confession to police admissible?", "IEA §§24-26"),
    ("What evidence is required to prove murder?", "IPC + IEA mix"),
    ("When can police arrest without warrant?", "CrPC §41"),
]

TOP_K = 10

print("=" * 80)
print(f"RETRIEVAL TEST RESULTS ({encoder_type} ENCODER)")
print("=" * 80)

for query_text, expected in queries:
    print(f"\nQuery: \"{query_text}\"")
    print(f"Expected: {expected}")
    print("-" * 80)
    
    # Encode query
    q_emb = query_encoder.encode([query_text], normalize_embeddings=True)
    q_emb = np.array(q_emb).astype("float32")
    
    # Search
    distances, indices = index.search(q_emb, TOP_K)
    
    # Display results
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        if idx == -1:
            continue
        chunk_id = chunk_ids[idx]
        score = float(dist)  # Inner product (higher is better for normalized vectors)
        print(f"{rank:2d}. {chunk_id:20s} (score: {score:.4f})")
    
    print()

print("=" * 80)
