"""
Test retrieval with the fine-tuned query encoder vs base encoder.
Compares results for validation queries.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def load_chunks_and_index():
    """Load combined IPC+CrPC chunks and FAISS index."""
    ipc_chunks_path = os.path.join(ROOT, "data", "processed_docs", "ipc_chunks.json")
    crpc_chunks_path = os.path.join(ROOT, "data", "processed_docs", "crpc_chunks.json")
    index_path = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc.index")
    chunk_ids_path = os.path.join(ROOT, "data", "faiss_index", "ipc_crpc_chunk_ids.json")
    
    # Load IPC chunks
    with open(ipc_chunks_path, "r", encoding="utf-8") as f:
        ipc_chunks = json.load(f)
    
    # Load CrPC chunks
    with open(crpc_chunks_path, "r", encoding="utf-8") as f:
        crpc_chunks = json.load(f)
    
    # Load chunk IDs to get correct ordering
    with open(chunk_ids_path, "r", encoding="utf-8") as f:
        chunk_ids = json.load(f)
    
    # Create lookup by chunk_id
    chunk_lookup = {}
    for c in ipc_chunks:
        chunk_lookup[c.get("chunk_id")] = c
    for c in crpc_chunks:
        chunk_lookup[c.get("chunk_id")] = c
    
    # Reorder chunks to match FAISS index order
    chunks = []
    for cid in chunk_ids:
        if cid in chunk_lookup:
            chunks.append(chunk_lookup[cid])
        else:
            # Fallback: create placeholder
            chunks.append({"chunk_id": cid, "text": f"[MISSING: {cid}]"})
    
    index = faiss.read_index(index_path)
    return chunks, index


def retrieve_with_model(query: str, model, index, chunks, top_k: int = 5):
    """Retrieve using a specific encoder model."""
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q_emb, top_k)
    
    results = []
    for idx, score in zip(ids[0], scores[0]):
        if idx == -1 or idx >= len(chunks):
            continue
        chunk = dict(chunks[idx])
        chunk["score"] = float(score)
        results.append(chunk)
    return results


def format_result(chunk):
    """Format a chunk for display."""
    cid = chunk.get("chunk_id", "?")
    score = chunk.get("score", 0)
    text = chunk.get("text", "")[:100].replace("\n", " ")
    return f"  [{score:.4f}] {cid}: {text}..."


def main():
    print("Loading chunks and index...")
    chunks, index = load_chunks_and_index()
    print(f"Loaded {len(chunks)} chunks, index has {index.ntotal} vectors")
    
    # Load models
    print("\nLoading models...")
    base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    finetuned_path = os.path.join(ROOT, "data", "models", "lexar_query_encoder_v1")
    if not os.path.exists(finetuned_path):
        print(f"ERROR: Fine-tuned model not found at {finetuned_path}")
        return
    finetuned_model = SentenceTransformer(finetuned_path)
    print("Models loaded.")
    
    # Test queries
    queries = [
        "punishment for murder",
        "when can police arrest without warrant",
        "is murder bailable"
    ]
    
    print("\n" + "="*80)
    print("RETRIEVAL COMPARISON: Base vs Fine-tuned Query Encoder")
    print("="*80)
    
    for query in queries:
        print(f"\n{'='*80}")
        print(f"QUERY: {query}")
        print("-"*80)
        
        # Base model results
        base_results = retrieve_with_model(query, base_model, index, chunks, top_k=5)
        print("\n[BASE ENCODER] Top-5:")
        for r in base_results:
            print(format_result(r))
        
        # Fine-tuned model results
        ft_results = retrieve_with_model(query, finetuned_model, index, chunks, top_k=5)
        print("\n[FINE-TUNED ENCODER] Top-5:")
        for r in ft_results:
            print(format_result(r))
        
        # Compare
        base_ids = [r.get("chunk_id") for r in base_results]
        ft_ids = [r.get("chunk_id") for r in ft_results]
        
        new_in_top5 = set(ft_ids) - set(base_ids)
        removed_from_top5 = set(base_ids) - set(ft_ids)
        
        print(f"\n  Δ New in top-5: {new_in_top5 or 'none'}")
        print(f"  Δ Removed from top-5: {removed_from_top5 or 'none'}")
    
    print("\n" + "="*80)
    print("Retrieval comparison complete.")
    print("="*80)


if __name__ == "__main__":
    main()
