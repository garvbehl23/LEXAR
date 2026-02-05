"""
Simplified weakly-supervised contrastive training for LEXAR retriever.

Fixed version: no DataLoader hangs, simple batch processing.
"""

import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IPC_PATH = os.path.join(ROOT, "data", "processed_docs", "ipc_chunks.json")
CRPC_PATH = os.path.join(ROOT, "data", "processed_docs", "crpc_chunks.json")

OUTPUT_DIR = os.path.join(ROOT, "data", "models", "lexar_query_encoder_v1")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
BATCH_SIZE = 16
EPOCHS = 1
LR = 2e-5
TEMPERATURE = 0.05
NEGATIVES_PER_QUERY = 4


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_chunks():
    with open(IPC_PATH, "r", encoding="utf-8") as f:
        ipc = json.load(f)
    with open(CRPC_PATH, "r", encoding="utf-8") as f:
        crpc = json.load(f)
    return ipc, crpc


def build_overlap_sections(ipc: List[dict], crpc: List[dict]) -> set:
    ipc_sections = {c.get("metadata", {}).get("section") for c in ipc}
    crpc_sections = {c.get("metadata", {}).get("section") for c in crpc}
    return {s for s in ipc_sections.intersection(crpc_sections) if s}


def generate_queries_simple(chunks: List[dict], statute: str, index_offset: int, overlap: set) -> Tuple[List[str], List[int]]:
    """Generate synthetic queries: return (queries, positive_indices)"""
    queries = []
    pos_indices = []
    
    templates = [
        "What is {title}?",
        "Explain {title}",
        "{title}",
    ]
    
    for idx, c in enumerate(chunks):
        meta = c.get("metadata", {})
        title = meta.get("section_title") or meta.get("chapter_title") or ""
        section = meta.get("section") or ""
        
        if not title:
            continue
        
        for tmpl in templates:
            q = tmpl.format(title=title, section=section, statute=statute)
            queries.append(q)
            pos_indices.append(idx + index_offset)
    
    return queries, pos_indices


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load chunks
    print("Loading chunks...")
    ipc_chunks, crpc_chunks = load_chunks()
    all_chunks = ipc_chunks + crpc_chunks
    
    print(f"IPC: {len(ipc_chunks)}, CrPC: {len(crpc_chunks)}, Total: {len(all_chunks)}")
    
    overlap = build_overlap_sections(ipc_chunks, crpc_chunks)
    
    # Generate queries
    print("Generating queries...")
    ipc_queries, ipc_pos = generate_queries_simple(ipc_chunks, "IPC", 0, overlap)
    crpc_queries, crpc_pos = generate_queries_simple(crpc_chunks, "CrPC", len(ipc_chunks), overlap)
    
    queries = ipc_queries + crpc_queries
    pos_indices = ipc_pos + crpc_pos
    
    print(f"Generated {len(queries)} queries")
    
    # Split train/eval
    rng = random.Random(SEED)
    indices = list(range(len(queries)))
    rng.shuffle(indices)
    
    split = int(0.9 * len(queries))
    train_indices = indices[:split]
    eval_indices = indices[split:]
    
    train_queries = [queries[i] for i in train_indices]
    train_pos = [pos_indices[i] for i in train_indices]
    
    eval_queries = [queries[i] for i in eval_indices]
    eval_pos = [pos_indices[i] for i in eval_indices]
    
    print(f"Train: {len(train_queries)}, Eval: {len(eval_queries)}")
    
    # Initialize models
    print("Loading models...")
    query_model = SentenceTransformer(MODEL_NAME)
    doc_model = SentenceTransformer(MODEL_NAME)
    
    query_model.to(device)
    doc_model.to(device)
    doc_model.eval()
    
    for p in doc_model.parameters():
        p.requires_grad = False
    
    # Precompute doc embeddings
    print("Encoding documents...")
    with torch.no_grad():
        doc_embeddings = doc_model.encode(
            [c["text"] for c in all_chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).to(device)
    
    print(f"Doc embeddings: {doc_embeddings.shape}")
    
    # Evaluate before training
    print("\nEvaluating before training...")
    query_model.eval()
    with torch.no_grad():
        eval_q_emb = query_model.encode(
            eval_queries,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).to(device)
        
        sims = eval_q_emb @ doc_embeddings.T
        top1 = torch.topk(sims, k=1, dim=1).indices.squeeze(1)
        top5 = torch.topk(sims, k=5, dim=1).indices
        
        eval_pos_tensor = torch.tensor(eval_pos, device=device).unsqueeze(1)
        recall1_before = (top1.unsqueeze(1) == eval_pos_tensor).float().mean().item()
        recall5_before = (top5 == eval_pos_tensor).any(dim=1).float().mean().item()
    
    print(f"Before: Recall@1={recall1_before:.4f}, Recall@5={recall5_before:.4f}")
    
    # Training
    print("\nTraining...")
    query_model.train()
    optimizer = torch.optim.AdamW(query_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create statute masks for negative sampling
    ipc_mask = torch.zeros(len(all_chunks), dtype=torch.bool, device=device)
    crpc_mask = torch.zeros(len(all_chunks), dtype=torch.bool, device=device)
    ipc_mask[:len(ipc_chunks)] = True
    crpc_mask[len(ipc_chunks):] = True
    
    for epoch in range(EPOCHS):
        total_loss = 0.0
        num_batches = 0
        
        # Mini batches
        for batch_start in range(0, len(train_queries), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(train_queries))
            batch_queries = train_queries[batch_start:batch_end]
            batch_pos = torch.tensor(train_pos[batch_start:batch_end], dtype=torch.long, device=device)
            
            # Encode queries (requires grad)
            features = query_model.tokenize(batch_queries)
            features = {k: v.to(device) for k, v in features.items()}
            q_output = query_model(features)
            q_emb = torch.nn.functional.normalize(q_output['sentence_embedding'], p=2, dim=1)
            
            # Positive embeddings
            pos_emb = doc_embeddings[batch_pos]
            
            # Sample negatives (cross-statute)
            neg_emb_list = []
            for pos_idx in batch_pos:
                if pos_idx < len(ipc_chunks):
                    # This is IPC, sample from CrPC
                    neg_pool = torch.where(crpc_mask)[0]
                else:
                    # This is CrPC, sample from IPC
                    neg_pool = torch.where(ipc_mask)[0]
                
                sample_k = min(len(neg_pool), NEGATIVES_PER_QUERY)
                if sample_k > 0:
                    sampled_neg = neg_pool[torch.randperm(len(neg_pool))[:sample_k]]
                    neg_emb_list.append(doc_embeddings[sampled_neg])
            
            # Pad negatives to same size
            if neg_emb_list:
                max_neg = max(e.shape[0] for e in neg_emb_list)
                neg_emb_padded = []
                for neg_emb in neg_emb_list:
                    if neg_emb.shape[0] < max_neg:
                        # Repeat to pad
                        repeat_factor = max_neg // neg_emb.shape[0] + 1
                        neg_emb = neg_emb.repeat(repeat_factor, 1)[:max_neg]
                    neg_emb_padded.append(neg_emb)
                
                neg_emb_all = torch.cat(neg_emb_padded, dim=0)
                doc_bank = torch.cat([pos_emb, neg_emb_all], dim=0)
            else:
                doc_bank = pos_emb
            
            # Loss
            logits = (q_emb @ doc_bank.T) / TEMPERATURE
            targets = torch.arange(q_emb.size(0), dtype=torch.long, device=device)
            loss = loss_fn(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    # Evaluate after training
    print("\nEvaluating after training...")
    query_model.eval()
    with torch.no_grad():
        eval_q_emb = query_model.encode(
            eval_queries,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).to(device)
        
        sims = eval_q_emb @ doc_embeddings.T
        top1 = torch.topk(sims, k=1, dim=1).indices.squeeze(1)
        top5 = torch.topk(sims, k=5, dim=1).indices
        
        eval_pos_tensor = torch.tensor(eval_pos, device=device).unsqueeze(1)
        recall1_after = (top1.unsqueeze(1) == eval_pos_tensor).float().mean().item()
        recall5_after = (top5 == eval_pos_tensor).any(dim=1).float().mean().item()
    
    print(f"After: Recall@1={recall1_after:.4f}, Recall@5={recall5_after:.4f}")
    print(f"Improvement: Recall@1 +{recall1_after - recall1_before:.4f}, Recall@5 +{recall5_after - recall5_before:.4f}")
    
    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query_model.save(OUTPUT_DIR)
    print(f"\nSaved model to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
