"""
Weakly-supervised contrastive training for LEXAR retriever (query encoder only).
Extended for IPC + CrPC + IEA corpus (2733 chunks).

Focus: Improve legal discrimination between substantive evidence law (IEA) and criminal procedure (CrPC).

Key improvements:
- IEA-specific templates for admissibility queries
- Hard cross-statute negatives (IEA vs CrPC vs IPC)
- Explicit positive pairs for IEA §§24-26 (confession admissibility)
- Validation on admissibility-focused queries
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
IEA_PATH = os.path.join(ROOT, "data", "processed_docs", "iea_1872_chunks.json")

OUTPUT_DIR = os.path.join(ROOT, "data", "models", "lexar_query_encoder_v2")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
BATCH_SIZE = 16
EPOCHS = 2
LR = 2e-5
TEMPERATURE = 0.05
NEGATIVES_PER_QUERY = 6


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_chunks() -> Tuple[List[dict], List[dict], List[dict]]:
    with open(IPC_PATH, "r", encoding="utf-8") as f:
        ipc = json.load(f)
    with open(CRPC_PATH, "r", encoding="utf-8") as f:
        crpc = json.load(f)
    with open(IEA_PATH, "r", encoding="utf-8") as f:
        iea = json.load(f)
    return ipc, crpc, iea


def generate_queries_ipc_crpc(chunks: List[dict], statute: str, index_offset: int) -> Tuple[List[str], List[int]]:
    """Generate queries for IPC/CrPC."""
    queries = []
    pos_indices = []
    
    templates = [
        "What is {title}?",
        "Explain {title}",
        "{title}",
        "What does Section {section} cover?",
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


def generate_queries_iea(chunks: List[dict], index_offset: int) -> Tuple[List[str], List[int]]:
    """
    Generate IEA-specific queries with focus on admissibility and relevance.
    Explicitly targets sections 24-26 (confession admissibility).
    """
    queries = []
    pos_indices = []
    
    # General templates
    general_templates = [
        "What is {title}?",
        "Explain {title}",
        "{title}",
        "Is {title}?",
        "When is {title}?",
    ]
    
    # Admissibility-focused templates (for confession/evidence sections)
    admissibility_templates = [
        "Is {title} admissible?",
        "Is {title} relevant?",
        "When is {title} not admissible?",
        "Under what circumstances is {title} irrelevant?",
        "Can {title} be proved in court?",
    ]
    
    # Confession-specific templates (for sections 24-26)
    confession_templates = [
        "Is a confession to police admissible?",
        "Can confession to police officer be used in court?",
        "Are confessions made in police custody admissible?",
        "When is a confession inadmissible?",
        "What confessions are not allowed as evidence?",
    ]
    
    confession_sections = {"24", "25", "26"}
    
    for idx, c in enumerate(chunks):
        meta = c.get("metadata", {})
        title = meta.get("section_title") or meta.get("chapter_title") or ""
        section = meta.get("section") or ""
        text = c.get("text", "").lower()
        
        if not title:
            continue
        
        # Check if this is a confession/admissibility section
        is_confession_section = section in confession_sections
        is_admissibility_section = any(kw in title.lower() or kw in text for kw in 
            ["admissible", "confession", "relevant", "irrelevant", "proved", "not to be proved"])
        
        # Select templates based on content
        if is_confession_section:
            # Use all templates for critical confession sections
            selected_templates = general_templates + admissibility_templates + confession_templates
        elif is_admissibility_section:
            selected_templates = general_templates + admissibility_templates
        else:
            selected_templates = general_templates
        
        for tmpl in selected_templates:
            q = tmpl.format(title=title, section=section)
            queries.append(q)
            pos_indices.append(idx + index_offset)
    
    return queries, pos_indices


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load chunks
    print("Loading chunks...")
    ipc_chunks, crpc_chunks, iea_chunks = load_chunks()
    all_chunks = ipc_chunks + crpc_chunks + iea_chunks
    
    print(f"IPC: {len(ipc_chunks)}, CrPC: {len(crpc_chunks)}, IEA: {len(iea_chunks)}, Total: {len(all_chunks)}")
    
    # Generate queries
    print("\nGenerating queries...")
    ipc_queries, ipc_pos = generate_queries_ipc_crpc(ipc_chunks, "IPC", 0)
    crpc_queries, crpc_pos = generate_queries_ipc_crpc(crpc_chunks, "CrPC", len(ipc_chunks))
    iea_queries, iea_pos = generate_queries_iea(iea_chunks, len(ipc_chunks) + len(crpc_chunks))
    
    queries = ipc_queries + crpc_queries + iea_queries
    pos_indices = ipc_pos + crpc_pos + iea_pos
    
    print(f"Generated {len(queries)} queries")
    print(f"  IPC queries: {len(ipc_queries)}")
    print(f"  CrPC queries: {len(crpc_queries)}")
    print(f"  IEA queries: {len(iea_queries)}")
    
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
    print("\nLoading models...")
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
    
    # Create statute masks for negative sampling
    ipc_mask = torch.zeros(len(all_chunks), dtype=torch.bool, device=device)
    crpc_mask = torch.zeros(len(all_chunks), dtype=torch.bool, device=device)
    iea_mask = torch.zeros(len(all_chunks), dtype=torch.bool, device=device)
    
    ipc_mask[:len(ipc_chunks)] = True
    crpc_mask[len(ipc_chunks):len(ipc_chunks)+len(crpc_chunks)] = True
    iea_mask[len(ipc_chunks)+len(crpc_chunks):] = True
    
    # Evaluate before training
    print("\n" + "="*80)
    print("EVALUATION BEFORE TRAINING")
    print("="*80)
    
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
    
    print(f"Recall@1: {recall1_before:.4f}")
    print(f"Recall@5: {recall5_before:.4f}")
    
    # Test validation query before
    print("\nValidation query (BEFORE): 'Is a confession to police admissible?'")
    test_query_before(query_model, doc_embeddings, all_chunks, device)
    
    # Training
    print("\n" + "="*80)
    print("TRAINING")
    print("="*80)
    
    query_model.train()
    optimizer = torch.optim.AdamW(query_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    
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
            
            # Sample hard cross-statute negatives
            neg_emb_list = []
            for pos_idx in batch_pos:
                if pos_idx < len(ipc_chunks):
                    # This is IPC, sample from CrPC and IEA
                    neg_pool = torch.where(crpc_mask | iea_mask)[0]
                elif pos_idx < len(ipc_chunks) + len(crpc_chunks):
                    # This is CrPC, sample from IPC and IEA
                    neg_pool = torch.where(ipc_mask | iea_mask)[0]
                else:
                    # This is IEA, sample from IPC and CrPC (hard negatives)
                    neg_pool = torch.where(ipc_mask | crpc_mask)[0]
                
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
            
            if num_batches % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {num_batches}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")
    
    # Evaluate after training
    print("\n" + "="*80)
    print("EVALUATION AFTER TRAINING")
    print("="*80)
    
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
    
    print(f"Recall@1: {recall1_after:.4f}")
    print(f"Recall@5: {recall5_after:.4f}")
    
    print("\n" + "-"*80)
    print("IMPROVEMENT:")
    print(f"Recall@1: {recall1_before:.4f} → {recall1_after:.4f} (Δ {recall1_after - recall1_before:+.4f})")
    print(f"Recall@5: {recall5_before:.4f} → {recall5_after:.4f} (Δ {recall5_after - recall5_before:+.4f})")
    print("-"*80)
    
    # Test validation query after
    print("\nValidation query (AFTER): 'Is a confession to police admissible?'")
    test_query_after(query_model, doc_embeddings, all_chunks, device)
    
    # Save model
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query_model.save(OUTPUT_DIR)
    print(f"\n✓ Saved fine-tuned model to {OUTPUT_DIR}")
    print("="*80)


def test_query_before(model, doc_embeddings, chunks, device):
    """Test query before training."""
    query = "Is a confession to police admissible?"
    
    with torch.no_grad():
        q_emb = model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(device)
        
        sims = q_emb @ doc_embeddings.T
        top_k = torch.topk(sims, k=5, dim=1)
        
        print("Top-5 results:")
        for rank, (idx, score) in enumerate(zip(top_k.indices[0], top_k.values[0]), 1):
            chunk = chunks[idx.item()]
            chunk_id = chunk.get("chunk_id", "?")
            text_preview = chunk.get("text", "")[:100].replace("\n", " ")
            print(f"  {rank}. [{score:.4f}] {chunk_id}: {text_preview}...")
        
        # Check IEA in top-3
        top3_ids = [chunks[idx.item()].get("chunk_id", "") for idx in top_k.indices[0][:3]]
        iea_in_top3 = [cid for cid in top3_ids if cid.startswith("iea")]
        
        if iea_in_top3:
            print(f"✓ IEA sections in top-3: {iea_in_top3}")
        else:
            print("✗ No IEA sections in top-3")


def test_query_after(model, doc_embeddings, chunks, device):
    """Test query after training."""
    query = "Is a confession to police admissible?"
    
    with torch.no_grad():
        q_emb = model.encode(
            [query],
            convert_to_tensor=True,
            normalize_embeddings=True
        ).to(device)
        
        sims = q_emb @ doc_embeddings.T
        top_k = torch.topk(sims, k=5, dim=1)
        
        print("Top-5 results:")
        for rank, (idx, score) in enumerate(zip(top_k.indices[0], top_k.values[0]), 1):
            chunk = chunks[idx.item()]
            chunk_id = chunk.get("chunk_id", "?")
            text_preview = chunk.get("text", "")[:100].replace("\n", " ")
            print(f"  {rank}. [{score:.4f}] {chunk_id}: {text_preview}...")
        
        # Check IEA in top-3
        top3_ids = [chunks[idx.item()].get("chunk_id", "") for idx in top_k.indices[0][:3]]
        iea_in_top3 = [cid for cid in top3_ids if cid.startswith("iea")]
        
        if iea_in_top3:
            print(f"✓ IEA sections in top-3: {iea_in_top3}")
        else:
            print("✗ No IEA sections in top-3")


if __name__ == "__main__":
    main()
