"""
Weakly-supervised contrastive training for LEXAR retriever (query encoder only).

Scope:
- Corpus: combined IPC + CrPC chunks (2331 total)
- Train query encoder only; chunk encoder remains frozen
- Generate synthetic queries from section titles + templates
- Positives: (query, originating chunk)
- Negatives: in-batch + random cross-statute
- Loss: InfoNCE with cosine similarity
- Evaluation: Recall@1, Recall@5 (before vs after)
- Reproducible (fixed seeds)

This script does NOT alter FAISS indices or generator behavior.
"""

import json
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IPC_PATH = os.path.join(ROOT, "data", "processed_docs", "ipc_chunks.json")
CRPC_PATH = os.path.join(ROOT, "data", "processed_docs", "crpc_chunks.json")

OUTPUT_DIR = os.path.join(ROOT, "data", "models", "lexar_query_encoder_v1")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42
BATCH_SIZE = 8
EPOCHS = 1
LR = 2e-5
TEMPERATURE = 0.05
NEGATIVES_PER_QUERY = 2


@dataclass
class QueryExample:
    query: str
    pos_idx: int
    statute: str
    ambiguous: bool


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_chunks() -> Tuple[List[dict], List[dict]]:
    with open(IPC_PATH, "r", encoding="utf-8") as f:
        ipc = json.load(f)
    with open(CRPC_PATH, "r", encoding="utf-8") as f:
        crpc = json.load(f)
    return ipc, crpc


def build_overlap_sections(ipc: List[dict], crpc: List[dict]) -> set:
    ipc_sections = {c.get("metadata", {}).get("section") for c in ipc}
    crpc_sections = {c.get("metadata", {}).get("section") for c in crpc}
    return {s for s in ipc_sections.intersection(crpc_sections) if s}


def generate_queries(
    chunks: List[dict],
    statute_label: str,
    overlap_sections: set,
    index_offset: int = 0
) -> List[QueryExample]:
    examples: List[QueryExample] = []

    templates_with_statute = [
        "What is {title} under {statute}?",
        "Explain {title} in {statute}.",
        "What does Section {section} of {statute} cover?",
        "{statute} Section {section}: {title}",
    ]

    templates_ambiguous = [
        "What does Section {section} cover?",
        "Explain Section {section}.",
        "{title}",
    ]

    for idx, c in enumerate(chunks):
        meta = c.get("metadata", {})
        title = meta.get("section_title") or meta.get("chapter_title") or ""
        section = meta.get("section") or ""

        # Skip if no usable metadata
        if not title and not section:
            continue

        # Standard queries with statute
        for tmpl in templates_with_statute:
            q = tmpl.format(title=title, section=section, statute=statute_label)
            examples.append(
                QueryExample(
                    query=q,
                    pos_idx=idx + index_offset,
                    statute=statute_label.lower(),
                    ambiguous=False
                )
            )

        # Ambiguous queries (cross-statute) only when section overlaps
        if section in overlap_sections:
            for tmpl in templates_ambiguous:
                q = tmpl.format(title=title, section=section, statute=statute_label)
                examples.append(
                    QueryExample(
                        query=q,
                        pos_idx=idx + index_offset,
                        statute=statute_label.lower(),
                        ambiguous=True
                    )
                )

    return examples


class QueryDataset(Dataset):
    def __init__(self, examples: List[QueryExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        return ex.query, ex.pos_idx, ex.statute, ex.ambiguous


def encode_queries(model: SentenceTransformer, queries: List[str], device: torch.device) -> torch.Tensor:
    features = model.tokenize(queries)
    features = {k: v.to(device) for k, v in features.items()}
    outputs = model(features)
    embeddings = outputs["sentence_embedding"]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings


def evaluate_recall(model: SentenceTransformer, queries: List[str], pos_indices: List[int],
                    doc_embeddings: torch.Tensor, device: torch.device) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        q_emb = []
        batch = 128
        for i in range(0, len(queries), batch):
            q_batch = queries[i:i+batch]
            q_emb.append(encode_queries(model, q_batch, device))
        q_emb = torch.cat(q_emb, dim=0)

        sims = q_emb @ doc_embeddings.T  # cosine similarity
        top1 = torch.topk(sims, k=1, dim=1).indices
        top5 = torch.topk(sims, k=5, dim=1).indices

        pos = torch.tensor(pos_indices, device=device).unsqueeze(1)
        recall1 = (top1 == pos).float().mean().item()
        recall5 = (top5 == pos).any(dim=1).float().mean().item()

    return recall1, recall5


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load chunks
    ipc_chunks, crpc_chunks = load_chunks()
    overlap_sections = build_overlap_sections(ipc_chunks, crpc_chunks)

    # Combine chunks for doc embeddings
    all_chunks = ipc_chunks + crpc_chunks
    statutes = [c.get("metadata", {}).get("statute", "") for c in all_chunks]

    # Generate synthetic queries
    ipc_examples = generate_queries(ipc_chunks, "IPC", overlap_sections, index_offset=0)
    crpc_examples = generate_queries(
        crpc_chunks,
        "CrPC",
        overlap_sections,
        index_offset=len(ipc_chunks)
    )
    all_examples = ipc_examples + crpc_examples

    # Build eval set with ambiguity
    ambiguous_examples = [ex for ex in all_examples if ex.ambiguous]
    random.shuffle(ambiguous_examples)
    eval_examples = ambiguous_examples[:200] if len(ambiguous_examples) > 200 else ambiguous_examples
    eval_ids = {id(ex) for ex in eval_examples}
    train_examples = [ex for ex in all_examples if id(ex) not in eval_ids]

    # Initialize encoders
    query_model = SentenceTransformer(MODEL_NAME)
    doc_model = SentenceTransformer(MODEL_NAME)

    query_model.to(device)
    doc_model.to(device)
    doc_model.eval()

    # Freeze doc encoder
    for p in doc_model.parameters():
        p.requires_grad = False

    # Precompute doc embeddings (frozen encoder)
    with torch.no_grad():
        doc_embeddings = doc_model.encode(
            [c["text"] for c in all_chunks],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).to(device)

    # Prepare training data
    train_dataset = QueryDataset(train_examples)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    # Evaluation before training
    eval_queries = [ex.query for ex in eval_examples]
    eval_pos = [ex.pos_idx for ex in eval_examples]
    r1_before, r5_before = evaluate_recall(query_model, eval_queries, eval_pos, doc_embeddings, device)

    print(f"Before training: Recall@1={r1_before:.4f}, Recall@5={r5_before:.4f}")

    # Training loop
    optimizer = torch.optim.AdamW(query_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    query_model.train()

    # Pre-compute statute indices as torch tensors
    ipc_indices = torch.tensor([i for i, s in enumerate(statutes) if s == "ipc"], dtype=torch.long, device=device)
    crpc_indices = torch.tensor([i for i, s in enumerate(statutes) if s == "crpc"], dtype=torch.long, device=device)

    rng = random.Random(SEED)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        batch_count = 0
        for batch_idx, batch in enumerate(train_loader):
            try:
                queries, pos_indices_list, statutes_b, _ = batch
                pos_indices = torch.as_tensor(list(pos_indices_list), dtype=torch.long, device=device)

                # Query embeddings
                q_emb = encode_queries(query_model, list(queries), device)

                # Positive doc embeddings
                pos_emb = doc_embeddings[pos_indices]

                # Build cross-statute negatives
                neg_indices_list = []
                for s_str in statutes_b:
                    neg_pool = crpc_indices if s_str == "ipc" else ipc_indices
                    sample_k = min(len(neg_pool), NEGATIVES_PER_QUERY)
                    if sample_k > 0:
                        sampled = rng.sample(neg_pool.cpu().tolist(), k=sample_k)
                        neg_indices_list.extend(sampled)

                if not neg_indices_list:
                    # Fallback to random negatives if no cross-statute available
                    all_idx = list(range(len(all_chunks)))
                    exclude_set = set(pos_indices_list.cpu().tolist())
                    available = [i for i in all_idx if i not in exclude_set]
                    if available:
                        sample_k = min(len(available), NEGATIVES_PER_QUERY * len(queries))
                        neg_indices_list = rng.sample(available, k=sample_k)

                if neg_indices_list:
                    neg_indices = torch.as_tensor(neg_indices_list, dtype=torch.long, device=device)
                    neg_emb = doc_embeddings[neg_indices]
                    doc_bank = torch.cat([pos_emb, neg_emb], dim=0)
                else:
                    # Only positives as fallback
                    doc_bank = pos_emb

                # Contrastive loss (InfoNCE)
                logits = (q_emb @ doc_bank.T) / TEMPERATURE
                targets = torch.arange(q_emb.size(0), dtype=torch.long, device=device)
                loss = loss_fn(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        avg_loss = total_loss / max(batch_count, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

    # Evaluation after training
    r1_after, r5_after = evaluate_recall(query_model, eval_queries, eval_pos, doc_embeddings, device)
    print(f"After training: Recall@1={r1_after:.4f}, Recall@5={r5_after:.4f}")

    # Save fine-tuned query encoder
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query_model.save(OUTPUT_DIR)
    print(f"Saved fine-tuned query encoder to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
