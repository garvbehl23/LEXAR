# LEXAR: Legal EXplainable Augmented Reasoner

[![PyPI version](https://badge.fury.io/py/lexar-ai.svg)](https://badge.fury.io/py/lexar-ai)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A pip-installable research framework for evidence-grounded legal question answering.**

```bash
pip install lexar-ai
```

```python
from lexar import LexarPipeline

pipeline = LexarPipeline()
result = pipeline.answer("What is the punishment for murder under IPC?")
print(result["answer"])
```

---

## 1. Project Overview

**LEXAR (Legal EXplainable Augmented Reasoner)** is a retrieval-augmented generation system for legal question answering that prioritizes explainability and evidence grounding. The system is designed with strict architectural constraints to prevent hallucination and ensure all answers are supported by cited legal text. LEXAR models legal QA as latent-variable inference: $P(y | q, D) \approx P(y | q, R(q))$, where retrieval is not optional and generation is constrained via hard attention masking. The decoder may attend only to retrieved legal chunks and the user query, making hallucination prevention architectural rather than post-hoc.

**Key Features**:
- 🎯 **Evidence-First Architecture**: No generation without retrieval; hard attention masking prevents hallucination
- 📊 **Explainable Provenance**: Token-level attribution maps every generated word to source legal text
- ✅ **Safety Guarantees**: Evidence sufficiency gating rejects answers with insufficient support
- 🔬 **Research-Grade**: Deterministic retrieval, reproducible training, CPU-compatible
- 📦 **Pip-Installable**: `pip install lexar-ai` for easy integration into your projects

---

## 2. System Architecture

LEXAR implements a modular, auditable pipeline with localized error attribution:

```
Query → Dense Retrieval → Evidence Ranking → Context Fusion → 
Evidence-Constrained Generation → Token Provenance → 
Citation-Aware Output
```

### Core Pipeline Stages

| Stage | Component | Purpose |
|-------|-----------|---------|
| **Retrieval** | Query encoder + FAISS index | Dense retrieval of top-K evidence chunks |
| **Ranking** | Cross-encoder reranker | Relevance-based re-ranking of retrieved chunks |
| **Evidence Selection** | Attention masking | Hard masking ensures attention only to retrieved evidence |
| **Generation** | T5-base decoder | Evidence-constrained text generation |
| **Attribution** | Token provenance tracker | Maps generated tokens to source chunks via attention weights |
| **Safety** | Sufficiency gating | Rejects answers with insufficient evidence (max attention < threshold) |

### Design Principles

1. **Modularity**: Errors must be localizable to a stage (retrieval, ranking, generation, attribution)
2. **Auditability**: No end-to-end black boxes; each component has interpretable inputs/outputs
3. **Hard Constraints**: Attention masking uses $\{0, -\infty\}$ to prevent attending outside evidence
4. **Determinism**: FAISS IndexFlatIP ensures reproducible retrieval; fixed seeds for reproducible training
5. **Evidence-First**: No generation without evidence; no citation without attention

---

## 3. Supported Legal Corpus

LEXAR v1.1 supports three Indian legal statutes with section-aligned chunking:

| Statute | Code | Chunks | Coverage |
|---------|------|--------|----------|
| **Indian Penal Code** | IPC | 922 | Sections 1–511 (substantive criminal law) |
| **Code of Criminal Procedure** | CrPC | 1,409 | Sections 1–565 (procedural rules) |
| **Indian Evidence Act** | IEA | 402 | Sections 1–167 (evidentiary rules) |
| **Total** | — | **2,733** | Complete coverage of core legal framework |

**Coverage Strategy**: Sections are chunked at the section boundary (each section forms a complete chunk), preserving legal structure and enabling section-level citations.

---

## 4. Data Ingestion

### Statute Chunking

Statutes are chunked deterministically at section boundaries:

1. **Section Identification**: Parse statute into sections (e.g., "Section 302" → IPC)
2. **Boundary Preservation**: Each chunk is exactly one section, no splitting across sections
3. **Metadata Attachment**: Each chunk stores statute, section number, and full section text
4. **Determinism**: Chunking is reproducible and independent of random state

**Example**:
```
Chunk ID: ipc_302
Statute: IPC
Section: 302
Text: "Punishment for murder. —Whoever commits murder shall, if the act by which 
the death is caused is committed with the intention of causing death, or with the 
knowledge that the act is likely to cause death, be punished with death, or 
life-imprisonment..."
```

### Adding New Statutes

To ingest a new statute:

```bash
# 1. Place statute text in data/raw_docs/
# 2. Run ingestion script
python scripts/ingest_corpus.py \
  --statute-path data/raw_docs/new_statute.txt \
  --statute-code NEW_CODE \
  --output-dir data/processed_docs/

# 3. Rebuild FAISS index with new chunks
python scripts/build_ipc_crpc_iea_faiss_index.py
```

The ingestion pipeline:
- Parses statute structure (sections, subsections)
- Creates section-aligned chunks
- Encodes chunks with frozen chunk encoder (sentence-transformers/all-MiniLM-L6-v2)
- Stores chunk metadata (statute, section, text)
- Appends to FAISS index (IndexFlatIP)

---

## 5. Retrieval System

### Architecture

The retrieval system consists of two components:

1. **Query Encoder** (fine-tuned): Encodes user queries into 384-dimensional normalized vectors
2. **Chunk Encoder** (frozen): Encodes legal chunks into the same space at index time
3. **FAISS Index** (IndexFlatIP): Deterministic inner-product search for cosine similarity

### Weakly Supervised Contrastive Training

The query encoder was fine-tuned using weakly supervised contrastive learning over IPC + CrPC:

**Training Data Generation**:
- Synthetic queries generated from section titles using three templates:
  - "What is {title}?"
  - "Explain {title}"
  - "{title}"
- Total queries: 6,993 (IPC: 3,446, CrPC: 3,547)
- Positive: The source section chunk
- Negatives: Four randomly sampled chunks from the opposite statute (cross-statute negatives)

**Training Configuration**:
- **Base Model**: sentence-transformers/all-MiniLM-L6-v2
- **Loss**: InfoNCE with temperature τ = 0.05
- **Optimizer**: AdamW (LR = 2e-5)
- **Batch Size**: 16
- **Epochs**: 1
- **Train/Eval Split**: 90/10 (6,293 train / 700 eval)
- **Random Seed**: 42 (reproducibility)

**Key Design**:
- Chunk encoder remains frozen (preserves original embeddings)
- Only query encoder is fine-tuned
- Cross-statute negatives improve cross-statute disambiguation
- Low temperature (0.05) encourages sharp similarity distinctions

### Search Procedure

```python
from sentence_transformers import SentenceTransformer
import faiss

# Load fine-tuned query encoder
query_encoder = SentenceTransformer("data/models/lexar_query_encoder_v1")

# Load FAISS index
index = faiss.read_index("data/faiss_index/ipc_crpc_iea.index")

# Encode query
query_embedding = query_encoder.encode("What is the punishment for murder?")
query_embedding = query_embedding / norm(query_embedding)  # Normalize

# Retrieve top-K chunks
distances, indices = index.search(query_embedding.reshape(1, -1), k=10)

# Map indices to chunk IDs and retrieve text
for idx in indices[0]:
    chunk_id = chunk_id_mapping[idx]
    chunk_text = retrieve_chunk_text(chunk_id)
```

---

## 6. Training Results

### Retrieval Performance

The fine-tuned query encoder achieved significant improvements in cross-statute retrieval:

| Metric | Before Training | After Training | Absolute Gain | Relative Gain |
|--------|----------------|----------------|---------------|---------------|
| **Recall@1** | 16.14% | 19.57% | +3.43 pp | +21.2% |
| **Recall@5** | 29.43% | 42.00% | +12.57 pp | +42.7% |

### Interpretation

- **Recall@1** (+21.2%): Probability the correct statute section is the top-ranked result increased by 21.2%
- **Recall@5** (+42.7%): Probability the correct section appears in top-5 results increased by 42.7%
- **Cross-Statute Disambiguation**: Training with cross-statute negatives (e.g., IPC queries with CrPC negatives) improved the model's ability to distinguish between related sections across codes

### Training Dynamics

- **Final Training Loss**: 2.2580 (converged)
- **Training Duration**: ~5 minutes (CPU)
- **Loss Trajectory**: Monotonically decreasing from 2.7684 to 1.0133 across 340 batches
- **Determinism**: Fixed seed (SEED=42) ensures reproducible results

### Model Artifacts

- **Location**: `/home/garv/projects/legalrag/data/models/lexar_query_encoder_v1/`
- **Format**: HuggingFace SentenceTransformers checkpoint
- **Base**: sentence-transformers/all-MiniLM-L6-v2 (fine-tuned)
- **Size**: ~45 MB

---

## 7. Evidence Sufficiency Gating

### Overview

Evidence Sufficiency Gating is a deterministic safety mechanism that ensures LEXAR never returns answers unless there is sufficient evidential support from retrieved legal chunks. This prevents hallucination and guarantees that all answers have grounding in specific legal text.

### Mathematical Definition

$$S = \max_i A(c_i)$$

Where:
- $A(c_i)$ = attention mass assigned to evidence chunk $c_i$ (normalized so $\sum_i A(c_i) = 1$)
- $S$ = sufficiency metric (maximum attention weight)

### Gating Decision

$$\text{Proceed} = \begin{cases}
\text{TRUE} & \text{if } S \geq \tau \\
\text{FALSE} & \text{if } S < \tau
\end{cases}$$

Where:
- $\tau$ = threshold (default 0.5, configurable)
- If FALSE: Return structured refusal instead of answer

### Threshold Configuration

| Threshold | Interpretation | Use Case |
|-----------|-----------------|----------|
| **0.3** | Loose gating | Beta/experimental deployments |
| **0.5** | Moderate gating (default) | Production, balanced safety/utility |
| **0.7** | Strict gating | High-stakes legal decisions |
| **0.9** | Very strict gating | Rare special cases |

### Implementation

```python
from lexar.generation.evidence_gating import EvidenceSufficiencyGate

# Initialize with default threshold (0.5)
gate = EvidenceSufficiencyGate()

# Evaluate evidence sufficiency
passes, gate_info = gate.evaluate(
    attention_distribution={"IPC_302": 0.65, "IPC_34": 0.35},
    evidence_chunks=[...],
    query="What is the punishment for murder?",
    answer="..."
)

if passes:
    print(f"✓ Answer grounded (max attention: {gate_info['max_attention']:.1%})")
else:
    print(f"✗ Refused: {gate_info['refusal']['reason']}")
    print(f"   Gap to threshold: {gate_info['refusal']['deficit']:.1%}")
```

### Structured Refusals

When gating rejects an answer (S < τ), the system returns:

```python
{
    "answer": None,
    "refused": True,
    "refusal": {
        "reason": "Insufficient evidence",
        "max_attention": 0.35,
        "threshold": 0.5,
        "deficit": 0.15,  # Gap to threshold
        "evidence_summary": [...],
        "suggestion": "Reformulate your question to be more specific..."
    }
}
```

---

## 8. Provenance & Citations

### Token-Level Provenance

LEXAR tracks the source of each generated token via attention weights:

1. **Attention Hook Registration**: During generation, hooks are registered on decoder self-attention layers
2. **Attention Weight Capture**: Each self-attention head records weights over the evidence context
3. **Token Attribution**: For each generated token, the attention distribution is stored
4. **Evidence Mapping**: Attention indices are mapped to source chunk IDs and positions

### Attention-Based Attribution

Each generated token is attributed to evidence via the attention distribution:

```
Generated Token: "murder"
↓
Attention Distribution: {
    "IPC_302": 0.65,
    "IPC_34": 0.25,
    "IPC_296": 0.10
}
↓
Primary Source: IPC Section 302
Secondary Sources: IPC Sections 34, 296
```

### Inline Statute Citations

Generated answers include inline citations to source statutes:

```
"Punishment for murder is **death or life imprisonment** [IPC §302], 
with possibility of fine. The act must be committed with intention to 
cause death or knowledge that it is likely to cause death [IPC §302]."
```

### Token Provenance Output

In debug mode, the system returns full provenance:

```python
result["token_provenances"] = [
    {
        "token": "murder",
        "position": 12,
        "attention_distribution": {"IPC_302": 0.65, ...},
        "top_source": "IPC_302",
        "attention_score": 0.65
    },
    ...
]
```

---

## 9. End-to-End Example

### Query

```
"What evidence is required to prove murder in India?"
```

### Retrieved Chunks (Top-5)

| Rank | Statute | Section | Snippet |
|------|---------|---------|---------|
| 1 | IEA | 101 | Admissions may be used to prove any fact |
| 2 | IPC | 300 | Definition of murder |
| 3 | IEA | 24 | Relevancy of statements as to mental condition |
| 4 | IPC | 302 | Punishment for murder |
| 5 | IEA | 29 | Relevancy of facts in issue or relevant facts |

### Evidence-Constrained Generation

The decoder receives hard attention masks, allowing attention only to top-5 chunks:

```
Decoder Input: [CLS] What evidence is required to prove murder in India? [SEP] <CHUNKS 1-5>
Attention Mask: [0 0 0 0 0 0 0 (query tokens) 0 0 0 0 0 (evidence chunks) -inf (outside)]
→ Hard constraint: Cannot attend outside evidence
```

### Generated Answer with Provenance

```
"To prove murder in India, the prosecution must establish:

1. **Mens Rea (Criminal Intent)**: The accused must have intended to cause death 
   or known that their act was likely to cause death [IPC §300, IEA §24].

2. **Actus Reus (Criminal Act)**: An overt act that caused death, whether direct 
   or indirect [IPC §302, IEA §29].

3. **Causal Connection**: Proof that the act directly caused the death 
   [IEA §101 - admissions proving facts in issue].

The evidence standard requires proof beyond reasonable doubt of all elements 
[IPC §300]."
```

### Token Provenance (Debug Mode)

```
Token Provenances:
- "murder": IPC_300 (0.68), IPC_302 (0.25), IEA_101 (0.07)
- "intended": IEA_24 (0.72), IPC_300 (0.20), IEA_29 (0.08)
- "death": IPC_302 (0.80), IEA_29 (0.15), IPC_300 (0.05)
- ...

Evidence Attribution Summary:
━━━━━━━━━━━━━━━━━━━━━━━━━━
IPC_300  │████████████ 48.0%  (Definition of murder)
IPC_302  │████████ 32.0%      (Punishment for murder)
IEA_24   │████ 15.0%          (Mental condition statements)
IEA_29   │ 3.0%               (Relevant facts)
IEA_101  │ 2.0%               (Admissions)
```

### Gating Evaluation

```
max_attention = 0.80 (token "death" → IPC_302)
threshold = 0.50
decision = 0.80 >= 0.50 → PASS ✓

Answer Status: ACCEPTED (sufficient evidence)
Confidence: 0.80
```

---

## 10. Reproducibility

### Deterministic Components

| Component | Mechanism |
|-----------|-----------|
| **FAISS Index** | IndexFlatIP (no randomness in search) |
| **Query Encoder** | Fixed checkpoint, deterministic inference |
| **Chunk Encoder** | Frozen base model, no updates |
| **Attention Masking** | Deterministic logic ($\max$, $\geq$) |
| **Gating** | Deterministic comparison |

### Random Seed Management

```python
# All random sources fixed
SEED = 42
random.seed(SEED)
numpy.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

**Impact**: Retrieval rankings, training trajectories, and probabilistic evaluations are identical across runs.

### CPU-Compatible Training

The training pipeline supports CPU execution:

- **Batch Size**: 16 (fits in CPU memory)
- **Optimizer State**: Stored efficiently with gradient checkpointing
- **Training Time**: ~5 minutes per epoch on modern CPU
- **No GPU Required**: Full reproducibility on CPU or GPU

### Index Reproducibility

```bash
# Build index (deterministic)
python scripts/build_ipc_crpc_iea_faiss_index.py
# Output: data/faiss_index/ipc_crpc_iea.index (2,733 vectors, fixed ordering)

# Retrieve from index (deterministic)
query_embedding = query_encoder.encode(query)
distances, indices = index.search(query_embedding, k=10)
# Same query → same indices, same distances (every time)
```

---

## 11. Repository Structure

```
/home/garv/projects/legalrag/
├── README.md                          # This file
├── pyproject.toml                     # PEP 621 packaging configuration
├── lexar/                             # Main installable package
│   ├── __init__.py                   # Public API exports
│   ├── __version__.py                # Version metadata
│   ├── cli.py                        # CLI entry point (lexar command)
│   ├── lexar_pipeline.py             # LexarPipeline (main entry point)
│   ├── config.py                     # Configuration
│   ├── generation/                   # T5 decoder & provenance tracking
│   │   ├── attention_mask.py         # Evidence-constrained attention
│   │   ├── token_provenance.py       # Token→chunk attribution
│   │   ├── evidence_gating.py        # Sufficiency gating
│   │   ├── lexar_generator.py        # End-to-end generation
│   │   └── decoder.py                # Custom T5 decoder
│   ├── retrieval/                    # FAISS index & query encoding
│   │   ├── embedder.py               # SentenceTransformer wrapper
│   │   ├── ipc_retriever.py          # IPC retrieval
│   │   └── multi_index_retriever.py  # Multi-corpus retrieval
│   ├── reranking/                    # Cross-encoder ranking
│   │   └── cross_encoder.py          # Legal cross-encoder
│   ├── ingestion/                    # Statute parsing & chunking
│   │   ├── ipc_ingestor.py           # IPC ingestion
│   │   └── judgment_ingestor.py      # Judgment ingestion
│   ├── chunking/                     # Chunk strategies
│   │   ├── ipc_chunker.py            # Section-aligned chunking
│   │   └── generic_chunker.py        # Generic text chunking
│   ├── citation/                     # Citation rendering
│   │   ├── citation_mapper.py        # Chunk→citation mapping
│   │   └── citation_renderer.py      # Citation formatting
│   └── utils/                        # Shared utilities
├── backend/                           # Optional API server (not installed)
│   └── app/
│       ├── main.py                   # FastAPI server
│       └── api/                      # REST endpoints
├── data/                              # Data artifacts (not installed)
│   ├── raw_docs/                     # Raw statute texts
│   ├── processed_docs/               # Parsed chunks
│   ├── faiss_index/
│   │   ├── ipc_crpc_iea.index       # FAISS index (2,733 vectors)
│   │   └── ipc_crpc_iea_chunk_ids.json # Chunk ID mapping
│   └── models/
│       └── lexar_query_encoder_v1/   # Fine-tuned query encoder
├── scripts/                           # Research scripts (not installed)
│   ├── ingest_corpus.py              # Statute ingestion
│   ├── build_ipc_crpc_iea_faiss_index.py  # Index building
│   ├── train_query_encoder_v2.py     # Query encoder training
│   └── test_retrieval_validation.py  # Validation on test queries
└── evaluation/                        # Evaluation results (not installed)
```

**Key Directories**:
- **lexar/**: Pip-installable framework (core LEXAR implementation)
- **backend/**: Optional FastAPI server (requires `pip install lexar-ai[server]`)
- **data/**: Corpora, indices, and trained models (user-provided or downloaded separately)
- **scripts/**: Data preparation, training, and evaluation pipelines (for research/development)
- **evaluation/**: Benchmark results and error analysis (for research)

---

## 12. Versioning

### Current Release: v1.1.0

**Release Date**: February 2026  
**Status**: Stable Research Milestone

### What v1.1 Includes

**Core Systems**:
- Dense retrieval with fine-tuned query encoder
- Section-aligned chunking for IPC, CrPC, IEA
- Deterministic FAISS index (2,733 chunks)
- Evidence-constrained attention masking
- Token-level provenance tracking
- Evidence sufficiency gating with configurable thresholds
- Citation-aware text generation

**Training & Validation**:
- Weakly supervised contrastive training (6,993 synthetic queries)
- Recall@5 improvement: +42.7%
- Cross-statute negative sampling
- Fixed seed reproducibility

**Safety & Auditability**:
- Hard attention masking (no hallucination without evidence)
- Deterministic retrieval (IndexFlatIP)
- Structured refusals (gating)
- Full token provenance (debug mode)

### Intentionally Excluded from v1.1

**Generator Fine-Tuning**: Generator (T5-base) remains frozen; only query encoder is fine-tuned. Generator fine-tuning risks degrading evidence constraints and is deferred to v2.

**Case Law Ingestion**: Current version covers statutes only (IPC, CrPC, IEA). Case law ingestion (Indian Supreme Court judgments, High Court decisions) is planned for v1.2+.

**Heuristic Re-ranking**: v1.1 uses only cross-encoder scoring. Domain-specific heuristics (e.g., boosting recent amendments, recent judgments) are deferred.

### Versioning Strategy

- **v1.0**: Initial evidence-constrained architecture (no query encoder training)
- **v1.1**: Query encoder fine-tuning + sufficiency gating (current)
- **v1.2**: Case law ingestion + cross-statute case law retrieval
- **v2.0**: Optional generator fine-tuning (separate branch)

---

## 13. Future Work

### Short-term (v1.2)

1. **Case Law Ingestion**
   - Parse and chunk Indian Supreme Court judgments
   - Index 10K–100K judgments alongside statutes
   - Improve retrieval coverage for legal principles

2. **Multi-Modal Retrieval** (optional)
   - Support PDF parsing and OCR for legacy judgment documents
   - Maintain deterministic chunking

3. **Expanded Corpus**
   - Add state-specific criminal laws
   - Add commercial law statutes (Contracts Act, etc.)

### Medium-term (v2.0)

4. **Generator Fine-Tuning** (optional, separate branch)
   - Fine-tune T5 on legal QA pairs
   - Maintain evidence constraints via attention masking
   - Risk: May degrade faithfulness; requires extensive validation

5. **Domain-Specific Re-ranking**
   - Heuristic boosting for recent amendments
   - Authority-weighted scoring (Supreme Court > High Court > District Court)
   - Temporal relevance scoring

### Long-term

6. **Legal Knowledge Graph Integration**
   - Represent case citations as a directed graph
   - Enable cross-case reasoning
   - Maintain interpretability

7. **Continual Learning**
   - Update embeddings as new judgments/amendments appear
   - Maintain reproducibility across versions

---

## Quick Start

### Installation

**Option 1: Install from PyPI (recommended for users)**

```bash
# Install LEXAR core framework (CPU-only, lightweight)
pip install lexar-ai

# Or with PyTorch for CPU inference
pip install lexar-ai[cpu]

# Or with PyTorch for GPU inference
pip install lexar-ai[gpu]

# Optional: Install with server support for REST API
pip install lexar-ai[server]

# Optional: Install with development tools
pip install lexar-ai[dev]

# Install everything (CPU version + server + dev tools)
pip install lexar-ai[all]
```

**Why separate PyTorch?** LEXAR's dependencies (`sentence-transformers`) will pull PyTorch automatically if needed. The `[cpu]` and `[gpu]` extras are provided for explicit control.

**Option 2: Install from source (for development)**

```bash
# Clone repository
git clone https://github.com/yourusername/legalrag
cd legalrag

# Install in editable mode (CPU)
pip install -e .[cpu]

# Or install with all optional dependencies
pip install -e .[all]
```

**Note**: The pip package includes only the LEXAR framework code. To use the system, you'll need to:
1. Download or build your own legal corpus and FAISS index
2. Obtain or train a query encoder model
3. Configure paths to these resources

See the repository for pre-built indices and trained models.

### Basic Usage

```python
from lexar import LexarPipeline

# Initialize pipeline
pipeline = LexarPipeline()

# Ask a legal question
result = pipeline.answer(
    query="What is the punishment for murder under IPC?"
)

print(f"Answer: {result['answer']}")
print(f"Evidence count: {result['evidence_count']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### CLI Usage

```bash
# Show version
lexar --version

# Ask a question
lexar query "What is IPC Section 302?"

# Enable debug mode for provenance information
lexar query "What evidence is required for murder?" --debug

# Show system information
lexar info
```

### Advanced Usage (Debug Mode)

```python
from lexar import LexarPipeline

# Enable debug mode to see evidence attribution
pipeline = LexarPipeline()
result = pipeline.answer(
    query="What evidence is required to prove murder?",
    debug_mode=True
)

# View token provenance
for token_prov in result["token_provenances"]:
    print(f"{token_prov['token']}: {token_prov['top_source']} ({token_prov['attention_score']:.2f})")
```

### Development Setup

For contributors working on LEXAR development:

```bash
# Clone repository
cd /home/garv/projects/legalrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install in editable mode with dev dependencies
pip install -e .[dev]

# Run tests (if available)
pytest

# Format code
black lexar/
isort lexar/
```

### Running Validation

```bash
# Test retrieval on validation queries
python scripts/test_retrieval_validation.py
```

---

## References

### Key Papers

- **Dense Passage Retrieval**: Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", EMNLP 2020.
- **Contrastive Learning**: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020.
- **Evidence-Based NLP**: Thorne et al., "Evidence-based Fact Checking of Claims", EMNLP 2018.

### Legal Domain

- **Indian Penal Code** (1860): https://legislative.gov.in/
- **Code of Criminal Procedure** (1973): https://legislative.gov.in/
- **Indian Evidence Act** (1872): https://legislative.gov.in/

---

## Contact & Citation

**Project**: LEXAR v1.1.0 — Legal EXplainable Augmented Reasoner  
**Repository**: /home/garv/projects/legalrag/  
**Status**: Stable Research Milestone  

For questions or contributions, please refer to the repository documentation.

---

**Last Updated**: February 2026  
**Release**: v1.1.0 (Stable)
LEXAR is a **research-oriented legal reasoning system** designed for **structured and explainable legal question answering** over statutory text (e.g., IPC).

Unlike black-box generation pipelines, LEXAR emphasizes:
- explainability,
- modular reasoning,
- and strong grounding in retrieved legal sources.

---

## What is LEXAR?

LEXAR (Legal EXplainable Augmented Reasoner) is a modular framework that combines:
- retrieval over legal text,
- structured reasoning,
- and controlled generation,

to answer legal queries in a way that is **traceable and interpretable**.

This repository currently hosts **LEXAR v0.2 (Medium-Scale)** — a research milestone focused on improving reasoning depth and stability.

---

## Current Release

**Version:** `v0.2`  
**Tag:** `lexar-v0.2-medium-scale`

### Key Improvements in v0.2
- Medium-scale reasoning backbone for deeper legal inference
- Improved alignment between retrieval and generation
- Reduced hallucination on statute-based questions
- Research-friendly modular design

> This is a **research release**, not a production legal advisory system.

---

## Design Philosophy

LEXAR is built around three principles:

1. **Explainability First**  
   Reasoning steps should be inspectable, not hidden.

2. **Grounded Legal Reasoning**  
   Answers must be tied to retrieved legal text.

3. **Modularity**  
   Retrieval, reasoning, and generation are cleanly separated to support experimentation.

---

## High-Level Architecture

Each component can be independently replaced or extended for research purposes.

---

## Intended Use Cases

- Legal Question Answering (IPC / statutory reasoning)
- Explainable AI research in legal NLP
- Retrieval-Augmented Generation (RAG) experiments
- Constrained or structured decoding research

---

## Disclaimer

LEXAR is provided **for research and educational purposes only**.  
It does **not** provide legal advice and should not be used for real-world legal decision-making.

