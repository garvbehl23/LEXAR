# LEXAR Architecture Overview

The system is deliberately modular and auditable.

SERVICES:
1. ingestion/
   - Legal document parsing
   - Structural chunking
2. retrieval/
   - Query encoder
   - FAISS index
3. reranker/
   - Cross-encoder relevance scoring
4. generator/
   - Decoder-only Transformer
   - Evidence-constrained attention
5. api/
   - Orchestrates end-to-end flow

DESIGN GOAL:
Errors must be localizable to a stage:
- Retrieval error
- Evidence selection error
- Reasoning error

NO END-TO-END BLACK BOXES.
