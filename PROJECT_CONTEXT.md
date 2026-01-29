# LEXAR Project Context

LEXAR (Legal EXplainable Augmented Reasoner) is a retrieval-augmented
legal question answering system designed with strict evidence grounding.

CORE PRINCIPLES:
1. Legal QA is modeled as latent-variable inference:
   P(y | q, D) ≈ P(y | q, R(q))
2. Retrieval is NOT optional — no generation without evidence.
3. Generation is constrained via HARD attention masking.
4. The decoder may attend ONLY to:
   - Retrieved legal chunks
   - The user query
5. Hallucination prevention is architectural, not post-hoc.

PIPELINE:
Query → Dense Retrieval → Evidence Re-ranking →
Context Fusion → Transformer Decoder →
Citation-Aware Output

IMPORTANT CONSTRAINTS:
- No unrestricted self-attention
- No generation from parametric memory
- No answer without citation
- Chunking respects legal structure (sections, clauses)

IMPLEMENTED VS PROPOSED:
Implemented:
- Dense retrieval (FAISS / embeddings)
- Evidence re-ranking
- Attention masking in decoder

Proposed:
- Differentiable faithfulness regularizer
- Soft evidence masking
- Legal knowledge graph integration

THIS IS NOT A GENERIC RAG SYSTEM.
