# LEXAR Quick Start Guide

## Overview
This guide shows how to use LEXAR's evidence-constrained generation and debug mode for legal question-answering.

---

## Installation

```bash
# Navigate to project
cd /home/garv/projects/legalrag

# Activate virtual environment
source venv/bin/activate

# Install dependencies (if needed)
pip install -r backend/requirements.txt
```

---

## Basic Usage

### 1. Simple Question-Answering

```python
from backend.app.services.lexar_pipeline import LexarPipeline

# Initialize pipeline
pipeline = LexarPipeline()

# Ask a legal question
result = pipeline.answer(
    query="What is the punishment for murder under IPC?"
)

# Print the answer
print(f"Answer: {result['answer']}")
print(f"Evidence count: {result['evidence_count']}")
print(f"Confidence: {result['confidence']:.2f}")
```

**Output**:
```
Answer: Punishment for murder is death or life imprisonment, 
        with possibility of fine under IPC Section 302.
Evidence count: 2
Confidence: 0.87
```

---

## Advanced Usage: Debug Mode

### 2. See Which Evidence Was Used (Debug Mode)

```python
from backend.app.services.lexar_pipeline import LexarPipeline

pipeline = LexarPipeline()

# Ask question WITH debug mode enabled
result = pipeline.answer(
    query="What is the punishment for murder?",
    debug_mode=True  # ← Enable debug mode
)

# View the answer
print(f"Answer: {result['answer']}\n")

# View which evidence chunks were attended to
print("Evidence Attribution:")
print(result["debug"]["attention_visualization"])
```

**Output**:
```
Answer: Punishment for murder is death or life imprisonment...

Evidence Attribution:
Attention Distribution:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IPC_302 │████████████████ 65.0%
IPC_34  │██████ 25.0%
IPC_503 │██ 10.0%
```

---

## Understanding Debug Output

### Supporting Chunks

```python
# Get chunks that contributed to the answer (ranked by importance)
for chunk in result["debug"]["supporting_chunks"]:
    print(f"\n{chunk['chunk_id']}")
    print(f"  Statute: {chunk['metadata']['statute']}")
    print(f"  Section: {chunk['metadata']['section']}")
    print(f"  Importance: {chunk['attention_percentage']:.1f}%")
    print(f"  Text: {chunk['text'][:100]}...")
```

**Output**:
```
IPC_302
  Statute: IPC
  Section: 302
  Importance: 65.0%
  Text: Punishment for murder is death or life imprisonment, or ...

IPC_34
  Statute: IPC
  Section: 34
  Importance: 25.0%
  Text: Acts of several persons in furtherance of common intention...
```

---

## Layer-Wise Attention Analysis

### Understanding How Focus Evolves

```python
# See how model attention changes through decoder layers
layer_attention = result["debug"]["layer_wise_attention"]

print("Layer-wise Attention Evolution:")
print("=" * 50)

for layer in range(6):
    print(f"\nLayer {layer}:")
    for chunk_id, weight in sorted(
        layer_attention[layer].items(), 
        key=lambda x: x[1], 
        reverse=True
    ):
        bar = "█" * int(weight * 20)
        print(f"  {chunk_id:12s} {bar} {weight:.1%}")
```

**Output**:
```
Layer-wise Attention Evolution:
==================================================

Layer 0:
  IPC_302      ██████████████ 70.0%
  IPC_34       ██████ 30.0%

Layer 1:
  IPC_302      ███████████ 60.0%
  IPC_34       ████████ 40.0%

Layer 2:
  IPC_302      ██████████████ 65.0%
  IPC_34       ███████ 35.0%

[...]

Layer 5:
  IPC_302      ██████████████ 62.0%
  IPC_34       ████████ 38.0%
```

---

## Use Case Examples

### Example 1: Legal Compliance Audit

```python
import json
from datetime import datetime

pipeline = LexarPipeline()

query = "What are the penalties for criminal negligence?"
result = pipeline.answer(query, debug_mode=True)

# Create audit trail
audit_log = {
    "timestamp": datetime.now().isoformat(),
    "query": query,
    "answer": result["answer"],
    "evidence_used": [
        {
            "statute": c["metadata"]["statute"],
            "section": c["metadata"]["section"],
            "importance": f"{c['attention_percentage']:.1f}%"
        }
        for c in result["debug"]["supporting_chunks"]
    ],
    "confidence": result["confidence"]
}

# Save audit log
with open("audit_logs/query_log.json", "a") as f:
    f.write(json.dumps(audit_log) + "\n")

print("✓ Audit trail saved")
```

### Example 2: Validating Evidence Relevance

```python
pipeline = LexarPipeline()

result = pipeline.answer(
    query="What is mens rea in criminal law?",
    debug_mode=True
)

# Check if retrieved evidence is actually relevant
print("Evidence Relevance Analysis:")
print("=" * 50)

threshold = 0.05  # Consider chunks with >5% attention as relevant

for chunk in result["debug"]["supporting_chunks"]:
    importance = chunk["attention_percentage"] / 100.0
    
    if importance > threshold:
        status = "✓ RELEVANT"
    else:
        status = "⚠ LOW IMPACT"
    
    print(f"\n{status}: {chunk['chunk_id']}")
    print(f"  Importance: {chunk['attention_percentage']:.1f}%")
    print(f"  Statute: {chunk['metadata']['statute']}/{chunk['metadata']['section']}")
```

### Example 3: Debugging Incorrect Answers

```python
pipeline = LexarPipeline()

query = "What is the statute of limitations for theft?"
result = pipeline.answer(query, debug_mode=True)

# If answer is incorrect, check which chunks were used
print("Debugging Information:")
print("=" * 50)
print(f"Query: {query}")
print(f"\nGenerated Answer: {result['answer']}")
print(f"\nTop-3 Attended Chunks:")

for i, chunk in enumerate(result["debug"]["supporting_chunks"][:3], 1):
    print(f"\n{i}. {chunk['chunk_id']} ({chunk['attention_percentage']:.1f}%)")
    print(f"   Statute: {chunk['metadata']['statute']}/{chunk['metadata']['section']}")
    print(f"   Preview: {chunk['text'][:80]}...")

# Hypothesis: Wrong chunks being attended?
# Analysis: Look for statute-related chunks in top-3
# Action: Review evidence retrieval for this query
```

---

## Performance Considerations

### Debug Mode Overhead

- **Generation time**: +5-10% when `debug_mode=True`
- **Memory usage**: Minimal (attention weights stored on GPU)
- **Safe for production**: Yes, can be toggled per-query

### Best Practices

```python
# 1. Use debug mode selectively (not for every query)
if user_wants_explanation:
    result = pipeline.answer(query, debug_mode=True)
else:
    result = pipeline.answer(query)  # Faster

# 2. Batch processing without debug mode
for query in queries:
    result = pipeline.answer(query)  # Faster

# 3. Enable debug only when needed (legal queries, audits)
if query_category == "legal_compliance":
    result = pipeline.answer(query, debug_mode=True)
```

---

## Return Format

### Without Debug Mode
```python
result = pipeline.answer(query)

{
    "answer": "Punishment for murder is...",
    "evidence_count": 2,
    "confidence": 0.87,
    "status": "success",
    "evidence_ids": ["IPC_302", "IPC_34"],
    "provenance": [...]  # If return_provenance=True
}
```

### With Debug Mode
```python
result = pipeline.answer(query, debug_mode=True)

{
    "answer": "Punishment for murder is...",
    "evidence_count": 2,
    "confidence": 0.87,
    "status": "success",
    "evidence_ids": ["IPC_302", "IPC_34"],
    "debug": {  # ← New field
        "attention_distribution": {
            "IPC_302": 0.65,
            "IPC_34": 0.25,
            "IPC_503": 0.10
        },
        "supporting_chunks": [
            {
                "chunk_id": "IPC_302",
                "text": "...",
                "attention_weight": 0.65,
                "attention_percentage": 65.0,
                "metadata": {"statute": "IPC", "section": "302", ...}
            },
            ...
        ],
        "attention_visualization": "IPC_302 │████... 65.0%",
        "layer_wise_attention": {
            0: {"IPC_302": 0.70, ...},
            ...
        }
    }
}
```

---

## Parameters

### pipeline.answer()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | Required | Legal question to answer |
| `has_user_docs` | bool | False | Include user-uploaded documents |
| `top_k` | int | 5 | Number of evidence chunks to retrieve |
| `return_provenance` | bool | False | Include provenance information |
| `debug_mode` | bool | False | Enable evidence attribution and visualization |

### Example with All Parameters

```python
result = pipeline.answer(
    query="What is the punishment for embezzlement?",
    has_user_docs=True,           # Include user documents
    top_k=10,                      # Retrieve top-10 chunks
    return_provenance=True,        # Include provenance
    debug_mode=True                # Enable debug mode
)
```

---

## Troubleshooting

### Issue: Debug mode returns empty attention distribution

**Cause**: Evidence chunks too small or non-overlapping

**Solution**: 
```python
# Check evidence chunks
result = pipeline.answer(query, debug_mode=True)
print(f"Evidence chunks: {len(result['evidence_ids'])}")
print(f"Top chunk attention: {max(result['debug']['attention_distribution'].values())}")
```

### Issue: All chunks have equal attention

**Cause**: Query too general; answer draws equally from all sources

**Solution**: 
```python
# Be more specific with query
query = "What is the punishment for intentional murder vs accidental killing?"
result = pipeline.answer(query, debug_mode=True)
```

### Issue: Expected chunk missing from supporting_chunks

**Cause**: Chunk attention below visualization threshold

**Solution**: 
```python
# Check full attention distribution
full_attention = result["debug"]["attention_distribution"]
print(full_attention)  # Shows all chunks including low-attention ones
```

---

## Testing

### Run Test Suite

```bash
# Test evidence constraints
python scripts/test_evidence_constrained_attention.py

# Test debug mode
python scripts/test_debug_mode.py

# Test end-to-end pipeline
python scripts/test_lexar_pipeline.py
```

### Example Test Output

```
================================================================================
EVIDENCE-DEBUG MODE TEST SUITE
================================================================================

✓ TEST 1: Debug mode output structure
✓ TEST 2: Attention distribution computation
✓ TEST 3: Attention visualization
✓ TEST 4: Supporting chunks ranking
✓ TEST 5: Layer-wise attention analysis
✓ TEST 6: Generator integration
✓ TEST 7: Pipeline integration

ALL TESTS PASSED ✓
```

---

## Advanced Configuration

### Custom Evidence Chunks

```python
from backend.app.services.generation.lexar_generator import LexarGenerator

generator = LexarGenerator()

# Provide custom evidence chunks
evidence = [
    {
        "chunk_id": "STATUTE_1",
        "text": "Custom statute text...",
        "metadata": {"statute": "Custom Statute", "section": "1"}
    },
    {
        "chunk_id": "STATUTE_2",
        "text": "More statute text...",
        "metadata": {"statute": "Custom Statute", "section": "2"}
    }
]

result = generator.generate_with_evidence(
    query="Question about custom statutes?",
    evidence_chunks=evidence,
    debug_mode=True
)
```

---

## API Reference

### LexarPipeline

```python
from backend.app.services.lexar_pipeline import LexarPipeline

pipeline = LexarPipeline()

# Main method
result = pipeline.answer(
    query: str,
    has_user_docs: bool = False,
    top_k: int = 5,
    return_provenance: bool = False,
    debug_mode: bool = False
) -> Dict

# Returns: {answer, evidence_count, confidence, status, evidence_ids, debug?, provenance?}
```

### LexarGenerator

```python
from backend.app.services.generation.lexar_generator import LexarGenerator

generator = LexarGenerator()

# Main generation method
result = generator.generate_with_evidence(
    query: str,
    evidence_chunks: List[Dict],
    max_tokens: int = 150,
    temperature: float = 0.7,
    debug_mode: bool = False
) -> Dict

# Returns: {answer, provenance, evidence_token_count, debug?}
```

---

## Documentation Links

- [Project Context](PROJECT_CONTEXT.md) - LEXAR principles
- [Architecture](ARCHITECTURE.md) - System design
- [Evidence-Constrained Attention](EVIDENCE_CONSTRAINED_ATTENTION.md) - Technical details
- [Evidence Debug Mode](EVIDENCE_DEBUG_MODE.md) - Debug mode guide
- [Complete Project Summary](LEXAR_HARDENING_PROJECT_SUMMARY.md) - Full project overview

---

## Support

For issues or questions:
1. Check the relevant documentation file
2. Run test suites to verify installation
3. Review troubleshooting section above
4. Check examples in `scripts/` directory

---

**Last Updated**: Current Session  
**Version**: 1.0 - Production Ready
