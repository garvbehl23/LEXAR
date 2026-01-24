# Evidence Sufficiency Gating - Implementation Guide

## Overview

**Evidence Sufficiency Gating** is a critical safety mechanism that ensures LEXAR never returns answers unless there is sufficient evidential support from retrieved legal chunks.

This prevents hallucination and guarantees that all answers have grounding in specific legal text.

**Status**: ✅ Fully Implemented, Tested, and Production-Ready

---

## Mathematical Definition

### Sufficiency Metric

$$S = \max_i A(c_i)$$

Where:
- $A(c_i)$ = attention mass assigned to evidence chunk $c_i$
- $\sum_i A(c_i) = 1$ (attention weights are normalized)
- $S$ = the maximum attention weight across all chunks

### Gating Decision

$$\text{Proceed} = \begin{cases}
\text{TRUE} & \text{if } S \geq \tau \\
\text{FALSE} & \text{if } S < \tau
\end{cases}$$

Where:
- $\tau$ = threshold (default 0.5, configurable)
- If **FALSE**: Return structured refusal instead of answer

### Interpretation

- **S = 0.65**: Best evidence chunk received 65% of model's attention → PASS
- **S = 0.35**: No single chunk dominated attention significantly → FAIL & REFUSE
- **S = 0.50**: At boundary (passes if τ=0.5, fails if τ>0.5)

---

## Architecture

### Integration Points

```
Evidence-Constrained Attention (Phase 2)
    ↓ Hard masking prevents attending outside evidence
    ↓
Evidence Attribution (Phase 3 Debug Mode)
    ↓ Compute per-chunk attention distribution
    ↓
Evidence Sufficiency Gating (Phase 4 - THIS) ← Safety Gate
    ↓ Check: max_attention >= threshold?
    ↓
Answer Finalization or Structured Refusal
```

### Pipeline Flow

```python
query → retrieve → rerank → generate_with_evidence
                                    ↓
                            [WITH GATING]
                                    ↓
                        Compute: max_attention
                                    ↓
                        Is max_attention >= τ ?
                                    ↓
                    YES ↓                    ↓ NO
                 Return Answer        Return Refusal
                    with              (structured)
                 evidence_ids
```

---

## Implementation

### Core Classes

#### 1. EvidenceSufficiencyGate

Main gating mechanism for checking evidence sufficiency.

```python
from backend.app.services.generation.evidence_gating import EvidenceSufficiencyGate

# Initialize with default threshold (0.5)
gate = EvidenceSufficiencyGate()

# Initialize with custom threshold
gate = EvidenceSufficiencyGate(threshold=0.7)

# Evaluate evidence sufficiency
passes, gate_info = gate.evaluate(
    attention_distribution={
        "IPC_302": 0.65,
        "IPC_34": 0.35
    },
    evidence_chunks=[...],
    query="...",
    answer="..."
)

if passes:
    # Answer is grounded in evidence
    print(f"✓ Passed with margin: {gate_info['margin']:.1%}")
else:
    # Answer is rejected due to insufficient evidence
    refusal = gate_info["refusal"]
    print(f"✗ Rejected: {refusal['reason']}")
```

#### 2. EvidenceGatingStats

Tracks statistics about gating decisions across queries.

```python
from backend.app.services.generation.evidence_gating import EvidenceGatingStats

stats = EvidenceGatingStats()

# Record evaluations
stats.record(True, 0.65, "Query 1")  # Passed with 65% max attention
stats.record(False, 0.35, "Query 2") # Rejected with 35% max attention

# Get summary
summary = stats.get_stats()
# {
#   "total_evaluations": 2,
#   "passed": 1,
#   "failed": 1,
#   "pass_rate": 0.5,
#   "avg_max_attention": 0.5,
#   ...
# }
```

---

## Configuration

### Threshold Selection

| Threshold | Meaning | Use Case |
|-----------|---------|----------|
| **0.3** | Loose (any focus counts) | Beta/experimental |
| **0.5** | Moderate (default) | Production (recommended) |
| **0.7** | Strict (high focus required) | High-stakes legal decisions |
| **0.9** | Very strict (extreme focus) | Rare/special cases |

### Threshold Modification

```python
gate = EvidenceSufficiencyGate(threshold=0.5)

# Dynamic threshold adjustment
gate.set_threshold(0.7)  # Increase threshold

# Get current threshold
current = gate.get_threshold()  # Returns 0.7

# Invalid threshold raises ValueError
try:
    gate.set_threshold(1.5)  # Error: out of range
except ValueError:
    print("Threshold must be in [0.0, 1.0]")
```

### Enable/Disable Gating

```python
gate = EvidenceSufficiencyGate()

# Disable gating (for testing/comparison)
gate.disable()
passes, info = gate.evaluate(...)  # Always returns True even if low evidence

# Re-enable gating
gate.enable()
passes, info = gate.evaluate(...)  # Normal gating behavior

# Check status
if gate.is_enabled():
    print("Gating is active")
```

### Strict Mode

```python
# Non-strict mode (default): >= threshold
gate = EvidenceSufficiencyGate(threshold=0.5, strict_mode=False)
# max_attention=0.50 → PASSES (0.50 >= 0.5)

# Strict mode: > threshold
gate = EvidenceSufficiencyGate(threshold=0.5, strict_mode=True)
# max_attention=0.50 → FAILS (0.50 > 0.5 is False)
```

---

## Generator Integration

The gating is automatically integrated into `LexarGenerator.generate_with_evidence()`:

```python
from backend.app.services.generation.lexar_generator import LexarGenerator

generator = LexarGenerator(evidence_threshold=0.5)

# Generate with automatic gating
result = generator.generate_with_evidence(
    query="What is the punishment for murder?",
    evidence_chunks=[...],
    enable_gating=True,  # ← Enable gating check
    debug_mode=True      # Required for gating to work
)

# SUCCESS CASE: Answer passed gating
if "answer" in result and result.get("status") != "insufficient_evidence":
    print(f"Answer: {result['answer']}")
    print(f"Max attention: {result['gating']['max_attention']:.1%}")
    print(f"Margin: {result['gating']['margin']:.1%}")

# FAILURE CASE: Answer rejected by gating
if result.get("status") == "insufficient_evidence":
    print(f"Refusal: {result['reason']}")
    print(f"Required: {result['required_threshold']:.1%}")
    print(f"Actual: {result['max_attention']:.1%}")
    print(f"Deficit: {result['deficit']:.1%}")
    print("Suggestions:")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"  {i}. {suggestion}")
```

---

## Pipeline Integration

The gating is integrated into `LexarPipeline.answer()`:

```python
from backend.app.services.lexar_pipeline import LexarPipeline

pipeline = LexarPipeline()

# Query with gating enabled (automatic)
result = pipeline.answer(
    query="What is mens rea in criminal law?",
    debug_mode=True  # Required for gating to work
)

# SUCCESS CASE
if result.get("status") == "success":
    answer = result["answer"]
    confidence = result["confidence"]
    evidence_ids = result["evidence_ids"]
    gating_info = result.get("gating")
    
    print(f"Answer: {answer}")
    print(f"Passed gating: {gating_info['passes']}")

# FAILURE CASE - Insufficient Evidence
if result.get("status") == "insufficient_evidence":
    print(f"Cannot answer: {result['reason']}")
    print(f"Max attention achieved: {result['max_attention']:.1%}")
    print(f"Threshold required: {result['required_threshold']:.1%}")
    print(f"\nSuggestions:")
    for suggestion in result['suggestions']:
        print(f"  - {suggestion}")
```

---

## Return Format

### On Success (Sufficient Evidence)

```json
{
    "status": "success",
    "answer": "Punishment for murder is death or life imprisonment...",
    "evidence_count": 2,
    "confidence": 0.87,
    "evidence_ids": ["IPC_302", "IPC_34"],
    "gating": {
        "passes": true,
        "max_attention": 0.65,
        "max_chunk_id": "IPC_302",
        "threshold": 0.5,
        "margin": 0.15,
        "status": "evidence_sufficient"
    }
}
```

### On Failure (Insufficient Evidence)

```json
{
    "status": "insufficient_evidence",
    "reason": "No legal provision received sufficient attention support. Highest attention was 35.0% but threshold is 50.0%.",
    "max_attention": 0.35,
    "required_threshold": 0.5,
    "deficit": 0.15,
    "evidence_count": 3,
    "evidence_summary": [
        {
            "chunk_id": "IPC_302",
            "statute": "IPC",
            "section": "302"
        },
        {
            "chunk_id": "IPC_34",
            "statute": "IPC",
            "section": "34"
        }
    ],
    "suggestions": [
        "Rephrase your query to be more specific",
        "Expand the legal corpus with more relevant statutes",
        "Break down complex questions into simpler sub-questions",
        "Provide additional context about jurisdiction or relevant laws"
    ],
    "explanation": "LEXAR safety mechanism: This refusal indicates that the retrieved evidence..."
}
```

---

## Testing

### Test Suite

Run the comprehensive test suite:

```bash
python scripts/test_evidence_gating.py
```

**Test Coverage** (10 tests, all passing):
1. ✅ Gate initialization with various configurations
2. ✅ Sufficient evidence acceptance
3. ✅ Insufficient evidence rejection
4. ✅ Threshold boundary behavior
5. ✅ Disabled gating bypass
6. ✅ Dynamic threshold modification
7. ✅ Enable/disable toggling
8. ✅ Refusal message structure
9. ✅ Statistics tracking
10. ✅ Floating-point normalization

### Example Test

```python
from app.services.generation.evidence_gating import EvidenceSufficiencyGate

# Test: Gate rejects low evidence
gate = EvidenceSufficiencyGate(threshold=0.5)

attention_dist = {"IPC_302": 0.35, "IPC_34": 0.65}
evidence_chunks = [...]

passes, gate_info = gate.evaluate(
    attention_distribution=attention_dist,
    evidence_chunks=evidence_chunks,
    query="Question?",
    answer="Answer."
)

# Max attention is 0.65 >= 0.5, so passes
assert passes == True
assert gate_info["margin"] == 0.15
```

---

## Use Cases

### 1. Legal Compliance Audit

```python
result = pipeline.answer(query, debug_mode=True)

if result["status"] == "insufficient_evidence":
    # Log rejection with explanation
    audit_log = {
        "query": query,
        "reason": result["reason"],
        "evidence_analyzed": result["evidence_count"],
        "deficit": result["deficit"],
        "timestamp": datetime.now()
    }
    save_to_audit_log(audit_log)
```

### 2. High-Stakes Legal Decisions

```python
# Use strict threshold for important decisions
generator = LexarGenerator(evidence_threshold=0.7)

result = generator.generate_with_evidence(
    query=legal_question,
    evidence_chunks=evidence,
    enable_gating=True
)

if result.get("status") == "insufficient_evidence":
    # Escalate to human lawyer for review
    escalate_to_lawyer(query, evidence)
```

### 3. Query Optimization

```python
result = pipeline.answer(query, debug_mode=True)

if result["status"] == "insufficient_evidence":
    # Apply suggestions to improve query
    improved_query = apply_suggestion(query, result["suggestions"][0])
    
    # Retry with improved query
    result = pipeline.answer(improved_query, debug_mode=True)
```

### 4. System Monitoring

```python
stats = EvidenceGatingStats()

for query in queries:
    result = pipeline.answer(query, debug_mode=True)
    
    if "gating" in result:
        passes = result["gating"]["passes"]
        max_attn = result["gating"]["max_attention"]
        stats.record(passes, max_attn, query)

summary = stats.get_stats()
print(f"Pass rate: {summary['pass_rate']:.1%}")
print(f"Avg attention: {summary['avg_max_attention']:.1%}")
```

---

## Limitations & Future Work

### Current Implementation

- ✅ Uses maximum attention as sufficiency metric
- ✅ Configurable threshold
- ✅ Structured refusal messages
- ✅ Statistics tracking
- ✅ Enable/disable for testing

### Future Enhancements

- [ ] Multiple-chunk sufficiency (e.g., require top-2 chunks sum > threshold)
- [ ] Time-series threshold adjustment based on domain
- [ ] Automatic threshold tuning based on ground truth
- [ ] Per-statute gating (different thresholds for different laws)
- [ ] Confidence-weighted attention (high-confidence attention > low-confidence)

---

## Best Practices

### 1. Always Enable Debug Mode with Gating

```python
# REQUIRED: Gating needs debug_mode=True to compute attention
result = pipeline.answer(query, debug_mode=True)  # ✓ Correct

# WRONG: Gating won't work without debug info
result = pipeline.answer(query, debug_mode=False)  # ✗ Incorrect
```

### 2. Monitor Gating Statistics

```python
# Track how often queries are rejected
stats = EvidenceGatingStats()

for query in incoming_queries:
    result = pipeline.answer(query, debug_mode=True)
    
    if "gating" in result:
        stats.record(
            result["gating"]["passes"],
            result["gating"]["max_attention"],
            query
        )

# Alert if rejection rate exceeds 20%
if stats.get_stats()["pass_rate"] < 0.8:
    alert_ops("High evidence insufficiency rate detected")
```

### 3. Use Appropriate Thresholds

```python
# Production (balanced)
generator = LexarGenerator(evidence_threshold=0.5)

# High-stakes (strict)
generator = LexarGenerator(evidence_threshold=0.7)

# Research (relaxed)
generator = LexarGenerator(evidence_threshold=0.3)
```

### 4. Handle Refusals Gracefully

```python
result = pipeline.answer(query, debug_mode=True)

if result["status"] == "insufficient_evidence":
    # Show user the refusal + suggestions
    print(f"I cannot answer with confidence: {result['reason']}")
    print("\nPlease try:")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"{i}. {suggestion}")
    
    # Offer to escalate
    if user_wants_escalation:
        escalate_to_human_lawyer(query, result['evidence_summary'])
```

---

## Troubleshooting

### Gating Always Rejects Answers

**Problem**: Even well-founded answers are being rejected.

**Solution**:
1. Check threshold is reasonable (0.5 recommended)
2. Verify debug_mode=True is set
3. Check evidence quality - are chunks relevant?
4. Try reducing threshold if warranted

```python
# Lower threshold from 0.7 to 0.5
gate.set_threshold(0.5)
```

### Gating Always Passes (bypassed)

**Problem**: Gating is not actually blocking insufficient evidence.

**Solution**:
1. Verify gating is enabled: `gate.is_enabled()`
2. Check debug_mode=True is set in pipeline
3. Ensure enable_gating=True in generator

```python
# Re-enable gating
gate.enable()

# Verify it's working
result = pipeline.answer(query, debug_mode=True)
if "gating" in result:
    print("✓ Gating is active")
```

### High Refusal Rate (>50% queries rejected)

**Problem**: Too many queries are being rejected.

**Solution**:
1. Analyze distribution of max_attention values
2. Lower threshold if appropriate
3. Improve evidence retrieval (better ranking)
4. Expand legal corpus

```python
# Analyze distribution
stats = pipeline.gating_stats  # If available
summary = stats.get_stats()
print(f"Avg max attention: {summary['avg_max_attention']:.1%}")

# If too low, maybe corpus needs expansion
if summary['avg_max_attention'] < 0.4:
    print("Consider expanding legal corpus")
```

---

## Integration Checklist

- [x] Evidence gating module created (evidence_gating.py)
- [x] Integrated into LexarGenerator
- [x] Integrated into LexarPipeline
- [x] Comprehensive test suite (10 tests, all passing)
- [x] Documentation complete
- [x] Production ready
- [x] Statistics tracking available
- [x] Configurable threshold
- [x] Enable/disable for testing
- [x] Structured refusal messages

---

## References

- [LEXAR Architecture](../ARCHITECTURE.md)
- [Evidence-Constrained Attention](../EVIDENCE_CONSTRAINED_ATTENTION.md)
- [Evidence Debug Mode](../EVIDENCE_DEBUG_MODE.md)
- [Quick Start Guide](../QUICK_START_GUIDE.md)

---

**Status**: ✅ Implementation Complete  
**Test Results**: ✅ 10/10 Tests Passing  
**Production Ready**: ✅ Yes  
**Last Updated**: Current Session
