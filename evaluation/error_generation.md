## Generation Artifacts

### Observation
Minor token artifacts (e.g., stray punctuation) appear in generated answers.

### Cause
Structured prompts containing indexed evidence markers may introduce
token-boundary artifacts in sequence-to-sequence decoders.

### Impact
Artifacts are cosmetic and do not affect legal correctness or faithfulness.
