import sys
import os
import math

# Ensure project root is on sys.path so `backend` package is importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.services.generation.token_provenance import (
    TokenProvenanceTracker,
    create_token_to_chunk_mapping,
)


class DummyTokenizer:
    def __init__(self):
        pass

    def encode(self, text, add_special_tokens=False):
        # Very simple tokenization: split by spaces
        return text.split()

    def convert_ids_to_tokens(self, tid):
        # In tests, we pass raw ids as strings already
        return str(tid)


def test_chunk_mapping_basic():
    tokenizer = DummyTokenizer()
    chunks = [
        {"chunk_id": "IPC_302", "text": "Punishment for murder is death"},
        {"chunk_id": "IPC_34", "text": "Acts done by several persons"},
    ]
    mapping = create_token_to_chunk_mapping(chunks, tokenizer)
    # First chunk has 5 words, second has 5 words (space-split)
    # Positions 0-4 -> IPC_302, 5-9 -> IPC_34
    assert mapping[0] == "IPC_302"
    assert mapping[4] == "IPC_302"
    assert mapping[5] == "IPC_34"
    assert mapping[9] == "IPC_34"


def test_provenance_single_layer_simple():
    import numpy as np

    tokenizer = DummyTokenizer()
    chunks = [
        {"chunk_id": "A", "text": "a a a"},       # 3 tokens: positions 0,1,2
        {"chunk_id": "B", "text": "b b b b"},    # 4 tokens: positions 3,4,5,6
    ]

    token_to_chunk = create_token_to_chunk_mapping(chunks, tokenizer)
    tracker = TokenProvenanceTracker(
        token_ids_to_chunk_ids=token_to_chunk,
        secondary_threshold=0.05,
        track_multi_layer=False,
        enable_tracking=True,
    )

    # Generated tokens (simulate ids as strings)
    gen_tokens = ["Y0", "Y1", "Y2"]
    for t in gen_tokens:
        tracker.record_token(t)

    # Attention for 3 generated tokens over 7 input positions
    # We'll make token 0 attend mostly to chunk A (positions 0-2)
    # token 1 attend mostly to chunk B (positions 3-6)
    # token 2 split attention
    attn = np.zeros((3, 7), dtype=float)
    # Token 0: strong on A
    attn[0, 0:3] = [0.2, 0.4, 0.3]
    attn[0, 3:7] = [0.025, 0.025, 0.025, 0.025]
    # Token 1: strong on B
    attn[1, 0:3] = [0.05, 0.05, 0.05]
    attn[1, 3:7] = [0.2, 0.2, 0.2, 0.25]
    # Token 2: mixed
    attn[2, 0:3] = [0.15, 0.10, 0.05]
    attn[2, 3:7] = [0.15, 0.15, 0.15, 0.25]

    tracker.track_layer_attention(layer_idx=0, attention_weights=attn, attention_heads=None)
    provs = tracker.compute_provenances()

    assert len(provs) == 3
    # Token 0 -> A
    assert provs[0].supporting_chunk == "A"
    assert math.isclose(provs[0].attention_mass, 0.9, rel_tol=1e-5)
    # Token 1 -> B
    assert provs[1].supporting_chunk == "B"
    assert math.isclose(provs[1].attention_mass, 0.85, rel_tol=1e-5)
    # Token 2 -> B (sum B ~0.7 vs A ~0.3)
    assert provs[2].supporting_chunk == "B"
    # Confidence should be primary / total (total is sum over A+B)
    assert 0.0 <= provs[2].confidence <= 1.0


if __name__ == "__main__":
    print("Running token provenance tests...")
    test_chunk_mapping_basic()
    print("✓ chunk_mapping_basic passed")
    test_provenance_single_layer_simple()
    print("✓ provenance_single_layer_simple passed")
    print("ALL TOKEN PROVENANCE TESTS PASSED ✓")
