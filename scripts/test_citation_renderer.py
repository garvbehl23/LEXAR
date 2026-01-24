import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.services.citation.citation_renderer import CitationRenderer

class DummyTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        # For test, ids are strings representing words
        return " ".join(ids)


def test_build_spans_and_inline():
    # Simulate generated ids as simple words
    gen_ids = [
        "Punishment", "for", "murder", "is", "death", "or", "life", "imprisonment"
    ]
    # Token provenance: same chunk for the whole sentence
    token_provenances = [
        {"token": w, "position": i, "supporting_chunk": "IPC_302", "confidence": 0.8}
        for i, w in enumerate(gen_ids)
    ]
    evidence_chunks = [{
        "chunk_id": "IPC_302",
        "text": "Section 302: Punishment for murder...",
        "metadata": {"statute": "IPC", "section": "302"}
    }]
    renderer = CitationRenderer()
    spans = renderer.build_spans(token_provenances, gen_ids, DummyTokenizer(), evidence_chunks)
    assert len(spans) == 1
    assert spans[0].chunk_id == "IPC_302"
    assert spans[0].section == "302"
    inline = renderer.render_inline(spans)
    assert inline.endswith("(IPC §302)")


if __name__ == "__main__":
    print("Running citation renderer tests...")
    test_build_spans_and_inline()
    print("✓ build_spans_and_inline passed")
    print("ALL CITATION RENDERER TESTS PASSED ✓")
