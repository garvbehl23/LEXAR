import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.app.services.chunking.statute_chunker import chunk_statute_text


def test_basic_parsing():
    text = (
        "THE SAMPLE ACT, 1900\n\n"
        "CHAPTER I — PRELIMINARY\n"
        "1. Short title.\n"
        "(1) This Act may be called the Sample Act.\n"
        "(2) It extends to India.\n\n"
        "2. Definitions.\n"
        "(a) 'Court' means any civil court.\n"
        "(b) 'Judge' includes any person authorized.\n"
    )

    chunks = chunk_statute_text(text, statute_name="Sample Act 1900", year=1900)
    assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"

    # First chunk should be section 1, subsection 1
    c0 = chunks[0]
    assert c0["metadata"]["section"] == "1"
    assert c0["metadata"]["subsection"] == "1"
    assert c0["metadata"]["clause"] is None
    assert c0["metadata"]["chapter_roman"] == "I"
    assert c0["metadata"]["year"] == 1900

    # Second chunk should be section 1, subsection 2
    c1 = chunks[1]
    assert c1["metadata"]["section"] == "1"
    assert c1["metadata"]["subsection"] == "2"

    # Third chunk should be section 2, clause a
    c2 = chunks[2]
    assert c2["metadata"]["section"] == "2"
    assert c2["metadata"]["subsection"] is None
    assert c2["metadata"]["clause"] == "a"

    # Fourth chunk should be section 2, clause b
    c3 = chunks[3]
    assert c3["metadata"]["section"] == "2"
    assert c3["metadata"]["clause"] == "b"

    print("ALL STATUTE CHUNKER TESTS PASSED ✓")


if __name__ == "__main__":
    test_basic_parsing()
