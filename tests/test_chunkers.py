"""Tests for chunking modules."""
import pytest
from lexar.chunking.ipc_chunker import chunk_ipc_by_section
from lexar.chunking.generic_chunker import chunk_generic_text
from lexar.chunking.statute_chunker import chunk_statute_text


SAMPLE_IPC = """
ARRANGEMENT OF SECTIONS
Section 1.  Title and extent of operation of the Code.
Section 2.  Punishment of offences committed within India.

CHAPTER I
INTRODUCTION
1. Title and extent of operation of the Code.
This Act shall be called the Indian Penal Code.

2. Punishment of offences committed within India.
Every person shall be liable to punishment under this Code.

302. Punishment for murder.
Whoever commits murder shall be punished with death, or imprisonment for life.

304. Punishment for culpable homicide not amounting to murder.
Whoever commits culpable homicide shall be punished with imprisonment.
"""


def test_ipc_chunker_returns_list():
    chunks = chunk_ipc_by_section(SAMPLE_IPC)
    assert isinstance(chunks, list)


def test_ipc_chunker_extracts_sections():
    chunks = chunk_ipc_by_section(SAMPLE_IPC)
    assert len(chunks) > 0


def test_ipc_chunker_section_302():
    chunks = chunk_ipc_by_section(SAMPLE_IPC)
    ids = [c["chunk_id"] for c in chunks]
    assert any("302" in cid for cid in ids), f"Section 302 not found in: {ids}"


def test_ipc_chunk_has_required_fields():
    chunks = chunk_ipc_by_section(SAMPLE_IPC)
    assert len(chunks) > 0
    chunk = chunks[0]
    assert "chunk_id" in chunk
    assert "text" in chunk
    assert "metadata" in chunk
    assert chunk["metadata"].get("statute") == "IPC"


def test_generic_chunker_basic():
    text = " ".join(["word"] * 500)
    chunks = chunk_generic_text(text, max_words=100, overlap=10)
    assert len(chunks) > 1
    for chunk in chunks:
        words = chunk["text"].split()
        assert len(words) <= 100


def test_generic_chunker_overlap():
    words = list(map(str, range(200)))
    text = " ".join(words)
    chunks = chunk_generic_text(text, max_words=50, overlap=10)
    assert len(chunks) >= 4


def test_generic_chunker_chunk_ids():
    chunks = chunk_generic_text("some text here", max_words=10)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"Generic_{i}"


def test_statute_chunker_returns_list():
    chunks = chunk_statute_text("Section 1. Short title.\nThis is the act.\n", statute_name="TestAct")
    assert isinstance(chunks, list)


def test_generic_chunker_empty_text():
    chunks = chunk_generic_text("")
    assert chunks == []
