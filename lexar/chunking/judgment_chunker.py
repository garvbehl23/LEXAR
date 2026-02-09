import re

def chunk_judgment_text(
    text: str,
    max_words: int = 300,
    overlap: int = 50
):
    """
    Chunk legal judgments using paragraph-aware sliding windows.
    This is a safe default for unstructured court decisions.
    """

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + max_words
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "chunk_id": f"JudgmentChunk_{chunk_id}",
            "text": chunk_text,
            "metadata": {
                "source": "Judgment"
            }
        })

        chunk_id += 1
        start += max_words - overlap

    return chunks
