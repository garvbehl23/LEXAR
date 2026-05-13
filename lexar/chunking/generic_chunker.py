def chunk_generic_text(
    text: str,
    max_words: int = 300,
    overlap: int = 50,
) -> list:
    if not text or not text.strip():
        return []

    words = text.split()
    chunks = []
    stride = max(1, max_words - overlap)
    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "chunk_id": f"Generic_{chunk_id}",
            "text": chunk_text,
            "metadata": {},
        })
        chunk_id += 1
        start += stride

    return chunks
