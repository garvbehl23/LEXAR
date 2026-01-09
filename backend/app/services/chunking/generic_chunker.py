def chunk_generic_text(
    text: str,
    max_words: int = 300,
    overlap: int = 50
):
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + max_words
        chunk_text = " ".join(words[start:end])

        chunks.append({
            "chunk_id": f"Generic_{chunk_id}",
            "text": chunk_text,
            "metadata": {}
        })

        chunk_id += 1
        start += max_words - overlap

    return chunks
