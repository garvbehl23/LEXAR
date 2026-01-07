import re
from typing import List, Dict

def legal_aware_chunk(text: str, max_chars: int = 1000) -> List[Dict]:
    """
    Chunk legal text using section and clause awareness.
    """
    chunks = []
    current_chunk = ""
    chunk_id = 0

    # Split on common legal section patterns
    sections = re.split(
        r"\n(?=(Section|SECTION|\d+\.|\(\w+\)))",
        text
    )

    for section in sections:
        if len(current_chunk) + len(section) <= max_chars:
            current_chunk += section + "\n"
        else:
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip()
            })
            chunk_id += 1
            current_chunk = section + "\n"

    if current_chunk.strip():
        chunks.append({
            "chunk_id": chunk_id,
            "text": current_chunk.strip()
        })

    return chunks
