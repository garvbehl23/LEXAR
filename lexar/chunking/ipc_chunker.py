import re

def chunk_ipc_by_section(text: str):
    """
    Robust IPC chunking based on numeric section headers (e.g., '302. Punishment for murder.')
    Handles real IPC PDFs with collapsed layout and table of contents.
    """

    # 1. Remove table of contents / arrangement of sections
    toc_pattern = r"ARRANGEMENT OF SECTIONS.*?CHAPTER I"
    text = re.sub(toc_pattern, "CHAPTER I", text, flags=re.DOTALL)

    # 2. Match section headers like '302. Punishment for murder.'
    section_pattern = r"(?<!\d)(\d{1,3})\.\s+[A-Z][^\n]{5,200}"

    matches = list(re.finditer(section_pattern, text))

    chunks = []

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        section_number = match.group(1)
        section_text = text[start:end].strip()

        chunks.append({
            "chunk_id": f"Section {section_number}",
            "text": section_text,
            "metadata": {
                "statute": "IPC",
                "section": section_number,
                "jurisdiction": "India"
            }
        })

    return chunks
def chunk_ipc_text(text: str):
    """
    Stable IPC chunking entrypoint for ingestion pipeline.
    """
    return chunk_ipc_by_section(text)
