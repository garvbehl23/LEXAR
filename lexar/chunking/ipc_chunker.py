"""
IPC section-level chunker.

Splits IPC PDF text on numeric section headers (e.g. "378. Theft.—…").
Deduplicates by section number, keeping the version with most content.
Discards sections shorter than 50 chars (TOC stubs).
"""
import re


# Matches a section header line: a 1-3 digit number, a dot, a space, then text
_SECTION_HEADER = re.compile(
    r"(?m)^(?P<num>\d{1,3})\.\s+[A-Z][^\n]{3,}"
)

# Remove "Arrangement of Sections" table-of-contents block
_TOC_RE = re.compile(
    r"ARRANGEMENT\s+OF\s+SECTIONS.*?(?=CHAPTER\s+[IVX]+\s*\nPREAMBLE|\bWHEREAS\b)",
    re.DOTALL | re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Normalize PDF extraction artifacts."""
    # Remove form-feed characters
    text = text.replace("\x0c", "\n")
    # Collapse runs of blank lines to single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are purely page numbers (e.g. "  42 " alone on a line)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    return text


def chunk_ipc_by_section(text: str) -> list[dict]:
    """
    Parse IPC text into one dict per section.

    Returns list of:
        {
            "chunk_id": "Section 378",
            "text": "378. Theft.—Whoever …",
            "metadata": {"statute": "IPC", "section": "Section 378", "section_number": 378}
        }

    Duplicates (TOC stub + body) are collapsed — the longer version wins.
    Chunks < 50 chars are discarded.
    """
    text = _clean_text(text)

    # Strip TOC block so its "378. Theft." stub lines don't produce false sections
    text = _TOC_RE.sub("", text)

    matches = list(_SECTION_HEADER.finditer(text))
    if not matches:
        return []

    raw: dict[int, str] = {}  # section_number → text

    for i, m in enumerate(matches):
        sec_num = int(m.group("num"))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # Keep the longer version when duplicates exist
        if sec_num not in raw or len(body) > len(raw[sec_num]):
            raw[sec_num] = body

    chunks = []
    for sec_num in sorted(raw):
        body = raw[sec_num]
        if len(body) < 50:
            continue
        chunks.append({
            "chunk_id": f"Section {sec_num}",
            "text": body,
            "metadata": {
                "statute": "IPC",
                "section": f"Section {sec_num}",
                "section_number": sec_num,
                "jurisdiction": "India",
            },
        })

    return chunks


def chunk_ipc_text(text: str) -> list[dict]:
    """Stable entrypoint used by ingestion pipeline."""
    return chunk_ipc_by_section(text)
