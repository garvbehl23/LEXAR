import re
from typing import List, Dict, Optional


def _extract_year_from_text_or_name(statute_name: str, text: str) -> Optional[int]:
    # Prefer a 4-digit year in the statute name; else in text
    m = re.search(r"(18|19|20)\d{2}", statute_name)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    m = re.search(r"(18|19|20)\d{2}", text)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    return None


def _find_chapter_header(line: str):
    # Match forms like: CHAPTER I — PRELIMINARY, Chapter II, etc.
    m = re.match(r"^\s*(CHAPTER|Chapter)\s+([IVXLC]+)\b(?:\s*[\-—:]\s*(.*))?", line)
    if m:
        roman = m.group(2)
        title = (m.group(3) or '').strip()
        return roman, title
    return None


SECTION_PATTERNS = [
    # Section N. Title
    re.compile(r"^\s*Section\s+([0-9A-Z]+)\.?\s*(.*)$"),
    # N. Title (common in many Acts)
    re.compile(r"^\s*([0-9A-Z]+)\.\s+(.*)$"),
]


def _find_section_header(line: str):
    for pat in SECTION_PATTERNS:
        m = pat.match(line)
        if m:
            # First capture group is section identifier, second is title
            sec = m.group(1)
            title = (m.group(2) or '').strip()
            return sec, title
    return None


def _split_by_subsections(body: str):
    # Subsections like (1), (2), ... at start of lines or after whitespace
    # Keep markers in content; return list of (label, text)
    # If no subsections found, return a single None subsection
    # Use finditer to get spans
    matches = list(re.finditer(r"(?m)^(\([0-9]+\))\s*", body))
    if not matches:
        return [(None, body.strip())]
    result = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        label = m.group(1)[1:-1]  # '1' from '(1)'
        chunk_text = body[start:end].strip()
        result.append((label, chunk_text))
    return result


def _split_by_clauses(body: str):
    # Clauses like (a), (b), ...
    matches = list(re.finditer(r"(?m)^(\([a-z]\))\s*", body))
    if not matches:
        return [(None, body.strip())]
    result = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        label = m.group(1)[1:-1]  # 'a' from '(a)'
        chunk_text = body[start:end].strip()
        result.append((label, chunk_text))
    return result


def chunk_statute_text(text: str, statute_name: str, year: Optional[int] = None) -> List[Dict]:
    """
    Deterministically parse statute text into hierarchical chunks:
    Act → Chapter → Section → Subsection → Clause.

    Returns list of dicts with fields:
    - id, text
    - act, chapter, section, subsection, clause
    - metadata: {jurisdiction, statute_name, year, chapter_roman, section_number, section_title}
    """
    if year is None:
        year = _extract_year_from_text_or_name(statute_name, text)

    lines = text.splitlines()
    chapter = None
    chapter_title = None

    chunks: List[Dict] = []

    # We'll accumulate lines for the current section
    current_section_num: Optional[str] = None
    current_section_title: str = ''
    current_section_lines: List[str] = []

    def flush_section():
        if not current_section_num or not current_section_lines:
            return
        section_body = "\n".join(current_section_lines).strip()
        # Split into subsections, then clauses
        for subsection_label, subsection_text in _split_by_subsections(section_body):
            # Further split subsection into clauses
            clause_segments = _split_by_clauses(subsection_text)
            for clause_label, clause_text in clause_segments:
                chunk_id = f"{statute_name}-{current_section_num}"
                if subsection_label is not None:
                    chunk_id += f"-{subsection_label}"
                if clause_label is not None:
                    chunk_id += f"-{clause_label}"
                chunk: Dict = {
                    "chunk_id": chunk_id,
                    "text": clause_text,
                    "metadata": {
                        "jurisdiction": "India",
                        "statute": statute_name,
                        "year": year,
                        "chapter_roman": chapter,
                        "chapter_title": chapter_title,
                        "section": current_section_num,
                        "section_title": current_section_title,
                        "subsection": subsection_label,
                        "clause": clause_label,
                    },
                }
                chunks.append(chunk)

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            # keep blank as separator inside section text
            if current_section_num is not None:
                current_section_lines.append("")
            continue

        # Check chapter header
        ch = _find_chapter_header(line_stripped)
        if ch:
            # New chapter encountered; flush any ongoing section
            flush_section()
            current_section_num = None
            current_section_title = ''
            current_section_lines = []
            chapter, chapter_title = ch
            continue

        # Check section header
        sec = _find_section_header(line_stripped)
        if sec:
            # Flush previous section
            flush_section()
            # Start new section
            current_section_num, current_section_title = sec
            current_section_lines = []
            continue

        # Regular content line belongs to current section
        if current_section_num is not None:
            current_section_lines.append(line)
        else:
            # Content before first section; ignore or attach to a preamble section 0
            # We'll skip to keep deterministic parsing
            pass

    # Flush last section at EOF
    flush_section()

    return chunks
