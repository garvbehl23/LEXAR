from pathlib import Path
import re


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract and clean text from a PDF using pdfplumber.
    Normalizes whitespace, removes isolated page numbers.
    """
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages.append(_clean_page(raw))

    full = "\n".join(pages)
    # Collapse 3+ blank lines to two
    full = re.sub(r"\n{3,}", "\n\n", full)
    return full


def _clean_page(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        # Drop lines that are only a page number
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        # Drop short separator lines
        if re.fullmatch(r"[-_=]{3,}", stripped):
            continue
        lines.append(line)

    # Merge lines broken mid-word: ends with '-', next starts lowercase
    merged: list[str] = []
    for line in lines:
        if merged and merged[-1].rstrip().endswith("-") and line and line[0].islower():
            merged[-1] = merged[-1].rstrip()[:-1] + line.lstrip()
        else:
            merged.append(line)
    return "\n".join(merged)
