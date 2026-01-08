import pdfplumber
import os
from backend.app.services.chunking.ipc_chunker import chunk_ipc_by_section

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    full_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    return "\n".join(full_text)


def chunk_text(text: str):
    return chunk_ipc_by_section(text)
