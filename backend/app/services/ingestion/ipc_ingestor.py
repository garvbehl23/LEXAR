from pathlib import Path
from .base import BaseIngestor
from .pdf_extractor import extract_text_from_pdf
from backend.app.services.chunking.ipc_chunker import chunk_ipc_by_section

class IPCIngestor(BaseIngestor):
    def ingest(self, pdf_path: Path):
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_ipc_by_section(text)

        for c in chunks:
            c["metadata"].update({
                "source": "IPC",
                "document": pdf_path.name
            })

        return chunks
