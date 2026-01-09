from .base import BaseIngestor
from .pdf_extractor import extract_text_from_pdf
from backend.app.services.chunking.generic_chunker import chunk_generic_text

class UserPDFIngestor(BaseIngestor):
    def ingest(self, pdf_path):
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_generic_text(text)

        for c in chunks:
            c["metadata"].update({
                "source": "UserPDF",
                "visibility": "private"
            })

        return chunks
