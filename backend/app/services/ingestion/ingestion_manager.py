from pathlib import Path
from .ipc_ingestor import IPCIngestor
from .judgment_ingestor import JudgmentIngestor
from .user_ingestor import UserPDFIngestor

class IngestionManager:
    def __init__(self):
        self.ipc = IPCIngestor()
        self.judgment = JudgmentIngestor()
        self.user = UserPDFIngestor()

    def ingest_ipc(self, pdf_path: Path):
        return self.ipc.ingest(pdf_path)

    def ingest_judgment(self, pdf_path: Path):
        return self.judgment.ingest(pdf_path)

    def ingest_user_pdf(self, pdf_path: Path):
        return self.user.ingest(pdf_path)
