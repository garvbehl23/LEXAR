from abc import ABC, abstractmethod
from typing import List, Dict

class BaseIngestor(ABC):
    @abstractmethod
    def ingest(self, source) -> List[Dict]:
        """
        Takes a document source (PDF path or text)
        Returns a list of chunks with metadata
        """
        pass
