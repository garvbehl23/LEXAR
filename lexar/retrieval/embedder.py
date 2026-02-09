import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Path to the fine-tuned query encoder
# embedder.py is at: backend/app/services/retrieval/embedder.py
# Need to go up 5 levels to reach project root: retrieval → services → app → backend → legalrag
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
_DEFAULT_QUERY_ENCODER_PATH = os.path.join(_PROJECT_ROOT, "data", "models", "lexar_query_encoder_v1")


class LegalEmbedder:
    """
    Embedder for legal documents and queries.
    
    Uses:
    - Base encoder (all-MiniLM-L6-v2) for chunk/text embeddings (to match FAISS index)
    - Fine-tuned query encoder (lexar_query_encoder_v1) for query embeddings
    
    This asymmetric setup allows improved retrieval without rebuilding the index.
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        query_encoder_path: str | None = None,
        use_finetuned_query_encoder: bool = True
    ):
        # Base model for chunk embeddings (must match FAISS index)
        self.model = SentenceTransformer(model_name)
        
        # Fine-tuned query encoder
        self.query_model = None
        if use_finetuned_query_encoder:
            encoder_path = query_encoder_path or _DEFAULT_QUERY_ENCODER_PATH
            if os.path.exists(encoder_path):
                self.query_model = SentenceTransformer(encoder_path)
                print(f"[LegalEmbedder] Loaded fine-tuned query encoder from {encoder_path}")
            else:
                print(f"[LegalEmbedder] Fine-tuned encoder not found at {encoder_path}, using base model")

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed texts/chunks using the BASE model (matches FAISS index)."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """Embed query using the FINE-TUNED model if available."""
        model = self.query_model if self.query_model else self.model
        embedding = model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return embedding[0]
