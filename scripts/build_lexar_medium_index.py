import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = "data/processed_docs/lexar_medium_chunks.json"
INDEX_PATH = "data/faiss_index/lexar_medium.index"

with open(CHUNKS_PATH) as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

print(f"Built lexar_medium.index with {len(texts)} chunks")
