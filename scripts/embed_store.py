import os, json
from sentence_transformers import SentenceTransformer
import chromadb

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    chunks_file = os.path.join(data_dir, "chunks.json")

    if not os.path.exists(chunks_file):
        raise FileNotFoundError(f"{chunks_file} not found. Run chunk_text.py first!")
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "rag", "chroma_db"))
    collection = client.get_or_create_collection("legal_docs")
    for chunk in chunks:
        emb = model.encode(chunk["text"]).tolist()
        collection.add(
            embeddings=[emb],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "page": chunk["page"]}],
            ids=[f"{chunk['source']}_{chunk['page']}"]
        )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {os.path.join(BASE_DIR, 'rag', 'chroma_db')}")
