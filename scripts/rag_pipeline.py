import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# -------------------------------
# Consistent BASE_DIR
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "rag", "chroma_db")

# -------------------------------
# Load embedder (same as embed_store.py)
# -------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_chunks(query, top_k=5):
    print("[INFO] Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("legal_docs")

    print(f"[INFO] Retrieving top {top_k} relevant chunks for query...")
    query_emb = embedder.encode(query).tolist()

    results = collection.query(query_embeddings=[query_emb], n_results=top_k)
    print(f"[INFO] Retrieved {len(results['documents'][0])} chunks.\n")
    return results

def generate_answer(query, retrieved):
    print("[INFO] Loading T5 model (this may take time on first run)...")
    generator = pipeline("text2text-generation", model="google/flan-t5-small", device=-1)

    # Combine retrieved chunks
    chunks = [doc for doc_list in retrieved["documents"] for doc in doc_list]
    context = "\n".join(chunks)

    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    print("[INFO] Generating final answer...\n")

    output = generator(prompt, max_new_tokens=200, do_sample=False)[0]["generated_text"]
    return output

if __name__ == "__main__":
    query = input("Enter your legal query: ")
    retrieved = retrieve_chunks(query)

    if retrieved["documents"] and retrieved["documents"][0]:
        answer = generate_answer(query, retrieved)
        print("\n--- Final Answer ---")
        print(answer)
    else:
        print("\n⚠️ No relevant chunks found in DB. Try re-running embed_store.py.")
