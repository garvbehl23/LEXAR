import os
import chromadb
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "rag", "chroma_db")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("legal_docs")

    print("🔎 Interactive Retriever Ready! (type 'exit' to quit)")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print(" Exiting retriever.")
            break
        query_emb = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3
        )

        print("\n=== Top Matches ===")
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            print(f"\nResult {i+1}:")
            print(f"Text   : {doc[:300]}{'...' if len(doc) > 300 else ''}")
            print(f"Source : {meta['source']} (page {meta['page']})")
