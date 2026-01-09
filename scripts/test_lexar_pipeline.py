from backend.app.services.retrieval.ipc_retriever import IPCRetriever
from backend.app.services.lexar_pipeline import LexarPipeline

ipc = IPCRetriever(
    "data/processed_docs/lexar_medium_chunks.json",
    "data/faiss_index/lexar_medium.index"
)

pipeline = LexarPipeline(ipc=ipc)

query = "What is the punishment for murder under IPC?"

answer = pipeline.answer(query)

print("\n--- LEXAR ANSWER ---\n")
print(answer)
