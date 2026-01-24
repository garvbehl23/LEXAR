from backend.app.services.retrieval.ipc_retriever import IPCRetriever
from backend.app.services.retrieval.multi_index_retriever import MultiIndexRetriever

ipc = IPCRetriever(
    "data/processed_docs/ipc_chunks.json",
    "data/faiss_index/ipc.index"
)

multi = MultiIndexRetriever(ipc=ipc)

results = multi.retrieve("What is the punishment for murder?")
print(len(results))
print(results[0]["metadata"])
