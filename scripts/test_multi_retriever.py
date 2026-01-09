from backend.app.services.retrieval.ipc_retriever import IPCRetriever
from backend.app.services.retrieval.multi_index_retriever import MultiIndexRetriever

ipc = IPCRetriever(
    "data/processed_docs/lexar_medium_chunks.json",
    "data/faiss_index/lexar_medium.index"
)

multi = MultiIndexRetriever(ipc=ipc)

results = multi.retrieve("What is the punishment for murder?")

print("Results:", len(results))
if results:
    print(results[0]["metadata"])
else:
    print("No results returned")
