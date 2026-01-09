import json
from pathlib import Path
from backend.app.services.chunking.generic_chunker import chunk_generic_text

IN_FILE = Path("data/raw_docs/ildc/ildc_en.jsonl")
OUT_FILE = Path("data/processed_docs/lexar_ildc_large_chunks.json")

all_chunks = []
doc_id = 0

with open(IN_FILE) as f:
    for line in f:
        obj = json.loads(line)
        chunks = chunk_generic_text(obj["text"])
        for c in chunks:
            c["metadata"]["source"] = "ILDC"
            c["metadata"]["document_id"] = doc_id
            all_chunks.append(c)
        doc_id += 1

with open(OUT_FILE, "w") as f:
    json.dump(all_chunks, f)

print(f"ILDC large corpus chunks: {len(all_chunks)}")
