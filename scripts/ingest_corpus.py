import json
import uuid
from pathlib import Path

from backend.app.services.ingestion import extract_text_from_pdf
from backend.app.services.chunking.ipc_chunker import chunk_ipc_text
from backend.app.services.chunking.judgment_chunker import chunk_judgment_text

RAW_IPC = Path("data/raw_docs/ipc/ipc.pdf")
RAW_JUDGMENTS = Path("data/raw_docs/judgments")
OUT_DIR = Path("data/processed_docs")
OUT_DIR.mkdir(exist_ok=True)

all_chunks = []

# --- IPC ---
ipc_text = extract_text_from_pdf(RAW_IPC)
ipc_chunks = chunk_ipc_text(ipc_text)

for c in ipc_chunks:
    c["metadata"]["source"] = "IPC"
    all_chunks.append(c)

# --- Judgments ---
for pdf in RAW_JUDGMENTS.glob("*.pdf"):
    text = extract_text_from_pdf(pdf)
    judgment_chunks = chunk_judgment_text(text)

    for c in judgment_chunks:
        c["metadata"]["source"] = "Judgment"
        c["metadata"]["document"] = pdf.name
        all_chunks.append(c)

out_file = OUT_DIR / "lexar_medium_chunks.json"
with open(out_file, "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Ingested {len(all_chunks)} chunks → {out_file}")
