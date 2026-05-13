"""
Generate stub chunk JSON files from committed chunk_id files.

This allows the retriever to load and the app to start even without
the source PDFs. The stub text is empty/placeholder — run prepare_data.py
with real PDFs for production-quality retrieval.

Usage:
    python scripts/generate_stub_chunks.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INDEX_DIR = ROOT / "data" / "faiss_index"
PROCESSED_DIR = ROOT / "data" / "processed_docs"

STUB_CONFIGS = [
    {
        "ids_file": "ipc_chunk_ids.json",
        "chunks_out": "ipc_chunks.json",
        "statute": "IPC",
    },
    {
        "ids_file": "ipc_crpc_chunk_ids.json",
        "chunks_out": "ipc_crpc_chunks.json",
        "statute": "IPC+CrPC",
    },
    {
        "ids_file": "ipc_crpc_iea_chunk_ids.json",
        "chunks_out": "ipc_crpc_iea_chunks.json",
        "statute": "IPC+CrPC+IEA",
    },
]


def generate_stubs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    for cfg in STUB_CONFIGS:
        ids_path = INDEX_DIR / cfg["ids_file"]
        if not ids_path.exists():
            print(f"[SKIP] {ids_path} not found")
            continue

        out_path = PROCESSED_DIR / cfg["chunks_out"]
        if out_path.exists():
            print(f"[EXISTS] {out_path} — skipping")
            continue

        with open(ids_path) as f:
            chunk_ids = json.load(f)

        chunks = []
        for cid in chunk_ids:
            section = cid.split("-")[-1] if "-" in cid else cid
            chunks.append({
                "chunk_id": cid,
                "text": f"[Stub] {cfg['statute']} provision {cid}. "
                        f"Run 'make data-prep' to populate with real legal text.",
                "metadata": {
                    "statute": cfg["statute"],
                    "section": section,
                    "jurisdiction": "India",
                    "stub": True,
                },
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Generated {len(chunks)} stub chunks → {out_path}")

    # Build lexar_medium stub as union
    all_chunks = []
    for cfg in STUB_CONFIGS[:1]:  # Use just IPC for medium stub
        out_path = PROCESSED_DIR / cfg["chunks_out"]
        if out_path.exists():
            with open(out_path) as f:
                all_chunks.extend(json.load(f))

    medium_out = PROCESSED_DIR / "lexar_medium_chunks.json"
    if not medium_out.exists() and all_chunks:
        with open(medium_out, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        print(f"Generated lexar_medium stubs ({len(all_chunks)} chunks) → {medium_out}")

    print("\nStub chunks generated. Retrieval will work but answer quality requires real data.")
    print("Run: python scripts/prepare_data.py --statutes ipc crpc")


if __name__ == "__main__":
    generate_stubs()
