"""
LEXAR Data Preparation Script

Downloads Indian legal statutes, processes them into chunks, and builds
FAISS indices. Run this once before starting the backend or frontend.

Usage:
    python scripts/prepare_data.py              # full pipeline
    python scripts/prepare_data.py --statutes ipc          # IPC only
    python scripts/prepare_data.py --statutes ipc crpc     # IPC + CrPC
    python scripts/prepare_data.py --skip-download         # assume PDFs exist
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


STATUTES_DIR = ROOT / "data" / "raw_docs" / "statutes"
PROCESSED_DIR = ROOT / "data" / "processed_docs"
INDEX_DIR = ROOT / "data" / "faiss_index"


STATUTE_SOURCES = {
    "ipc": {
        "url": "https://www.indiacode.nic.in/bitstream/123456789/2263/1/A1860-45.pdf",
        "filename": "Indian_Penal_Code_1860.pdf",
        "chunks_out": "ipc_chunks.json",
        "index_out": "ipc.index",
        "ids_out": "ipc_chunk_ids.json",
    },
    "crpc": {
        "url": "https://www.indiacode.nic.in/bitstream/123456789/1611/1/973208.pdf",
        "filename": "Code_of_Criminal_Procedure_1973.pdf",
        "chunks_out": "crpc_chunks.json",
        "index_out": None,
        "ids_out": None,
    },
    "iea": {
        "url": "https://www.indiacode.nic.in/bitstream/123456789/2263/1/A1872-01.pdf",
        "filename": "Indian_Evidence_Act_1872.pdf",
        "chunks_out": "iea_1872_chunks.json",
        "index_out": None,
        "ids_out": None,
    },
}


def _download(url: str, dest: Path) -> None:
    import urllib.request
    print(f"  Downloading {url} → {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved {dest.stat().st_size // 1024} KB")
    except Exception as exc:
        print(f"  [WARN] Download failed ({exc}). Place the PDF manually at {dest}")


def _ingest_pdf(pdf_path: Path):
    from lexar.ingestion.pdf_extractor import extract_text_from_pdf
    from lexar.chunking.ipc_chunker import chunk_ipc_by_section
    from lexar.chunking.statute_chunker import chunk_statute_text
    from lexar.utils.text_cleaner import clean_text

    name = pdf_path.stem.lower()
    text = extract_text_from_pdf(pdf_path)
    text = clean_text(text)

    if "ipc" in name or "penal" in name:
        chunks = chunk_ipc_by_section(text)
        statute = "IPC"
    elif "crpc" in name or "criminal procedure" in name:
        chunks = chunk_statute_text(text, statute_name="CrPC", year=1973)
        statute = "CrPC"
    elif "evidence" in name:
        chunks = chunk_statute_text(text, statute_name="IEA", year=1872)
        statute = "IEA"
    else:
        chunks = chunk_statute_text(text)
        statute = pdf_path.stem

    for chunk in chunks:
        chunk.setdefault("metadata", {}).update({"statute": statute, "source": pdf_path.name})

    return chunks


def _build_faiss_index(chunks: list, index_path: Path, ids_path: Path) -> None:
    import faiss
    import numpy as np
    from lexar.retrieval.embedder import LegalEmbedder

    embedder = LegalEmbedder()
    texts = [c.get("text", "") for c in chunks]
    print(f"  Encoding {len(texts)} chunks ...")
    vectors = embedder.embed_texts(texts).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    faiss.write_index(index, str(index_path))
    print(f"  Index saved: {index_path} ({index.ntotal} vectors, dim={dim})")

    chunk_ids = [c.get("chunk_id") for c in chunks]
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(chunk_ids, f, ensure_ascii=False, indent=2)
    print(f"  Chunk IDs saved: {ids_path}")


def _build_combined_index(statutes: list[str]) -> None:
    all_chunks = []
    for s in statutes:
        chunks_path = PROCESSED_DIR / STATUTE_SOURCES[s]["chunks_out"]
        if not chunks_path.exists():
            print(f"  [SKIP] {chunks_path} not found — run ingestion first")
            continue
        with open(chunks_path, "r", encoding="utf-8") as f:
            all_chunks.extend(json.load(f))

    if not all_chunks:
        print("  No chunks found for combined index. Skipping.")
        return

    suffix = "_".join(statutes)
    index_path = INDEX_DIR / f"{'_'.join(s for s in statutes if STATUTE_SOURCES[s]['index_out'])}.index"
    ids_path = INDEX_DIR / f"{'_'.join(s for s in statutes)}_chunk_ids.json"
    _build_faiss_index(all_chunks, index_path, ids_path)


def _build_lexar_medium(statutes: list[str]) -> None:
    """Build the combined lexar_medium index from all available chunks."""
    all_chunks = []
    for s in statutes:
        chunks_path = PROCESSED_DIR / STATUTE_SOURCES[s]["chunks_out"]
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))

    if not all_chunks:
        return

    out_chunks = PROCESSED_DIR / "lexar_medium_chunks.json"
    with open(out_chunks, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"  lexar_medium chunks: {out_chunks} ({len(all_chunks)} chunks)")

    _build_faiss_index(all_chunks, INDEX_DIR / "lexar_medium.index", INDEX_DIR / "lexar_medium_chunk_ids.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LEXAR data: ingest → chunk → index")
    parser.add_argument("--statutes", nargs="*", default=list(STATUTE_SOURCES), choices=list(STATUTE_SOURCES),
                        help="Statutes to prepare (default: all)")
    parser.add_argument("--skip-download", action="store_true", help="Skip downloading PDFs")
    parser.add_argument("--skip-index", action="store_true", help="Skip building FAISS indices")
    args = parser.parse_args()

    STATUTES_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for statute_key in args.statutes:
        cfg = STATUTE_SOURCES[statute_key]
        pdf_path = STATUTES_DIR / cfg["filename"]

        print(f"\n=== {statute_key.upper()} ===")

        if not args.skip_download and not pdf_path.exists():
            _download(cfg["url"], pdf_path)

        if not pdf_path.exists():
            print(f"  [SKIP] PDF not found: {pdf_path}")
            continue

        chunks_out = PROCESSED_DIR / cfg["chunks_out"]
        if chunks_out.exists():
            print(f"  [CACHED] {chunks_out}")
            with open(chunks_out, "r") as f:
                chunks = json.load(f)
        else:
            print(f"  Ingesting {pdf_path.name} ...")
            chunks = _ingest_pdf(pdf_path)
            with open(chunks_out, "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            print(f"  Chunks saved: {chunks_out} ({len(chunks)} chunks)")

        if not args.skip_index and cfg.get("index_out"):
            index_path = INDEX_DIR / cfg["index_out"]
            ids_path = INDEX_DIR / cfg["ids_out"]
            if index_path.exists():
                print(f"  [CACHED] FAISS index: {index_path}")
            else:
                _build_faiss_index(chunks, index_path, ids_path)

    # Build combined indices
    if not args.skip_index:
        available = [s for s in args.statutes if (PROCESSED_DIR / STATUTE_SOURCES[s]["chunks_out"]).exists()]
        if len(available) >= 2:
            print("\n=== Building combined lexar_medium index ===")
            _build_lexar_medium(available)

    print("\n✓ Data preparation complete.")
    print("You can now start the backend: uvicorn backend.app.main:app --reload")


if __name__ == "__main__":
    main()
