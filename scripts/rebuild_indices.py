"""
Rebuild all LEXAR FAISS indices from raw statute PDFs.

Produces:
  data/processed_docs/ipc_chunks.json
  data/processed_docs/ipc_crpc_chunks.json
  data/processed_docs/ipc_crpc_iea_chunks.json
  data/processed_docs/lexar_medium_chunks.json
  data/faiss_index/ipc.index + ipc_chunk_ids.json
  data/faiss_index/ipc_crpc.index + ipc_crpc_chunk_ids.json
  data/faiss_index/ipc_crpc_iea.index + ipc_crpc_iea_chunk_ids.json
  data/faiss_index/lexar_medium.index + lexar_medium_chunk_ids.json

Usage:
    cd /home/garv/LEXAR
    python scripts/rebuild_indices.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROC_DIR  = ROOT / "data" / "processed_docs"
INDEX_DIR = ROOT / "data" / "faiss_index"
RAW_DIR   = ROOT / "data" / "raw_docs" / "statutes"

PROC_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ── PDF → chunks ──────────────────────────────────────────────────────────────

def _extract(pdf: Path) -> str:
    from lexar.ingestion.pdf_extractor import extract_text_from_pdf
    print(f"  Extracting {pdf.name} …", end=" ", flush=True)
    text = extract_text_from_pdf(pdf)
    print(f"{len(text):,} chars")
    return text


def _chunk_ipc(text: str, pdf: Path) -> list[dict]:
    from lexar.chunking.ipc_chunker import chunk_ipc_by_section
    chunks = chunk_ipc_by_section(text)
    for c in chunks:
        c["metadata"]["document"] = pdf.name
    return chunks


def _chunk_statute(text: str, statute: str, pdf: Path) -> list[dict]:
    """
    Generic section-based chunker for non-IPC statutes.
    Falls back to sliding-window if no section headers found.
    """
    chunks = _section_chunk(text, statute, pdf)
    if len(chunks) < 10:
        print(f"    [warn] only {len(chunks)} section chunks; using sliding window")
        chunks = _window_chunk(text, statute, pdf)
    return chunks


def _section_chunk(text: str, statute: str, pdf: Path) -> list[dict]:
    """Split on numeric section headers like '302. …'."""
    pat = re.compile(r"(?m)^(?P<num>\d{1,3})\.\s+[A-Z][^\n]{3,}")
    matches = list(pat.finditer(text))
    if not matches:
        return []
    raw: dict[int, str] = {}
    for i, m in enumerate(matches):
        num = int(m.group("num"))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if num not in raw or len(body) > len(raw[num]):
            raw[num] = body
    chunks = []
    for num in sorted(raw):
        body = raw[num]
        if len(body) < 50:
            continue
        chunks.append({
            "chunk_id": f"{statute}_Sec_{num}",
            "text": body,
            "metadata": {
                "statute": statute,
                "section": f"Section {num}",
                "section_number": num,
                "document": pdf.name,
                "jurisdiction": "India",
            },
        })
    return chunks


def _window_chunk(text: str, statute: str, pdf: Path,
                  max_words: int = 300, overlap: int = 50) -> list[dict]:
    words = text.split()
    stride = max(1, max_words - overlap)
    chunks, idx = [], 0
    for start in range(0, len(words), stride):
        body = " ".join(words[start: start + max_words])
        if len(body) < 50:
            continue
        chunks.append({
            "chunk_id": f"{statute}_{idx}",
            "text": body,
            "metadata": {"statute": statute, "document": pdf.name, "jurisdiction": "India"},
        })
        idx += 1
    return chunks


# ── FAISS index builder ───────────────────────────────────────────────────────

def _build_faiss(chunks: list[dict], index_stem: str) -> None:
    import faiss
    import numpy as np
    from lexar.retrieval.embedder import LegalEmbedder

    print(f"  Encoding {len(chunks)} chunks …", end=" ", flush=True)
    embedder = LegalEmbedder()
    texts = []
    for c in chunks:
        title = c.get("metadata", {}).get("section", "")
        texts.append(f"{title}\n{c['text']}" if title else c["text"])

    vecs = embedder.embed_texts(texts).astype("float32")
    print(f"dim={vecs.shape[1]}")

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    idx_path = INDEX_DIR / f"{index_stem}.index"
    ids_path = INDEX_DIR / f"{index_stem}_chunk_ids.json"

    faiss.write_index(index, str(idx_path))
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump([c.get("chunk_id") for c in chunks], f, ensure_ascii=False)

    print(f"  Saved {index_stem}.index ({index.ntotal} vectors)")


def _save_chunks(chunks: list[dict], name: str) -> None:
    path = PROC_DIR / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  Saved {name} ({len(chunks)} chunks)")


# ── Main ─────────────────────────────────────────────────────────────────────

STATUTE_FILES = {
    "IPC":  "Indian_Penal_Code_1860.pdf",
    "CRPC": "Code_of_Criminal_Procedure_1973.pdf",
    "IEA":  "Indian_Evidence_Act_1872.pdf",
    "DPA":  "Dowry_Prohibition_Act_1961.pdf",
    "NDPS": "NDPS_Act_1985.pdf",
    "MVA":  "Motor_Vehicles_Act_1988.pdf",
}


def main() -> None:
    print("=== LEXAR Index Rebuild ===\n")

    all_chunks: dict[str, list[dict]] = {}

    for statute, filename in STATUTE_FILES.items():
        pdf = RAW_DIR / filename
        if not pdf.exists():
            print(f"[SKIP] {filename} not found at {pdf}")
            continue
        print(f"\n[{statute}]")
        text = _extract(pdf)
        if statute == "IPC":
            chunks = _chunk_ipc(text, pdf)
        else:
            chunks = _chunk_statute(text, statute, pdf)
        print(f"  Chunked → {len(chunks)} sections")
        all_chunks[statute] = chunks

    if "IPC" not in all_chunks:
        print("\nFATAL: IPC PDF not found. Cannot build indices.")
        sys.exit(1)

    # ── Per-statute and combined chunk files ──────────────────────────────────

    print("\n=== Saving chunk files ===")

    ipc = all_chunks["IPC"]
    _save_chunks(ipc, "ipc_chunks.json")

    ipc_crpc = ipc + all_chunks.get("CRPC", [])
    _save_chunks(ipc_crpc, "ipc_crpc_chunks.json")

    ipc_crpc_iea = ipc_crpc + all_chunks.get("IEA", [])
    _save_chunks(ipc_crpc_iea, "ipc_crpc_iea_chunks.json")

    medium = []
    for statute in ["IPC", "CRPC", "IEA", "DPA", "NDPS", "MVA"]:
        medium.extend(all_chunks.get(statute, []))
    _save_chunks(medium, "lexar_medium_chunks.json")

    # ── FAISS indices ─────────────────────────────────────────────────────────

    print("\n=== Building FAISS indices ===")

    print("\n[ipc]")
    _build_faiss(ipc, "ipc")

    print("\n[ipc_crpc]")
    _build_faiss(ipc_crpc, "ipc_crpc")

    print("\n[ipc_crpc_iea]")
    _build_faiss(ipc_crpc_iea, "ipc_crpc_iea")

    print("\n[lexar_medium]")
    _build_faiss(medium, "lexar_medium")

    print("\n=== Done ===")
    print(f"IPC:         {len(ipc)} chunks")
    print(f"IPC+CrPC:    {len(ipc_crpc)} chunks")
    print(f"IPC+CrPC+IEA:{len(ipc_crpc_iea)} chunks")
    print(f"Medium:      {len(medium)} chunks")


if __name__ == "__main__":
    main()
