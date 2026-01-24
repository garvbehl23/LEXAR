import argparse
import json
import os
import re
from typing import List, Dict, Optional

# Adjust path for backend imports when running as a script
import sys
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.app.services.ingestion import pdf_extractor  # type: ignore
from backend.app.services.chunking import ipc_chunker  # type: ignore
from backend.app.services.chunking.statute_chunker import chunk_statute_text  # type: ignore


def _statute_from_filename(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace('_', ' ').strip()
    return name


def _year_from_filename(filename: str) -> Optional[int]:
    m = re.search(r"(18|19|20)\d{2}", filename)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            return None
    return None


def ingest_pdf(pdf_path: str) -> List[Dict]:
    text = pdf_extractor.extract_text_from_pdf(pdf_path)
    statute_name = _statute_from_filename(pdf_path)
    year = _year_from_filename(pdf_path)

    # Prefer IPC specialized chunker when applicable
    if 'Indian Penal Code' in statute_name or re.search(r"\bIPC\b", statute_name):
        return ipc_chunker.chunk_ipc_by_section(text)

    # Otherwise use generic statute chunker
    return chunk_statute_text(text, statute_name=statute_name, year=year)


def save_chunks(chunks: List[Dict], out_dir: str, statute_name: str):
    os.makedirs(out_dir, exist_ok=True)
    safe_name = re.sub(r"\s+", "_", statute_name)
    out_path = os.path.join(out_dir, f"{safe_name}_chunks.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return out_path


STATUTE_SYNONYMS = {
    "ipc": ["ipc", "indian penal code"],
    "crpc": ["crpc", "code of criminal procedure"],
    "cpc": ["cpc", "code of civil procedure"],
    "evidence": ["evidence", "indian evidence act"],
}


def _matches_statute(filename: str, tokens: List[str]) -> bool:
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    for t in tokens:
        tnorm = t.lower()
        synonyms = STATUTE_SYNONYMS.get(tnorm, [tnorm])
        for s in synonyms:
            if s in name:
                return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Ingest Indian statute PDFs into hierarchical chunks")
    parser.add_argument('--input-dir', default=os.path.join(ROOT, 'data', 'raw_docs', 'statutes'), help='Directory containing statute PDFs')
    parser.add_argument('--output-dir', default=os.path.join(ROOT, 'data', 'processed_docs'), help='Directory to write chunk JSON files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of PDFs to process')
    parser.add_argument('--statutes', nargs='+', help='Filter by statute tokens, e.g., ipc crpc cpc evidence')
    args = parser.parse_args()

    pdfs = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith('.pdf')]
    pdfs.sort()
    if args.statutes:
        pdfs = [p for p in pdfs if _matches_statute(p, args.statutes)]
    if args.limit:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        available = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.pdf')]
        print(f"No matching PDFs found in {args.input_dir}. Available: {available}")
        return

    for pdf in pdfs:
        print(f"Processing {pdf} ...")
        try:
            chunks = ingest_pdf(pdf)
            statute_name = _statute_from_filename(pdf)
            out_path = save_chunks(chunks, args.output_dir, statute_name)
            print(f"Wrote {len(chunks)} chunks to {out_path}")
        except Exception as e:
            print(f"Failed to process {pdf}: {e}")


if __name__ == '__main__':
    main()
