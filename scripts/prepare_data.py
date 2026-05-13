import requests
from pathlib import Path
import json
import time

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data/raw_docs/statutes"
PROC_DIR = ROOT / "data/processed_docs"
INDEX_DIR = ROOT / "data/faiss_index"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# 🔥 YOUR EXACT LINKS (UNCHANGED)
PDF_SOURCES = {
    "ipc": ("Indian_Penal_Code_1860.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/15289/1/ipc_act.pdf"),

    "crpc": ("Code_of_Criminal_Procedure_1973.pdf",
             "https://www.indiacode.nic.in/bitstream/123456789/15272/1/the_code_of_criminal_procedure,_1973.pdf"),

    "iea": ("Indian_Evidence_Act_1872.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/15351/1/iea_1872.pdf"),

    "dpa": ("Dowry_Prohibition_Act_1961.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/2435/1/a1961-43.pdf"),

    "ndps": ("NDPS_Act_1985.pdf",
             "https://www.indiacode.nic.in/bitstream/123456789/18974/1/narcotic-drugs-and-psychotropic-substances-act-1985.pdf"),

    "mva": ("Motor_Vehicles_Act_1988.pdf",
            "https://www.indiacode.nic.in/bitstream/123456789/9460/1/a1988-59.pdf"),
}


HEADERS = {
    "User-Agent": "Mozilla/5.0",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def download_pdf(name, url):
    path = RAW_DIR / name

    if path.exists():
        log(f"✅ Already exists: {name}")
        return path

    log(f"⬇ Downloading: {name}")

    try:
        r = requests.get(url, headers=HEADERS, timeout=30)

        if r.status_code == 200:
            with open(path, "wb") as f:
                f.write(r.content)

            log(f"✅ Downloaded: {name} ({len(r.content)//1024} KB)")
            return path

        else:
            log(f"❌ Failed ({r.status_code}) → {name}")

    except Exception as e:
        log(f"❌ Error downloading {name}: {e}")

    log(f"⚠️ Manual download required: {name}")
    return None


def extract_text(pdf_path):
    import fitz

    log(f"📄 Extracting text: {pdf_path.name}")

    doc = fitz.open(pdf_path)
    text = ""

    for i, page in enumerate(doc):
        text += page.get_text()

    log(f"✅ Extracted {len(text)} characters")
    return text


def chunk_text(text, statute):
    log(f"✂️ Chunking: {statute}")

    if statute == "ipc":
        from lexar.chunking.ipc_chunker import chunk_ipc_by_section
        raw = chunk_ipc_by_section(text)
        # Normalise to the flat format this script uses (meta key)
        chunks = [
            {"text": c["text"], "meta": c.get("metadata", c.get("meta", {"statute": "IPC"}))}
            for c in raw if len(c.get("text", "")) >= 50
        ]
    else:
        import re
        size = 800
        chunks = []
        for i in range(0, len(text), size):
            piece = text[i:i+size].strip()
            if len(piece) >= 50:
                chunks.append({"text": piece, "meta": {"statute": statute.upper()}})

    log(f"✅ {len(chunks)} chunks created")
    return chunks


def build_index(chunks):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    log("🧠 Building embeddings...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in chunks]
    vectors = model.encode(texts, show_progress_bar=True)

    vectors = np.array(vectors).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    out_path = INDEX_DIR / "lexar.index"
    faiss.write_index(index, str(out_path))

    log(f"✅ FAISS index saved → {out_path}")


def main():
    log("🚀 STARTING LEXAR DATA PIPELINE")

    all_chunks = []

    for key, (name, url) in PDF_SOURCES.items():
        log(f"\n=== Processing {key.upper()} ===")

        pdf_path = download_pdf(name, url)

        if not pdf_path:
            continue

        text = extract_text(pdf_path)
        chunks = chunk_text(text, key)

        all_chunks.extend(chunks)

    if not all_chunks:
        log("❌ No data processed. Exiting.")
        return

    out_file = PROC_DIR / "lexar_chunks.json"

    with open(out_file, "w") as f:
        json.dump(all_chunks, f)

    log(f"\n✅ Saved all chunks → {out_file}")
    log(f"📊 Total chunks: {len(all_chunks)}")

    build_index(all_chunks)

    log("\n🎉 DATA PIPELINE COMPLETE")
    log("👉 Restart backend now")


if __name__ == "__main__":
    main()