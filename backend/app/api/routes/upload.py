import os
import uuid
import json
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.services.ingestion import (
    extract_text_from_pdf,
    chunk_text
)
from backend.app.utils.text_cleaner import clean_text

UPLOAD_DIR = "data/raw_docs"
PROCESSED_DIR = "data/processed_docs"
MAX_FILE_SIZE_MB = 10

router = APIRouter()


@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    # ---------- Validation ----------
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed"
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail="File too large (max 10MB)"
        )

    # ---------- Directory Setup ----------
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # ---------- Save PDF ----------
    document_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    with open(pdf_path, "wb") as f:
        f.write(contents)

    # ---------- Ingestion ----------
    raw_text = extract_text_from_pdf(pdf_path)
    clean_content = clean_text(raw_text)

    text_path = os.path.join(PROCESSED_DIR, f"{document_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(clean_content)

    # ---------- Legal-aware Chunking ----------
    chunks = chunk_text(clean_content)

    chunks_path = os.path.join(
        PROCESSED_DIR,
        f"{document_id}_chunks.json"
    )

    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # ---------- Response ----------
    return {
        "document_id": document_id,
        "original_filename": file.filename,
        "size_mb": round(size_mb, 2),
        "text_length": len(clean_content),
        "num_chunks": len(chunks),
        "status": "ingested_and_chunked"
    }
