from __future__ import annotations

import logging
import os
import uuid
import json

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from lexar.ingestion.pdf_extractor import extract_text_from_pdf
from lexar.chunking.ipc_chunker import chunk_ipc_by_section
from lexar.chunking.generic_chunker import chunk_generic_text
from lexar.utils.text_cleaner import clean_text

logger = logging.getLogger("lexar.backend.upload")

UPLOAD_DIR = "data/raw_docs"
PROCESSED_DIR = "data/processed_docs"
MAX_FILE_SIZE_MB = 10

router = APIRouter()


class UploadResponse(BaseModel):
    document_id: str
    original_filename: str
    size_mb: float
    text_length: int
    num_chunks: int
    chunks_path: str
    status: str


@router.post("/", response_model=UploadResponse, summary="Upload and ingest a PDF")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, extract text, clean it, and chunk it.
    Returns the document_id and chunk metadata.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_FILE_SIZE_MB} MB)")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    document_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    with open(pdf_path, "wb") as f:
        f.write(contents)

    try:
        from pathlib import Path
        raw_text = extract_text_from_pdf(Path(pdf_path))
    except Exception as exc:
        logger.exception("PDF extraction failed for %s", file.filename)
        os.remove(pdf_path)
        raise HTTPException(status_code=422, detail=f"PDF text extraction failed: {exc}") from exc

    clean_content = clean_text(raw_text)

    text_path = os.path.join(PROCESSED_DIR, f"{document_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(clean_content)

    # Try IPC-style chunking first; fall back to generic sliding window
    chunks = chunk_ipc_by_section(clean_content)
    if not chunks:
        chunks = chunk_generic_text(clean_content)

    # Tag user source
    for chunk in chunks:
        chunk.setdefault("metadata", {}).update({
            "source": "UserUpload",
            "document_id": document_id,
            "original_filename": file.filename,
        })

    chunks_path = os.path.join(PROCESSED_DIR, f"{document_id}_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    logger.info("Ingested %s → %d chunks (doc_id=%s)", file.filename, len(chunks), document_id)

    return UploadResponse(
        document_id=document_id,
        original_filename=file.filename,
        size_mb=round(size_mb, 3),
        text_length=len(clean_content),
        num_chunks=len(chunks),
        chunks_path=chunks_path,
        status="ingested",
    )
