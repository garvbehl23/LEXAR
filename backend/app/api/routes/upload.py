import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.app.services.ingestion import extract_text_from_pdf
from backend.app.utils.text_cleaner import clean_text

UPLOAD_DIR = "data/raw_docs"
PROCESSED_DIR = "data/processed_docs"
MAX_FILE_SIZE_MB = 10

router = APIRouter()

@router.post("/")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    doc_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")

    with open(pdf_path, "wb") as f:
        f.write(contents)

    # 🔹 INGESTION STEP
    raw_text = extract_text_from_pdf(pdf_path)
    clean_content = clean_text(raw_text)

    text_path = os.path.join(PROCESSED_DIR, f"{doc_id}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(clean_content)

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "text_length": len(clean_content),
        "status": "ingested"
    }
