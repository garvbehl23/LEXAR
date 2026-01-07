import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException

UPLOAD_DIR = "data/raw_docs"
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
    doc_id = str(uuid.uuid4())
    save_path = os.path.join(UPLOAD_DIR, f"{doc_id}.pdf")
    with open(save_path, "wb") as f:
        f.write(contents)

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "size_mb": round(size_mb, 2),
        "status": "uploaded"
    }
