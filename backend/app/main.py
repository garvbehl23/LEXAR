from fastapi import FastAPI
from backend.app.api.routes import health, upload

app = FastAPI(
    title="LegalRAG",
    description="Production-grade Legal Document RAG System",
    version="1.0.0"
)

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
