FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies for PDF processing and ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY lexar/ ./lexar/
COPY backend/ ./backend/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --no-deps

# Copy data (indices + model — excluded from build if large via .dockerignore)
COPY data/ ./data/

# ── Backend image ──────────────────────────────────────────────────────────
FROM base AS backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/')"
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ── Frontend image ─────────────────────────────────────────────────────────
FROM base AS frontend
COPY frontend/ ./frontend/
RUN pip install --no-cache-dir streamlit>=1.32.0 plotly>=5.18.0
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -sf http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "frontend/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
