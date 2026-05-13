# LEXAR — Complete macOS Setup Guide

> **Fresh machine? Start here.** This guide takes you from zero to a running LEXAR instance.

---

## 0. System Requirements

- macOS 13 (Ventura) or later
- 8 GB RAM minimum (16 GB recommended)
- 3 GB free disk space

---

## 1. Install System Dependencies

### Homebrew
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/homebrew/install/HEAD/install.sh)"
```

After install, follow the instructions printed at the end to add Homebrew to your PATH (Apple Silicon Macs need an extra step).

### Python, Node, Git
```bash
brew install python@3.11 node git
```

Verify:
```bash
python3.11 --version   # Python 3.11.x
node --version         # v18+
npm --version          # 9+
git --version
```

---

## 2. Clone the Repository

```bash
git clone <YOUR_REPO_URL> LEXAR
cd LEXAR
```

---

## 3. Python Environment

```bash
# Create virtualenv
python3.11 -m venv .venv

# Activate it (do this every time you open a new terminal)
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the LEXAR package + all Python dependencies
pip install -e .
pip install -r requirements.txt
```

**Verify Python deps:**
```bash
python -c "import faiss, sentence_transformers, pdfplumber, fastapi; print('All good')"
```

---

## 4. Node / Frontend Dependencies

```bash
cd web
npm install
cd ..
```

---

## 5. Environment Variables

```bash
cp .env.example .env
```

Open `.env` in any editor and fill in:

```env
GEMINI_API_KEY=your_key_here
BACKEND_URL=http://localhost:8001
```

Get a **free** Gemini API key at → https://aistudio.google.com/app/apikey

Create the frontend env file:
```bash
cat > web/.env.local << 'EOF'
NEXT_PUBLIC_API_URL=http://localhost:8001
BACKEND_URL=http://localhost:8001
EOF
```

---

## 6. Download PDFs + Build Indices  *(one-time, ~5–10 min)*

This downloads the IPC, CrPC, IEA and other statute PDFs from the Government of India and builds the vector search indices.

```bash
# Make sure venv is active
source .venv/bin/activate

# Download all PDFs (needs internet, ~50 MB)
python scripts/prepare_data.py

# Build FAISS indices from the PDFs
python scripts/rebuild_indices.py
```

Expected output from `rebuild_indices.py`:
```
[IPC]   Chunked → 455 sections   ✓
[CRPC]  Chunked → 483 sections   ✓
[IEA]   Chunked → 165 sections   ✓
...
Saved ipc.index           (455 vectors)
Saved ipc_crpc.index      (938 vectors)
Saved lexar_medium.index  (~1400 vectors)
=== Done ===
```

---

## 7. Run the Backend

**Open Terminal A:**

```bash
cd LEXAR
source .venv/bin/activate

# Load env vars
export $(grep -v '^#' .env | xargs)

# Start backend on port 8001
uvicorn backend.app.main:app --host 0.0.0.0 --port 8001 --reload
```

Check it's working:
```bash
curl http://localhost:8001/health/
# → {"status":"ok","data_ready":true,...}
```

---

## 8. Run the Frontend

**Open Terminal B:**

```bash
cd LEXAR/web
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## 9. Optional — Ollama (Local LLM, no API key needed)

```bash
# Install
brew install ollama

# Pull a model (one-time download, ~4 GB)
ollama pull llama3

# Start the Ollama server (keep this running)
ollama serve
```

Then in the LEXAR UI, click the model pill in the input bar and switch to **Ollama**.

---

## 10. Run Tests

```bash
source .venv/bin/activate
cd LEXAR
pytest tests/ -v --tb=short
# 30 tests should pass
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'lexar'` | `pip install -e .` from project root |
| `Index not found. Run rebuild_indices.py` | `python scripts/rebuild_indices.py` |
| `GEMINI_API_KEY not set` | Check `.env`, then `export $(grep -v '^#' .env \| xargs)` |
| Backend 404 / not responding | Make sure you're on port **8001**, not 8000 |
| `npm: command not found` | `brew install node` |
| `brew: command not found` | Install Homebrew first (Step 1) |
| Ollama timeout in UI | Run `ollama serve` in a separate terminal |
| PDF download fails | Check internet; retry `python scripts/prepare_data.py` |
| `SSL: CERTIFICATE_VERIFY_FAILED` on macOS | Run `/Applications/Python\ 3.11/Install\ Certificates.command` |

---

## Quick Reference

| Action | Command |
|---|---|
| Activate Python env | `source .venv/bin/activate` |
| Start backend | `uvicorn backend.app.main:app --port 8001 --reload` |
| Start frontend | `cd web && npm run dev` |
| Download PDFs | `python scripts/prepare_data.py` |
| Build indices | `python scripts/rebuild_indices.py` |
| Run tests | `pytest tests/ -v` |
| Start Ollama | `ollama serve` |

---

## Folder Structure (what matters)

```
LEXAR/
├── backend/          ← FastAPI backend
├── web/              ← Next.js frontend (port 3000)
├── lexar/            ← Core ML pipeline
├── data/
│   ├── raw_docs/     ← Downloaded PDFs go here
│   ├── processed_docs/ ← JSON chunk files
│   └── faiss_index/  ← Vector indices
├── scripts/
│   ├── prepare_data.py     ← Download PDFs
│   └── rebuild_indices.py  ← Build FAISS indices
├── .env              ← Your API keys (never commit this)
└── RUN.md            ← This file
```
