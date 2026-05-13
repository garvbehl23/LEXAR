# LEXAR — Run Guide

Full-stack Indian Legal AI.
Backend: FastAPI (Python) · Frontend: Next.js 14

---

## Quick Start (all platforms)

### 1. Backend

```bash
# From the LEXAR root directory
cd /path/to/LEXAR

# Activate the virtual environment
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\activate             # Windows (PowerShell)

# Start FastAPI
uvicorn backend.app.main:app --reload --port 8000
```

Backend runs at: http://localhost:8000
API docs at: http://localhost:8000/docs

### 2. Frontend

```bash
cd web
npm install          # first time only
npm run dev
```

Frontend runs at: **http://localhost:3000**

---

## Platform-Specific Setup

---

### Windows

**Node.js**
Download from https://nodejs.org (LTS version)

**Python (if not installed)**
Download from https://python.org (3.11+)

**Virtual environment**
```powershell
cd LEXAR
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Backend**
```powershell
uvicorn backend.app.main:app --reload --port 8000
```

**Frontend**
```powershell
cd web
npm install
npm run dev
```

**Ollama (optional — for local LLM)**
Download: https://ollama.com/download/windows
```powershell
ollama run llama3
```

---

### Linux (Ubuntu / Debian)

**Node.js**
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

**Python venv + deps**
```bash
cd LEXAR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Backend**
```bash
uvicorn backend.app.main:app --reload --port 8000
```

**Frontend**
```bash
cd web
npm install
npm run dev
```

**Ollama (optional)**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama run llama3
```

---

### macOS

**Node.js (via Homebrew)**
```bash
brew install node
```

**Python venv + deps**
```bash
cd LEXAR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Backend**
```bash
uvicorn backend.app.main:app --reload --port 8000
```

**Frontend**
```bash
cd web
npm install
npm run dev
```

**Ollama (optional)**
```bash
brew install ollama
ollama run llama3
```

---

## Environment Variables

### `LEXAR/.env` (backend)
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.0-flash
DEFAULT_INDEX=ipc
CORS_ORIGINS=["http://localhost:3000","http://localhost:8501"]
```

Get a free Gemini key at: https://aistudio.google.com/app/apikey

### `web/.env.local` (frontend)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
BACKEND_URL=http://localhost:8000
```

---

## Model Selector

The model selector lives **inside the input bar** (bottom of chat).

| Model | Requires | Notes |
|-------|----------|-------|
| Gemini | `GEMINI_API_KEY` in `.env` | Fast, high quality |
| Ollama | Ollama running locally | Private, no API key |
| Flan-T5 | Nothing | Slow fallback, offline |

Switch models by clicking the pill next to the input field.

---

## Data Preparation

Before querying, ensure FAISS indices are built:

```bash
# From LEXAR root, with venv active:
python scripts/build_index.py          # builds IPC index
python scripts/build_all_indices.py    # builds all indices
```

Indices are stored in `data/faiss_index/`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| "Backend not available" | Run `uvicorn backend.app.main:app --reload` |
| "Gemini quota exceeded" | Switch to Ollama in model selector |
| "Local model unavailable" | Run `ollama run llama3` |
| "Knowledge base not ready" | Run data preparation scripts |
| Frontend won't start | Run `npm install` in `web/` folder |
| Port 8000 in use | `lsof -i :8000` then kill the process |
| Port 3000 in use | `npm run dev -- --port 3001` |

---

## Development URLs

| Service | URL |
|---------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Ollama | http://localhost:11434 |

---

## Production Build

```bash
cd web
npm run build
npm start
```

For backend production:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 2
```
