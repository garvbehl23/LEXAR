#!/usr/bin/env bash
# LEXAR — one-shot startup script
# Usage: bash start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
BACKEND_PORT=8000
FRONTEND_PORT=8501
BACKEND_URL="http://localhost:$BACKEND_PORT/health/"
FRONTEND_URL="http://localhost:$FRONTEND_PORT"

# ── Colors ────────────────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RESET="\033[0m"

log()  { echo -e "${CYAN}[LEXAR]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK]${RESET}   $*"; }
warn() { echo -e "${YELLOW}[WARN]${RESET} $*"; }

cd "$SCRIPT_DIR"

# ── 0. Create .env if missing ─────────────────────────────────────────────────
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    warn ".env not found — created from .env.example."
    warn "Edit .env and add your GEMINI_API_KEY for AI-powered answers."
fi

# Propagate GEMINI_API_KEY from .env into the shell (so child processes see it)
if [ -f ".env" ]; then
    set -a
    # shellcheck disable=SC1091
    source .env 2>/dev/null || true
    set +a
fi

# ── 1. Python venv ─────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 2. Install dependencies ────────────────────────────────────────────────────
log "Installing / verifying dependencies ..."
pip install --quiet --upgrade pip
if [ -f requirements.txt ]; then
    pip install --quiet -r requirements.txt
fi
# Install lexar package itself (editable)
pip install --quiet -e .

# ── 3. Stub data (so retriever loads without PDFs) ─────────────────────────────
CHUNKS_FILE="$SCRIPT_DIR/data/processed_docs/ipc_chunks.json"
if [ ! -f "$CHUNKS_FILE" ]; then
    log "Generating stub chunk files (no PDFs required) ..."
    python scripts/generate_stub_chunks.py
    ok "Stub chunks generated."
else
    ok "Chunk files already present — skipping generation."
fi

# ── 4. Kill any stale processes on our ports ──────────────────────────────────
for PORT in $BACKEND_PORT $FRONTEND_PORT; do
    PIDS=$(lsof -ti tcp:"$PORT" 2>/dev/null || true)
    if [ -n "$PIDS" ]; then
        warn "Port $PORT already in use — killing stale process(es): $PIDS"
        echo "$PIDS" | xargs kill -9 2>/dev/null || true
        sleep 0.5
    fi
done

# ── 5. Start backend ───────────────────────────────────────────────────────────
log "Starting FastAPI backend on port $BACKEND_PORT ..."
BACKEND_LOG="$SCRIPT_DIR/logs/backend.log"
mkdir -p "$SCRIPT_DIR/logs"
nohup uvicorn backend.app.main:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --log-level info \
    > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$SCRIPT_DIR/logs/backend.pid"
ok "Backend PID: $BACKEND_PID  (log: logs/backend.log)"

# ── 6. Start frontend ──────────────────────────────────────────────────────────
log "Starting Streamlit frontend on port $FRONTEND_PORT ..."
FRONTEND_LOG="$SCRIPT_DIR/logs/frontend.log"
nohup streamlit run frontend/app.py \
    --server.port "$FRONTEND_PORT" \
    --server.headless true \
    --browser.gatherUsageStats false \
    > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$SCRIPT_DIR/logs/frontend.pid"
ok "Frontend PID: $FRONTEND_PID  (log: logs/frontend.log)"

# ── 7. Wait for backend health ─────────────────────────────────────────────────
log "Waiting for backend to be ready ..."
MAX_WAIT=40
ELAPSED=0
until curl -sf "$BACKEND_URL" > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        warn "Backend did not respond in ${MAX_WAIT}s. Check logs/backend.log"
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if curl -sf "$BACKEND_URL" > /dev/null 2>&1; then
    ok "Backend is healthy."
fi

# ── 8. Wait for frontend ───────────────────────────────────────────────────────
log "Waiting for frontend to be ready ..."
ELAPSED=0
until curl -sf "$FRONTEND_URL" > /dev/null 2>&1; do
    if [ "$ELAPSED" -ge "$MAX_WAIT" ]; then
        warn "Frontend did not respond in ${MAX_WAIT}s. Check logs/frontend.log"
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
done
if curl -sf "$FRONTEND_URL" > /dev/null 2>&1; then
    ok "Frontend is ready."
fi

# ── 9. Open Chrome ─────────────────────────────────────────────────────────────
log "Opening $FRONTEND_URL in Google Chrome ..."
if command -v google-chrome &>/dev/null; then
    google-chrome "$FRONTEND_URL" &>/dev/null &
elif command -v google-chrome-stable &>/dev/null; then
    google-chrome-stable "$FRONTEND_URL" &>/dev/null &
elif command -v chromium-browser &>/dev/null; then
    chromium-browser "$FRONTEND_URL" &>/dev/null &
elif command -v chromium &>/dev/null; then
    chromium "$FRONTEND_URL" &>/dev/null &
elif command -v xdg-open &>/dev/null; then
    xdg-open "$FRONTEND_URL" &>/dev/null &
elif command -v open &>/dev/null; then
    open "$FRONTEND_URL"
else
    warn "Could not detect a browser. Open manually: $FRONTEND_URL"
fi

echo ""
echo -e "${BOLD}LEXAR is running${RESET}"
echo -e "  Frontend : ${CYAN}$FRONTEND_URL${RESET}"
echo -e "  Backend  : ${CYAN}http://localhost:$BACKEND_PORT${RESET}"
echo -e "  Logs     : logs/backend.log  |  logs/frontend.log"
echo ""
echo "Press Ctrl+C to stop both services."

# ── 10. Trap SIGINT to clean up ────────────────────────────────────────────────
cleanup() {
    echo ""
    log "Shutting down ..."
    kill "$BACKEND_PID"  2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    ok "Stopped."
}
trap cleanup INT TERM

# Keep script alive so Ctrl+C works
wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
