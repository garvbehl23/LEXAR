"""LEXAR — Indian Legal AI — Production Frontend"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Generator, Optional

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="LEXAR",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS  — single clean block, no duplication, no conflicts
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ── 1. Typography ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: Inter, system-ui, -apple-system, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* ── 2. Remove Streamlit chrome ────────────────────────────────────────── */
#MainMenu, footer, .stDeployButton,
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stStatusWidget"], .stAppHeader { display: none !important; }

/* ── 3. Full-screen app background ────────────────────────────────────── */
.stApp,
section[data-testid="stMain"],
[data-testid="stAppViewContainer"] { background: #ffffff !important; }

[data-testid="stMainBlockContainer"] {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── 4. Block container — alignment + bottom overflow fix ─────────────── */
/*
   With layout="wide", the main area already starts AFTER the sidebar.
   We cap width and push left padding to align text column with sidebar edge.
   padding-bottom prevents content hiding under fixed chat input.
*/
.block-container {
    max-width: 900px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 0 !important;
    padding-bottom: 90px !important;
}

/* ── 5. Sidebar ────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #1c1c1c !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

[data-testid="stSidebar"] p {
    color: #6b7280 !important;
    font-size: 0.81rem !important;
    margin: 0 !important;
}

/* Sidebar buttons — base */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #d4d4d4 !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 0.83rem !important;
    font-weight: 400 !important;
    padding: 0.4rem 0.75rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    box-shadow: none !important;
    height: auto !important;
    min-height: 34px !important;
    white-space: normal !important;
    word-break: break-word !important;
    transition: background 0.1s ease !important;
    transform: none !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1c1c1c !important;
    color: #f5f5f5 !important;
    transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: #181818 !important;
    color: #525252 !important;
    cursor: default !important;
    transform: none !important;
}

/* New Chat button */
[data-testid="stSidebar"] .new-chat-btn button {
    background: #1c1c1c !important;
    color: #f5f5f5 !important;
    border: 1px solid #2a2a2a !important;
    font-weight: 500 !important;
    font-size: 0.86rem !important;
}
[data-testid="stSidebar"] .new-chat-btn button:hover { background: #252525 !important; }

/* Active nav */
[data-testid="stSidebar"] .nav-active button {
    background: #1c1c1c !important;
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* Sidebar selects */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #1a1a1a !important;
    border-color: #2a2a2a !important;
    border-radius: 6px !important;
    color: #d4d4d4 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] label {
    color: #525252 !important;
    font-size: 0.62rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── 6. Chat input — FIXED bottom, aligned with sidebar ───────────────── */
/*
   CRITICAL: position:fixed with left:260px ensures the bar starts
   right after the sidebar and does not float or overlap it.
*/
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 260px !important;
    right: 0 !important;
    background: #ffffff !important;
    border-top: 1px solid #e5e7eb !important;
    padding: 12px 16px !important;
    z-index: 100 !important;
}

[data-testid="stChatInputTextArea"],
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1px solid #e5e7eb !important;
    padding: 12px 48px 12px 16px !important;
    font-size: 14px !important;
    background: #f9fafb !important;
    color: #111827 !important;
    font-family: Inter, system-ui, sans-serif !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    resize: none !important;
}
[data-testid="stChatInputTextArea"]:focus,
[data-testid="stChatInput"] textarea:focus {
    border-color: #111827 !important;
    box-shadow: 0 0 0 2px rgba(17,24,39,0.08) !important;
    background: #ffffff !important;
    outline: none !important;
}
[data-testid="stChatInputTextArea"]::placeholder { color: #9ca3af !important; }

[data-testid="stChatInputSubmitButton"] > button {
    background: #111827 !important;
    border-radius: 10px !important;
    border: none !important;
    height: 38px !important;
    width: 38px !important;
    transition: background 0.15s ease !important;
    transform: none !important;
}
[data-testid="stChatInputSubmitButton"] > button:hover {
    background: #1f2937 !important;
    transform: translateY(-1px) !important;
}

/* ── 7. Welcome screen ─────────────────────────────────────────────────── */
.welcome-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 4rem 1rem 2rem;
    text-align: center;
}
.welcome-title {
    font-size: 32px;
    font-weight: 600;
    color: #111827;
    letter-spacing: -0.04em;
    margin-bottom: 0.5rem;
}
.welcome-sub {
    font-size: 0.93rem;
    color: #6b7280;
    max-width: 420px;
    line-height: 1.65;
    margin-bottom: 2.5rem;
}

/* Suggestion buttons */
.sugg-btn button {
    background: #111827 !important;
    color: #e5e7eb !important;
    border: 1px solid transparent !important;
    border-radius: 12px !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 14px 18px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    width: 100% !important;
    height: auto !important;
    min-height: 56px !important;
    white-space: normal !important;
    line-height: 1.4 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
    transform: none !important;
}
.sugg-btn button:hover {
    background: #1f2937 !important;
    border-color: #374151 !important;
    transform: translateY(-2px) !important;
}

/* ── 8. Chat bubble rows ───────────────────────────────────────────────── */
.chat-user-row {
    display: flex;
    justify-content: flex-end;
    margin: 8px 0;
    padding: 0;
}

.chat-ai-row {
    display: flex;
    justify-content: flex-start;
    margin: 8px 0;
    padding: 0;
}

/* USER bubble — dark, right-aligned, asymmetric radius */
.user-bubble {
    background: #111827;
    color: #f9fafb;
    border-radius: 18px 18px 6px 18px;
    padding: 10px 14px;
    font-size: 14px;
    line-height: 1.6;
    max-width: 65%;
    word-break: break-word;
}

/* AI bubble — white, subtle border, left-aligned */
.ai-bubble {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px 18px 18px 6px;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.7;
    max-width: 78%;
    word-break: break-word;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}

/* ── 9. Streaming animation ────────────────────────────────────────────── */
@keyframes fadeIn {
    from { opacity: 0.5; }
    to   { opacity: 1; }
}

.streaming-text {
    font-size: 14px;
    color: #111827;
    line-height: 1.75;
    white-space: pre-wrap;
    animation: fadeIn 0.2s ease forwards;
}

/* Subtle blinking cursor */
@keyframes blink {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0; }
}
.cursor {
    display: inline-block;
    width: 2px;
    height: 14px;
    background: #9ca3af;
    margin-left: 2px;
    vertical-align: middle;
    animation: blink 1s ease-in-out infinite;
    border-radius: 1px;
}

/* ── 10. Typing indicator ──────────────────────────────────────────────── */
.typing-wrap { margin: 8px 0; }
.typing-bubble {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 18px 18px 18px 6px;
    padding: 12px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.typing-dots { display: flex; align-items: center; gap: 5px; }
.typing-dots span {
    width: 7px;
    height: 7px;
    background: #9ca3af;
    border-radius: 50%;
    animation: dot-bounce 1.2s infinite ease-in-out;
}
.typing-dots span:nth-child(2) { animation-delay: 0.16s; }
.typing-dots span:nth-child(3) { animation-delay: 0.32s; }

@keyframes dot-bounce {
    0%, 60%, 100% { transform: translateY(0);   opacity: 0.5; }
    30%           { transform: translateY(-7px); opacity: 1;   }
}

/* ── 11. Structured AI response ────────────────────────────────────────── */
.response-wrap { font-size: 14px; color: #111827; }

.r-section { margin-bottom: 0; }

.r-label {
    display: block;
    font-size: 10px;
    font-weight: 700;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 6px;
}

.r-answer {
    font-size: 14px;
    color: #111827;
    line-height: 1.8;
}

.r-divider {
    border: none;
    border-top: 1px solid #f3f4f6;
    margin: 14px 0;
}

.law-pills { display: flex; flex-wrap: wrap; gap: 6px; }
.law-pill {
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    color: #374151;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 5px;
    letter-spacing: 0.01em;
}

.r-explanation {
    background: #fafafa;
    border-left: 2px solid #d1d5db;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 13px;
    color: #6b7280;
    line-height: 1.7;
    font-style: italic;
}
.r-expl-src {
    font-size: 10px;
    font-weight: 600;
    color: #9ca3af;
    font-style: normal;
    margin-top: 5px;
    letter-spacing: 0.03em;
}

.r-sources { display: flex; flex-direction: column; gap: 6px; }
.r-src {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    padding: 8px 10px;
    background: #fafafa;
    border: 1px solid #f0f0f0;
    border-radius: 8px;
}
.r-src-n {
    font-size: 10px;
    font-weight: 700;
    color: #d1d5db;
    min-width: 14px;
    padding-top: 2px;
    flex-shrink: 0;
}
.r-src-body { flex: 1; min-width: 0; }
.r-src-text { font-size: 12px; color: #6b7280; line-height: 1.5; }
.r-src-meta { font-size: 10px; color: #9ca3af; font-weight: 500; margin-top: 2px; }

.r-meta {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid #f3f4f6;
}
.r-meta-item { font-size: 11px; color: #9ca3af; }
.r-conf      { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 9999px; }
.conf-hi { background: #f0fdf4; color: #166534; }
.conf-md { background: #fffbeb; color: #92400e; }
.conf-lo { background: #fef2f2; color: #991b1b; }
.r-mode  { font-size: 10px; color: #d1d5db; font-style: italic; }

/* ── 12. Non-chat page structure ───────────────────────────────────────── */
.pg-hdr {
    padding: 1.4rem 0 1rem;
    border-bottom: 1px solid #f3f4f6;
    margin-bottom: 1.5rem;
}
.pg-title { font-size: 1.1rem; font-weight: 600; color: #111827; }
.pg-sub   { font-size: 0.82rem; color: #6b7280; margin-top: 3px; }

/* ── 13. Stat cards ─────────────────────────────────────────────────────── */
.stat-card {
    background: #fafafa;
    border: 1px solid #f0f0f0;
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.stat-lbl {
    font-size: 10px;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 4px;
}
.stat-val { font-size: 1.55rem; font-weight: 700; color: #111827; line-height: 1; }
.stat-sub { font-size: 12px; color: #6b7280; margin-top: 4px; }

/* ── 14. Data tables & badges ───────────────────────────────────────────── */
.dtable {
    width: 100%;
    border-collapse: collapse;
    background: #fafafa;
    border: 1px solid #f0f0f0;
    border-radius: 10px;
    overflow: hidden;
    font-size: 13px;
}
.dtable th {
    padding: 8px 12px;
    text-align: left;
    font-size: 10px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    background: #f5f5f5;
    border-bottom: 1px solid #e5e7eb;
}
.dtable td {
    padding: 8px 12px;
    color: #374151;
    border-bottom: 1px solid #f5f5f5;
}
.dtable tr:last-child td { border-bottom: none; }

.badge { display: inline-block; font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 9999px; }
.badge-ok   { background: #f0fdf4; color: #166534; }
.badge-miss { background: #fef2f2; color: #991b1b; }

/* ── 15. Quick upload panel (inline in chat) ───────────────────────────── */
.upload-panel {
    border: 1px dashed #e5e7eb;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
    background: #fafafa;
}
.upload-panel-title {
    font-size: 12px;
    font-weight: 600;
    color: #374151;
    margin-bottom: 8px;
}
.upload-success {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #166534;
    font-size: 12px;
    font-weight: 500;
    padding: 6px 12px;
    border-radius: 8px;
    margin-top: 4px;
}

/* ── 16. About page cards ───────────────────────────────────────────────── */
.card {
    background: #111827;
    color: #e5e7eb;
    padding: 14px 18px;
    border-radius: 10px;
    margin: 8px 0;
    border: 1px solid transparent;
    transition: all 0.2s ease;
}
.card:hover { background: #1f2937; border-color: #374151; }
.card-title { font-weight: 600; font-size: 13px; margin-bottom: 4px; }
.card-text  { font-size: 12px; color: #9ca3af; line-height: 1.5; }

/* ── 17. Misc ───────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 13px !important; }
[data-testid="stMarkdownContainer"] p {
    color: #4b5563 !important;
    font-size: 13px !important;
    line-height: 1.6 !important;
}
[data-testid="stExpander"] { border: 1px solid #f0f0f0 !important; border-radius: 8px !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-thumb { background: #e5e7eb; border-radius: 10px; }
hr { border-color: #f3f4f6 !important; }
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PAGES = ["Chat", "Upload", "Evaluation", "About"]

INDEX_OPTIONS = {
    "LEXAR Medium — All statutes": "lexar_medium",
    "IPC + CrPC + IEA":            "ipc_crpc_iea",
    "IPC + CrPC":                   "ipc_crpc",
    "IPC only":                     "ipc",
}

OLLAMA_MODELS = ["llama3", "llama3.1", "llama3.2", "mistral", "gemma2", "phi3", "deepseek-r1"]

SUGGESTIONS = [
    ("Punishment for murder",          "What is the punishment for murder under the Indian Penal Code?"),
    ("Bail under CrPC 437",            "What are the conditions for bail under CrPC Section 437?"),
    ("Burden of proof",                "Who bears the burden of proof in criminal cases under the Indian Evidence Act?"),
    ("Rights in police custody",       "What are the legal rights of an accused during police custody in India?"),
]

_CITE_RE = re.compile(
    r"\b(?:Section|Sec\.?|s\.)\s*\d+[A-Z]?(?:\(\d+\))?(?:\s+(?:of\s+)?(?:IPC|CrPC|IEA))?"
    r"|\b(?:IPC|CrPC|IEA)\s+(?:Section|Sec\.?|s\.)\s*\d+[A-Z]?(?:\(\d+\))?",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_pipeline(index_key: str, _sig: int):
    from lexar.lexar_pipeline import LexarPipeline
    return LexarPipeline(index_name=index_key)


def _chunk_sig() -> int:
    total = 0
    for p in sorted((ROOT / "data" / "processed_docs").glob("*.json")):
        try:
            total += p.stat().st_mtime_ns
        except OSError:
            pass
    return total


def retrieve_evidence(query: str, index_key: str, top_k: int, rerank_k: int) -> tuple[list, float]:
    pipe = _load_pipeline(index_key, _chunk_sig())
    retrieved = pipe._retrieve(query, has_user_docs=False, top_k=top_k)
    if not retrieved:
        return [], 0.0
    return pipe._rerank_and_score(query, retrieved, rerank_k)


def _build_rag_prompt(query: str, evidence: list) -> str:
    blocks = []
    for i, ev in enumerate(evidence, 1):
        statute = ev.get("metadata", {}).get("statute", "")
        section = ev.get("metadata", {}).get("section", "")
        label   = f"[{i}] {statute} Sec. {section}" if section else f"[{i}] {statute or 'Source'}"
        blocks.append(f"{label}:\n{ev.get('text', '').strip()}")

    return (
        "You are LEXAR, an expert Indian legal AI assistant.\n"
        "Answer ONLY using the evidence below. Do not use any outside knowledge.\n\n"
        "Evidence:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {query}\n\n"
        "Format your answer as:\n"
        "**Answer:** [Direct answer]\n"
        "**Applicable Law:** [Section numbers and statute names]\n"
        "**Explanation:** [Detailed legal reasoning]\n\nAnswer:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND — STREAMING GENERATORS
# ─────────────────────────────────────────────────────────────────────────────

def _gemini_stream(prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> Generator[str, None, None]:
    import google.generativeai as genai  # type: ignore
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(
        model_name=model,
        system_instruction="You are LEXAR, an expert Indian legal AI. Answer strictly from provided evidence.",
    )
    for chunk in m.generate_content(prompt, stream=True):
        if hasattr(chunk, "text") and chunk.text:
            yield chunk.text


def _ollama_stream(prompt: str, model: str = "llama3") -> Generator[str, None, None]:
    import requests as _req
    resp = _req.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    for raw in resp.iter_lines():
        if not raw:
            continue
        try:
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except json.JSONDecodeError:
            continue
        if data.get("done"):
            break
        token = data.get("response", "")
        if token:
            yield token


def _flan_stream(pipeline, query: str, evidence: list) -> Generator[str, None, None]:
    result = pipeline._generate_with_evidence(query, evidence)
    text   = result.get("answer", "Unable to generate an answer from the provided evidence.")
    words  = text.split()
    for i, word in enumerate(words):
        yield word + ("" if i == len(words) - 1 else " ")


def _ollama_available(timeout: float = 2.0) -> bool:
    try:
        import requests as _req
        return _req.get("http://localhost:11434/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def _resolve_stream(prompt: str, index_key: str, evidence: list, confidence: float) -> tuple[Generator, str]:
    """
    Return (generator, mode_label) using fallback order:
    Gemini → Ollama → flan-t5
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        try:
            gen   = _gemini_stream(prompt, api_key)
            first = next(gen)
            def _prepend(tok, rest):
                yield tok
                yield from rest
            return _prepend(first, gen), "gemini"
        except Exception:
            pass

    if _ollama_available():
        model = st.session_state.get("ollama_model", "llama3")
        try:
            return _ollama_stream(prompt, model=model), f"ollama/{model}"
        except Exception:
            pass

    pipe = _load_pipeline(index_key, _chunk_sig())
    return _flan_stream(pipe, prompt, evidence), "flan-t5"


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE PARSING & HTML
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sections(text: str) -> dict[str, str]:
    patterns = {
        "answer":  re.compile(r"\*\*Answer[:\s]*\*\*\s*(.*?)(?=\*\*(?:Applicable|Explanation)|$)", re.DOTALL | re.IGNORECASE),
        "law":     re.compile(r"\*\*Applicable Law[:\s]*\*\*\s*(.*?)(?=\*\*Explanation|$)",         re.DOTALL | re.IGNORECASE),
        "explain": re.compile(r"\*\*Explanation[:\s]*\*\*\s*(.*?)$",                                  re.DOTALL | re.IGNORECASE),
    }
    out: dict[str, str] = {}
    for k, pat in patterns.items():
        m = pat.search(text)
        out[k] = m.group(1).strip() if m else ""
    if not any(out.values()):
        out["answer"] = text.strip()
    return out


def _law_pills(text: str, ev_ids: list) -> list[str]:
    seen: list[str] = []
    for m in _CITE_RE.finditer(text):
        tag = m.group(0).strip()
        if tag and tag not in seen:
            seen.append(tag)
    if not seen:
        for eid in ev_ids[:5]:
            parts = str(eid).split("_", 1)
            if len(parts) == 2:
                tag = f"{parts[0].upper()} Sec. {parts[1]}"
                if tag not in seen:
                    seen.append(tag)
    return seen[:8]


def _pct(conf: float) -> int:
    try:
        v = float(conf)
        return 0 if v <= 0 else (int(v * 100) if v <= 1.0 else min(99, int(v)))
    except (TypeError, ValueError):
        return 0


def _conf_cls(pct: int) -> str:
    return "conf-hi" if pct >= 70 else ("conf-md" if pct >= 40 else "conf-lo")


def _e(s: str) -> str:
    return _html.escape(str(s))


def render_response_html(content: dict) -> str:
    text       = content.get("text", "")
    evidence   = content.get("evidence", [])
    confidence = content.get("confidence", 0.0)
    mode       = content.get("mode", "")
    elapsed    = content.get("elapsed", 0.0)
    ev_ids     = [ev.get("chunk_id", "") for ev in evidence]

    secs  = _parse_sections(text)
    pct   = _pct(confidence)
    cls   = _conf_cls(pct)
    cites = _law_pills(secs.get("law") or text, ev_ids)

    out: list[str] = ['<div class="response-wrap">']

    # Answer
    out.append(
        f'<div class="r-section">'
        f'<span class="r-label">Answer</span>'
        f'<div class="r-answer">{_e(secs.get("answer") or text)}</div>'
        f'</div>'
    )

    # Applicable Law
    if cites:
        pills = "".join(f'<span class="law-pill">{_e(c)}</span>' for c in cites)
        out.append(
            f'<hr class="r-divider">'
            f'<div class="r-section">'
            f'<span class="r-label">Applicable Law</span>'
            f'<div class="law-pills">{pills}</div>'
            f'</div>'
        )

    # Explanation
    expl = secs.get("explain", "")
    if expl:
        out.append(
            f'<hr class="r-divider">'
            f'<div class="r-section">'
            f'<span class="r-label">Explanation</span>'
            f'<div class="r-answer">{_e(expl)}</div>'
            f'</div>'
        )
    elif evidence:
        ev0     = evidence[0]
        raw_ev  = ev0.get("text", "")
        snippet = _e(raw_ev[:400].rstrip()) + ("..." if len(raw_ev) > 400 else "")
        statute = _e(ev0.get("metadata", {}).get("statute", ""))
        section = _e(ev0.get("metadata", {}).get("section", ""))
        src_lbl = " · ".join(filter(None, [statute, f"Section {section}" if section else ""]))
        src_div = f'<div class="r-expl-src">{src_lbl}</div>' if src_lbl else ""
        out.append(
            f'<hr class="r-divider">'
            f'<div class="r-section">'
            f'<span class="r-label">Explanation</span>'
            f'<div class="r-explanation">{snippet}{src_div}</div>'
            f'</div>'
        )

    # Sources
    if evidence:
        items: list[str] = []
        for i, ev in enumerate(evidence[:5], 1):
            txt   = _e(ev.get("text", "")[:155].rstrip())
            dots  = "..." if len(ev.get("text", "")) > 155 else ""
            stk   = _e(ev.get("metadata", {}).get("statute", ""))
            sec   = _e(ev.get("metadata", {}).get("section", ""))
            meta  = " · ".join(filter(None, [stk, f"Sec. {sec}" if sec else ""]))
            m_div = f'<div class="r-src-meta">{meta}</div>' if meta else ""
            items.append(
                f'<div class="r-src">'
                f'<span class="r-src-n">{i}</span>'
                f'<div class="r-src-body">'
                f'<div class="r-src-text">{txt}{dots}</div>{m_div}'
                f'</div></div>'
            )
        out.append(
            f'<hr class="r-divider">'
            f'<div class="r-section">'
            f'<span class="r-label">Sources</span>'
            f'<div class="r-sources">{"".join(items)}</div>'
            f'</div>'
        )

    # Meta bar
    meta_parts: list[str] = []
    if evidence:
        meta_parts.append(f'<span class="r-meta-item">{len(evidence)} passages</span>')
    if elapsed:
        meta_parts.append(f'<span class="r-meta-item">{elapsed:.1f}s</span>')
    meta_parts.append(f'<span class="r-conf {cls}">{pct}% confidence</span>')
    if mode:
        meta_parts.append(f'<span class="r-mode">via {_e(mode)}</span>')
    out.append(f'<div class="r-meta">{"".join(meta_parts)}</div>')

    out.append("</div>")
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

def _init() -> None:
    defaults: dict = {
        "page":          "Chat",
        "messages":      [],
        "sessions":      [],
        "index_key":     "lexar_medium",
        "top_k":         10,
        "rerank_k":      5,
        "ollama_model":  "llama3",
        "has_user_docs": False,
        "upload_log":    [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _new_chat() -> None:
    msgs = st.session_state.messages
    if msgs:
        first_q = next(
            (m["content"] for m in msgs if m["role"] == "user" and isinstance(m["content"], str)),
            "Untitled",
        )
        st.session_state.sessions.append({"title": first_q[:60], "messages": msgs[:]})
    st.session_state.messages = []
    st.session_state.page = "Chat"
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR  — minimal: title + new chat + nav + history + settings
# ─────────────────────────────────────────────────────────────────────────────

def _sidebar() -> None:
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="padding:1.1rem 1rem 0.85rem;border-bottom:1px solid #1c1c1c;margin-bottom:0.4rem">
          <div style="font-size:1.05rem;font-weight:700;color:#f5f5f5;letter-spacing:-0.03em">LEXAR</div>
          <div style="font-size:0.58rem;color:#404040;letter-spacing:0.1em;text-transform:uppercase;margin-top:2px">Indian Legal AI</div>
        </div>
        """, unsafe_allow_html=True)

        # New Chat
        st.markdown('<div class="new-chat-btn">', unsafe_allow_html=True)
        if st.button("+ New Chat", key="btn_new_chat", use_container_width=True):
            _new_chat()
        st.markdown("</div>", unsafe_allow_html=True)

        _vgap("0.5rem")

        # Navigation
        _slbl("Navigation")
        for page in PAGES:
            active = st.session_state.page == page
            st.markdown(f'<div class="{"nav-active" if active else ""}">', unsafe_allow_html=True)
            if st.button(page, key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # History
        msgs = st.session_state.messages
        past = st.session_state.sessions
        if msgs or past:
            _vgap("0.75rem")
            _slbl("Recent")
            if msgs:
                first_q = next(
                    (m["content"] for m in msgs if m["role"] == "user" and isinstance(m["content"], str)),
                    "Current chat",
                )
                st.button(
                    first_q[:36] + ("..." if len(first_q) > 36 else ""),
                    key="curr_sess", use_container_width=True, disabled=True,
                )
            for i, sess in enumerate(reversed(past[-8:])):
                title = sess.get("title", "Chat")[:36]
                if st.button(title, key=f"hist_{i}", use_container_width=True):
                    st.session_state.messages = sess["messages"]
                    st.session_state.page = "Chat"
                    st.rerun()

        # Settings — collapsed by default
        st.markdown(
            "<div style='height:0.75rem;border-top:1px solid #1c1c1c;margin-top:0.75rem'></div>",
            unsafe_allow_html=True,
        )
        _slbl("Settings")

        lbl = st.selectbox(
            "Knowledge Base",
            list(INDEX_OPTIONS.keys()),
            index=0,
            key="kb_select",
            label_visibility="visible",
        )
        st.session_state.index_key = INDEX_OPTIONS[lbl]

        om = st.selectbox(
            "Ollama Model",
            OLLAMA_MODELS,
            index=0,
            key="ollama_select",
            label_visibility="visible",
        )
        st.session_state.ollama_model = om

        _vgap("0.5rem")


def _slbl(text: str) -> None:
    st.markdown(
        f'<div style="padding:0.15rem 0.75rem 0.2rem">'
        f'<span style="font-size:0.58rem;font-weight:600;color:#333333;'
        f'text-transform:uppercase;letter-spacing:0.09em">{text}</span></div>',
        unsafe_allow_html=True,
    )


def _vgap(h: str) -> None:
    st.markdown(f"<div style='height:{h}'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHAT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _render_messages(messages: list) -> None:
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(
                f'<div class="chat-user-row">'
                f'<span class="user-bubble">{_e(content) if isinstance(content, str) else ""}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            if not isinstance(content, dict):
                continue
            if content.get("status") == "no_evidence":
                st.markdown(
                    '<div class="chat-ai-row">'
                    '<div class="ai-bubble" style="color:#6b7280;font-size:13px">'
                    'No relevant passages found. Try rephrasing or choosing a broader knowledge base.'
                    '</div></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-ai-row">'
                    f'<div class="ai-bubble">{render_response_html(content)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


def _run_query(query: str) -> None:
    """Full pipeline: show user bubble → typing → stream → persist → rerun."""
    ik    = st.session_state.index_key
    top_k = st.session_state.top_k
    rk    = st.session_state.rerank_k

    # User bubble
    st.markdown(
        f'<div class="chat-user-row">'
        f'<span class="user-bubble">{_e(query)}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Typing indicator while retrieving evidence
    typing_ph = st.empty()
    typing_ph.markdown(
        '<div class="chat-ai-row typing-wrap">'
        '<div class="typing-bubble">'
        '<div class="typing-dots"><span></span><span></span><span></span></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    t0 = time.time()
    evidence, confidence = retrieve_evidence(query, ik, top_k, rk)
    typing_ph.empty()

    # No evidence
    if not evidence:
        st.session_state.messages.append({"role": "user",      "content": query})
        st.session_state.messages.append({"role": "assistant", "content": {
            "status": "no_evidence", "text": "", "evidence": [], "confidence": 0.0,
        }})
        st.rerun()
        return

    # Build prompt and resolve streaming backend
    prompt     = _build_rag_prompt(query, evidence)
    stream_gen, mode = _resolve_stream(prompt, ik, evidence, confidence)

    # Stream into placeholder — update every 2–3 tokens for smooth animation
    stream_ph   = st.empty()
    full_text   = ""
    tok_counter = 0

    for token in stream_gen:
        full_text   += token
        tok_counter += 1
        if tok_counter % 3 == 0:
            stream_ph.markdown(
                f'<div class="chat-ai-row">'
                f'<div class="ai-bubble">'
                f'<div class="streaming-text">{full_text}'
                f'<span class="cursor"></span></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    # Final frame — no cursor
    stream_ph.markdown(
        f'<div class="chat-ai-row">'
        f'<div class="ai-bubble">'
        f'<div class="streaming-text">{full_text}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Persist and re-render structured response
    result = {
        "status":     "success",
        "text":       full_text,
        "evidence":   evidence,
        "confidence": confidence,
        "mode":       mode,
        "elapsed":    time.time() - t0,
    }
    st.session_state.messages.append({"role": "user",      "content": query})
    st.session_state.messages.append({"role": "assistant", "content": result})
    st.rerun()


def _quick_upload() -> None:
    """Inline upload panel at top of chat — collapses when not needed."""
    with st.expander("Attach a document to this conversation", expanded=False):
        uploaded = st.file_uploader(
            "PDF only, max 10 MB",
            type=["pdf"],
            key="chat_uploader",
            label_visibility="collapsed",
        )
        if uploaded:
            with st.spinner("Processing..."):
                try:
                    from lexar.ingestion.pdf_extractor import extract_text_from_pdf
                    from lexar.chunking.statute_chunker import chunk_statute_text
                    from lexar.utils.text_cleaner import clean_text

                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name

                    text   = clean_text(extract_text_from_pdf(Path(tmp_path)))
                    chunks = chunk_statute_text(text, statute_name=uploaded.name, year=0)
                    os.unlink(tmp_path)

                    st.session_state.upload_log.append({"name": uploaded.name, "chunks": len(chunks)})
                    st.session_state.has_user_docs = True

                    # Inject a system notice into the chat history
                    st.session_state.messages.append({
                        "role":    "assistant",
                        "content": {
                            "status":     "upload_notice",
                            "text":       f"Document processed: {uploaded.name} ({len(chunks)} chunks ready for search).",
                            "evidence":   [],
                            "confidence": 1.0,
                            "mode":       "system",
                            "elapsed":    0.0,
                        },
                    })
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not process: {exc}")


def _page_chat() -> None:
    pending = st.session_state.pop("_pending_q", None)
    if pending:
        _run_query(pending)
        return

    msgs = st.session_state.messages

    # Quick upload panel — always available at top
    _quick_upload()

    if not msgs:
        st.markdown("""
        <div class="welcome-wrap">
          <div class="welcome-title">LEXAR</div>
          <div class="welcome-sub">Ask anything about Indian law. Every answer is grounded in cited statute passages.</div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        for idx, (short, full) in enumerate(SUGGESTIONS):
            col = c1 if idx % 2 == 0 else c2
            with col:
                st.markdown('<div class="sugg-btn">', unsafe_allow_html=True)
                if st.button(f"{short}\n\n{full}", key=f"sugg_{idx}", use_container_width=True):
                    st.session_state["_pending_q"] = full
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        _render_messages(msgs)

    query = st.chat_input("Ask a legal question...")
    if query and query.strip():
        _run_query(query.strip())


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _page_upload() -> None:
    st.markdown("""
    <div class="pg-hdr">
      <div class="pg-title">Upload Documents</div>
      <div class="pg-sub">Add PDFs to search alongside the built-in statute indices</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop a PDF or click to browse (max 10 MB)",
        type=["pdf"],
        key="page_uploader",
    )

    if uploaded:
        with st.spinner("Extracting and chunking..."):
            try:
                from lexar.ingestion.pdf_extractor import extract_text_from_pdf
                from lexar.chunking.statute_chunker import chunk_statute_text
                from lexar.utils.text_cleaner import clean_text

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name

                text   = clean_text(extract_text_from_pdf(Path(tmp_path)))
                chunks = chunk_statute_text(text, statute_name=uploaded.name, year=0)
                os.unlink(tmp_path)

                st.session_state.upload_log.append({"name": uploaded.name, "chunks": len(chunks)})
                st.session_state.has_user_docs = True
                st.success(f"Processed — {len(chunks)} chunks extracted from {uploaded.name}")
            except Exception as exc:
                st.error(f"Could not process: {exc}")

    log = st.session_state.upload_log
    if log:
        st.markdown("<div style='margin-top:1.2rem'>", unsafe_allow_html=True)
        rows = "".join(
            f"<tr><td style='font-weight:500'>{_e(d['name'])}</td>"
            f"<td style='color:#6b7280'>{d['chunks']} chunks</td></tr>"
            for d in log
        )
        st.markdown(
            f'<table class="dtable"><thead><tr><th>File</th><th>Chunks</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _page_evaluation() -> None:
    st.markdown("""
    <div class="pg-hdr">
      <div class="pg-title">Evaluation</div>
      <div class="pg-sub">System diagnostics, index statistics, session metrics</div>
    </div>
    """, unsafe_allow_html=True)

    processed  = ROOT / "data" / "processed_docs"
    faiss_dir  = ROOT / "data" / "faiss_index"
    has_gemini = bool(os.getenv("GEMINI_API_KEY", "").strip())
    has_ollama = _ollama_available(timeout=1.5)

    chunk_counts: dict[str, int] = {}
    for name in ("ipc", "ipc_crpc", "ipc_crpc_iea", "lexar_medium"):
        p = processed / f"{name}_chunks.json"
        try:
            chunk_counts[name] = len(json.loads(p.read_text())) if p.exists() else 0
        except Exception:
            chunk_counts[name] = 0

    active_idx  = sum(1 for n in chunk_counts if (faiss_dir / f"{n}.index").exists())
    sess_q      = sum(1 for m in st.session_state.messages if m["role"] == "user")
    gen_backend = "Gemini" if has_gemini else ("Ollama" if has_ollama else "flan-t5")

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val, sub in [
        (c1, "Active Indices",  str(active_idx),                             "FAISS loaded"),
        (c2, "Chunks",          f"{chunk_counts.get('lexar_medium', 0):,}",  "LEXAR Medium"),
        (c3, "Generator",       gen_backend,                                  "active backend"),
        (c4, "Session Queries", str(sess_q),                                  "this session"),
    ]:
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-lbl">{lbl}</div>'
                f'<div class="stat-val">{val}</div><div class="stat-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Backends
    _tbl_header("Backends")
    backends = [
        ("Gemini API", has_gemini, "GEMINI_API_KEY set" if has_gemini else "API key not found"),
        ("Ollama",     has_ollama, "Reachable at :11434" if has_ollama else "Not running"),
        ("flan-t5",    True,       "Always available — final fallback"),
    ]
    rows = "".join(
        f'<tr><td style="font-weight:500">{n}</td>'
        f'<td><span class="badge {"badge-ok" if ok else "badge-miss"}">'
        f'{"Ready" if ok else "Unavailable"}</span></td>'
        f'<td style="color:#6b7280">{note}</td></tr>'
        for n, ok, note in backends
    )
    st.markdown(
        f'<table class="dtable"><thead><tr><th>Backend</th><th>Status</th><th>Note</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # Indices
    _tbl_header("Indices")
    rows2 = ""
    for name, label in [
        ("ipc",          "Indian Penal Code"),
        ("ipc_crpc",     "IPC + CrPC"),
        ("ipc_crpc_iea", "IPC + CrPC + IEA"),
        ("lexar_medium", "LEXAR Medium"),
    ]:
        ok    = (faiss_dir / f"{name}.index").exists()
        cnt   = f'{chunk_counts.get(name, 0):,}'
        rows2 += (
            f'<tr><td style="font-weight:500">{label}</td>'
            f'<td style="font-family:monospace;font-size:11px;color:#9ca3af">{name}</td>'
            f'<td>{cnt}</td>'
            f'<td><span class="badge {"badge-ok" if ok else "badge-miss"}">'
            f'{"Active" if ok else "Missing"}</span></td></tr>'
        )
    st.markdown(
        f'<table class="dtable"><thead><tr><th>Index</th><th>Key</th><th>Chunks</th><th>Status</th></tr></thead>'
        f'<tbody>{rows2}</tbody></table>',
        unsafe_allow_html=True,
    )


def _tbl_header(text: str) -> None:
    st.markdown(
        f'<div style="font-size:10px;font-weight:600;color:#9ca3af;text-transform:uppercase;'
        f'letter-spacing:0.09em;margin-bottom:8px">{text}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ABOUT PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _page_about() -> None:
    st.markdown("""
    <div class="pg-hdr">
      <div class="pg-title">About LEXAR</div>
      <div class="pg-sub">Evidence-constrained legal AI for Indian law</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1.75rem">
      <div style="font-size:10px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px">What is LEXAR</div>
      <p style="font-size:14px;color:#374151;line-height:1.75;margin:0">
        LEXAR is a retrieval-augmented generation system for Indian legal research.
        Every answer is grounded in retrieved statute passages — hard attention masking
        prevents the model from generating facts not present in the evidence.
      </p>
    </div>

    <div style="margin-bottom:1.75rem">
      <div style="font-size:10px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:12px">Generation Backends</div>
      <div style="display:flex;flex-direction:column;gap:8px">
        <div class="card"><div class="card-title">Gemini 2.0 Flash</div><div class="card-text">Primary. Set GEMINI_API_KEY in .env for streaming generation.</div></div>
        <div class="card"><div class="card-title">Ollama (local)</div><div class="card-text">Fallback. Run llama3, mistral, gemma2 or any model at localhost:11434.</div></div>
        <div class="card"><div class="card-title">flan-t5-base</div><div class="card-text">Final fallback. Always available, offline, CPU-only.</div></div>
      </div>
    </div>

    <div>
      <div style="font-size:10px;font-weight:700;color:#9ca3af;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px">Version</div>
      <p style="font-size:13px;color:#6b7280;line-height:1.7;margin:0">
        LEXAR v1.1.1 — FastAPI · FAISS · sentence-transformers · cross-encoder · Gemini 2.0 Flash · Ollama
      </p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    _init()
    _sidebar()

    page = st.session_state.page
    if   page == "Chat":       _page_chat()
    elif page == "Upload":     _page_upload()
    elif page == "Evaluation": _page_evaluation()
    elif page == "About":      _page_about()


if __name__ == "__main__":
    main()
