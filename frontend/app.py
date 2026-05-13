"""
LEXAR Legal AI — Streamlit Frontend  (Spotify-dark redesign)
=============================================================
Run from the project root:
    streamlit run frontend/app.py

Requires (in addition to pyproject.toml deps):
    pip install streamlit plotly
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import streamlit as st

# ── Project root on sys.path so we can import lexar / backend ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LEXAR · Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "LEXAR v1.1.1 — Legal Explainable Augmented Reasoner by Garv Behl"},
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  — Spotify-style dark theme
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & base ─────────────────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Kill Streamlit chrome ────────────────────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* ── App background ───────────────────────────────────────────────────── */
.stApp {
    background: #0a0a0a !important;
}

/* ── Main content area ────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.5rem 2.5rem 4rem 2.5rem !important;
    max-width: 1100px;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
[data-testid="stSidebar"] * { color: #b3b3b3 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* ── Sidebar selectbox / slider / checkbox ────────────────────────────── */
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1a1a1a !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #fff !important;
}
[data-testid="stSidebar"] .stSlider .stSlider { color: #1db954 !important; }
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: #1db954 !important;
    border-color: #1db954 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid #1e1e1e !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    color: #6b6b6b !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.75rem 1.5rem !important;
    border: none !important;
    border-radius: 0 !important;
    transition: color 0.2s !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #fff !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1db954 !important;
    border-bottom: 2px solid #1db954 !important;
}

/* ── Metric cards ─────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #181818 !important;
    border: 1px solid #282828 !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
}
[data-testid="metric-container"] label {
    color: #a7a7a7 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────── */
.stButton > button[kind="primary"] {
    background: #1db954 !important;
    color: #000 !important;
    border: none !important;
    border-radius: 500px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    transition: transform 0.1s, background 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1ed760 !important;
    transform: scale(1.03) !important;
    color: #000 !important;
}
.stButton > button[kind="primary"]:active { transform: scale(0.98) !important; }

.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #b3b3b3 !important;
    border: 1px solid #333 !important;
    border-radius: 500px !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 0.6rem 1.5rem !important;
    transition: border-color 0.2s, color 0.2s !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #fff !important;
    color: #fff !important;
}

/* ── Text inputs / textareas ──────────────────────────────────────────── */
.stTextArea textarea, .stTextInput input {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    color: #ffffff !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s !important;
    padding: 0.85rem 1rem !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #1db954 !important;
    box-shadow: 0 0 0 2px rgba(29,185,84,0.18) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #535353 !important;
}

/* ── File uploader ────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #161616 !important;
    border: 2px dashed #282828 !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #1db954 !important; }
[data-testid="stFileUploader"] * { color: #a7a7a7 !important; }

/* ── Expanders ────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #161616 !important;
    border: 1px solid #242424 !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary {
    color: #b3b3b3 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.04em !important;
}
[data-testid="stExpander"] summary:hover { color: #fff !important; }

/* ── DataFrames ───────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }
[data-testid="stDataFrame"] div[data-grid-canvas] { background: #161616 !important; }

/* ── Progress bar ─────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #1db954 0%, #1ed760 100%) !important;
    border-radius: 4px !important;
}
.stProgress > div > div { background: #282828 !important; border-radius: 4px !important; }

/* ── Alerts / Info blocks ─────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
    font-size: 0.9rem !important;
}

/* ── Divider ──────────────────────────────────────────────────────────── */
hr { border-color: #1e1e1e !important; margin: 1.5rem 0 !important; }

/* ─────────────────────────────────────────────────────────────────────── */
/* CUSTOM COMPONENTS                                                        */
/* ─────────────────────────────────────────────────────────────────────── */

/* ── Page hero ────────────────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0d2818 0%, #0a1a0e 40%, #050d07 100%);
    border: 1px solid #1a3a21;
    border-radius: 16px;
    padding: 2.5rem 2.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(29,185,84,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(29,185,84,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #1db954;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 900;
    color: #ffffff;
    line-height: 1.1;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1rem;
    color: #a7a7a7;
    margin: 0;
    font-weight: 400;
}

/* ── Section label ────────────────────────────────────────────────────── */
.section-label {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #6b6b6b;
    margin-bottom: 0.75rem;
    margin-top: 2rem;
}

/* ── Answer card ──────────────────────────────────────────────────────── */
.answer-card {
    background: #181818;
    border: 1px solid #282828;
    border-radius: 14px;
    padding: 1.75rem 2rem;
    margin: 1.25rem 0;
    position: relative;
}
.answer-card-success { border-left: 3px solid #1db954; }
.answer-card-warn    { border-left: 3px solid #f59e0b; background: #171209; border-color: #2a2010; }
.answer-card-error   { border-left: 3px solid #ef4444; background: #170909; border-color: #2a1010; }

.answer-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #1db954;
    margin-bottom: 0.85rem;
}
.answer-label-warn  { color: #f59e0b; }
.answer-label-error { color: #ef4444; }

.answer-text {
    font-size: 1.05rem;
    line-height: 1.8;
    color: #e4e4e4;
    font-weight: 400;
}

/* ── Citation pill ────────────────────────────────────────────────────── */
.citation-row { margin-top: 1.25rem; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.cit-label { font-size: 0.7rem; color: #535353; letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600; }
.cit-primary {
    background: #1db954;
    color: #000;
    border-radius: 500px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.cit-secondary {
    background: #282828;
    color: #b3b3b3;
    border: 1px solid #333;
    border-radius: 500px;
    padding: 3px 13px;
    font-size: 0.78rem;
    font-weight: 500;
}

/* ── Stage progress row ───────────────────────────────────────────────── */
.stage-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1a1a1a;
}
.stage-dot-active  { width:8px; height:8px; border-radius:50%; background:#1db954; flex-shrink:0; animation: pulse 1s infinite; }
.stage-dot-done    { width:8px; height:8px; border-radius:50%; background:#1db954; flex-shrink:0; }
.stage-dot-pending { width:8px; height:8px; border-radius:50%; background:#333; flex-shrink:0; }
.stage-text-active { color:#e4e4e4; font-size:0.88rem; font-weight:500; }
.stage-text-done   { color:#535353; font-size:0.88rem; text-decoration: line-through; }
.stage-text-pending{ color:#333;    font-size:0.88rem; }

@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 0 rgba(29,185,84,0.4); }
    50%      { box-shadow: 0 0 0 6px rgba(29,185,84,0); }
}

/* ── History pill ─────────────────────────────────────────────────────── */
.history-pill-row { display:flex; flex-wrap:wrap; gap:8px; margin-top:0.5rem; }
.history-pill {
    background: #1a1a1a;
    border: 1px solid #282828;
    border-radius: 500px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #a7a7a7;
    cursor: pointer;
    transition: border-color 0.15s, color 0.15s;
    display: inline-block;
}
.history-pill:hover { border-color: #1db954; color: #fff; }

/* ── Model card ───────────────────────────────────────────────────────── */
.model-card {
    background: #181818;
    border: 1px solid #242424;
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, transform 0.15s;
}
.model-card:hover { border-color: #333; transform: translateY(-2px); }
.model-card-icon { font-size: 1.5rem; margin-bottom: 0.5rem; }
.model-card-title { font-size: 0.85rem; font-weight: 700; color: #fff; margin-bottom: 0.25rem; }
.model-card-desc  { font-size: 0.78rem; color: #6b6b6b; line-height: 1.5; }

/* ── Index badge ──────────────────────────────────────────────────────── */
.index-badge {
    display:inline-block;
    background:#1a2a1e;
    border:1px solid #1d3a23;
    color:#1db954;
    border-radius:6px;
    padding:2px 10px;
    font-size:0.72rem;
    font-weight:700;
    letter-spacing:0.08em;
    text-transform:uppercase;
    margin-bottom:0.3rem;
}

/* ── Eval metric card ─────────────────────────────────────────────────── */
.eval-card {
    background: #181818;
    border: 1px solid #242424;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.eval-card-num  { font-size: 2.4rem; font-weight: 800; color: #1db954; letter-spacing:-0.02em; }
.eval-card-label{ font-size: 0.7rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#535353; margin-top:0.3rem; }

/* ── Sidebar logo area ────────────────────────────────────────────────── */
.sb-logo {
    display:flex; align-items:center; gap:10px;
    padding:0.5rem 0 1.5rem 0;
}
.sb-logo-icon { font-size:1.8rem; }
.sb-logo-text { font-size:1.1rem; font-weight:800; color:#fff !important; letter-spacing:-0.01em; }
.sb-logo-ver  { font-size:0.65rem; color:#535353 !important; font-weight:500; letter-spacing:0.08em; }

/* ── Sidebar nav item ─────────────────────────────────────────────────── */
.sb-nav-label {
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #535353 !important;
    font-weight: 700;
    margin-top: 1.5rem;
    margin-bottom: 0.4rem;
}
.sb-status-dot-green { display:inline-block; width:7px; height:7px; border-radius:50%; background:#1db954; margin-right:6px; vertical-align:middle; }
.sb-status-dot-grey  { display:inline-block; width:7px; height:7px; border-radius:50%; background:#535353; margin-right:6px; vertical-align:middle; }

/* ── Pipeline status bar ──────────────────────────────────────────────── */
.pipeline-status-ready {
    display:flex; align-items:center; gap:8px;
    background:#0d1f12; border:1px solid #1a3a21;
    border-radius:8px; padding:0.6rem 1rem;
    margin-top:0.5rem;
}
.pipeline-status-none {
    display:flex; align-items:center; gap:8px;
    background:#161616; border:1px solid #242424;
    border-radius:8px; padding:0.6rem 1rem;
    margin-top:0.5rem;
}
.ps-label { font-size:0.82rem; color:#b3b3b3 !important; font-weight:500; }
.ps-sub   { font-size:0.72rem; color:#535353 !important; }

/* ── Loading skeleton shimmer ─────────────────────────────────────────── */
.shimmer {
    background: linear-gradient(90deg, #1a1a1a 25%, #242424 50%, #1a1a1a 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    border-radius:8px; height:18px; margin-bottom:10px;
}
@keyframes shimmer { 0%{background-position:200% 0} 100%{background-position:-200% 0} }

/* ── Quick suggestion chips ───────────────────────────────────────────── */
.suggestion-chip {
    display:inline-block;
    background:#161616;
    border:1px solid #242424;
    color:#a7a7a7;
    border-radius:500px;
    padding:6px 16px;
    font-size:0.8rem;
    cursor:pointer;
    transition: all 0.15s;
    margin: 3px;
}
.suggestion-chip:hover { background:#1a2a1e; border-color:#1db954; color:#fff; }

/* ── Upload zone extra ────────────────────────────────────────────────── */
.upload-success-card {
    background:#0d1f12;
    border:1px solid #1a3a21;
    border-radius:12px;
    padding:1.5rem 1.75rem;
    margin-top:1rem;
}
.upload-success-title { font-size:1rem; font-weight:700; color:#1db954; margin-bottom:0.5rem; }
.upload-success-row   { font-size:0.85rem; color:#a7a7a7; margin-bottom:0.25rem; }
.upload-success-val   { color:#fff; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR = ROOT / "data"

INDEX_CONFIGS = {
    "⭐ LEXAR Medium (Recommended)": {
        "chunks": [DATA_DIR / "processed_docs" / "lexar_medium_chunks.json"],
        "index": DATA_DIR / "faiss_index" / "lexar_medium.index",
        "description": "Comprehensive multi-law index (IPC + supplemental statutes). Best for general queries.",
        "short": "LEXAR Medium",
    },
    "📖 IPC Only": {
        "chunks": [DATA_DIR / "processed_docs" / "ipc_chunks.json"],
        "index": DATA_DIR / "faiss_index" / "ipc.index",
        "description": "Indian Penal Code sections only. Fast, focused on criminal offences.",
        "short": "IPC",
    },
    "📚 IPC + CrPC": {
        "chunks": [
            DATA_DIR / "processed_docs" / "ipc_chunks.json",
            DATA_DIR / "processed_docs" / "crpc_chunks.json",
        ],
        "index": DATA_DIR / "faiss_index" / "ipc_crpc.index",
        "description": "IPC + Code of Criminal Procedure. Useful for procedure and offence queries.",
        "short": "IPC+CrPC",
    },
    "🏛 IPC + CrPC + IEA": {
        "chunks": [
            DATA_DIR / "processed_docs" / "ipc_chunks.json",
            DATA_DIR / "processed_docs" / "crpc_chunks.json",
            DATA_DIR / "processed_docs" / "iea_1872_chunks.json",
        ],
        "index": DATA_DIR / "faiss_index" / "ipc_crpc_iea.index",
        "description": "IPC + CrPC + Indian Evidence Act 1872. Broadest statutory coverage.",
        "short": "IPC+CrPC+IEA",
    },
}

EVAL_CHUNKS_PATH = DATA_DIR / "processed_docs" / "ipc2_chunks.json"
EVAL_INDEX_PATH = DATA_DIR / "faiss_index" / "ipc.index"
GOLD_QUERIES_PATH = ROOT / "evaluation" / "gold_queries.json"


# ── Session state bootstrap ────────────────────────────────────────────────
def _init_state():
    defaults = {
        "pipeline": None,
        "pipeline_config": None,
        "user_chunks": None,
        "use_user_doc": False,
        "last_result": None,
        "query_history": [],
        "eval_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Pipeline loader ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_pipeline_cached(index_name: str, chunks_key: str):
    """
    Load the LexarPipeline.  Cached by (index_name, chunks_key) so
    switching indexes re-instantiates while re-running the same config is free.
    """
    from lexar.lexar_pipeline import LexarPipeline
    from lexar.retrieval.ipc_retriever import IPCRetriever

    cfg = INDEX_CONFIGS[index_name]
    chunk_paths = cfg["chunks"]
    index_path = str(cfg["index"])

    # Merge multiple chunk files if needed
    if len(chunk_paths) == 1:
        merged_path = str(chunk_paths[0])
    else:
        all_chunks = []
        for p in chunk_paths:
            with open(p, "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))
        tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        )
        json.dump(all_chunks, tmp, ensure_ascii=False)
        tmp.close()
        merged_path = tmp.name

    ipc = IPCRetriever(merged_path, index_path)
    return LexarPipeline(ipc=ipc)


def _files_exist(index_name: str) -> tuple[bool, list[str]]:
    cfg = INDEX_CONFIGS[index_name]
    missing = []
    if not cfg["index"].exists():
        missing.append(str(cfg["index"]))
    for p in cfg["chunks"]:
        if not p.exists():
            missing.append(str(p))
    return len(missing) == 0, missing


# ── Sidebar ────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
<div class='sb-logo'>
  <span class='sb-logo-icon'>⚖️</span>
  <div>
    <div class='sb-logo-text'>LEXAR</div>
    <div class='sb-logo-ver'>Legal AI · v1.1.1</div>
  </div>
</div>""", unsafe_allow_html=True)

        st.divider()

        # ── Knowledge Base
        st.markdown("<div class='sb-nav-label'>Knowledge Base</div>", unsafe_allow_html=True)
        index_name = st.selectbox(
            "Index",
            options=list(INDEX_CONFIGS.keys()),
            index=0,
            help="The FAISS index + corpus searched for evidence.",
            label_visibility="collapsed",
        )
        st.caption(INDEX_CONFIGS[index_name]["description"])

        ok, missing = _files_exist(index_name)
        if not ok:
            st.error("Missing: " + ", ".join(Path(m).name for m in missing))

        st.divider()

        # ── Advanced Settings
        st.markdown("<div class='sb-nav-label'>Pipeline Settings</div>", unsafe_allow_html=True)
        with st.expander("⚙️ Configure", expanded=False):
            top_k = st.slider("Top-K Retrieval", 3, 20, 10,
                help="Chunks fetched from FAISS before reranking.")
            rerank_k = st.slider("Reranking Top-K", 1, 5, 3,
                help="Top reranked chunks passed to generator.")
            citation_mode = st.radio("Citation Mode", ["inline", "footnote"],
                horizontal=True, help="How citations are attached to the answer.")
            debug_mode = st.checkbox("Debug Mode", value=False,
                help="Return attention distribution per chunk.")
            return_provenance = st.checkbox("Return Provenance", value=False,
                help="Include token-level source attribution.")
        
        # ── expose top_k / rerank_k outside expander with defaults if not interacted
        if "top_k" not in dir():
            top_k = 10
        if "rerank_k" not in dir():
            rerank_k = 3
        if "citation_mode" not in dir():
            citation_mode = "inline"
        if "debug_mode" not in dir():
            debug_mode = False
        if "return_provenance" not in dir():
            return_provenance = False

        st.divider()

        # ── Load Pipeline
        st.markdown("<div class='sb-nav-label'>Engine</div>", unsafe_allow_html=True)
        load_btn = st.button(
            "▶  Load Pipeline",
            use_container_width=True,
            type="primary",
            disabled=not ok,
        )

        if load_btn:
            chunks_key = "|".join(str(p) for p in INDEX_CONFIGS[index_name]["chunks"])
            prog_ph = st.empty()
            stages = [
                "Loading tokenizer…",
                "Loading FAISS index…",
                "Loading cross-encoder…",
                "Loading generator…",
                "Warming up…",
            ]
            prog_ph.markdown(_loading_stages_html(stages, active=0), unsafe_allow_html=True)
            try:
                pipeline = _load_pipeline_cached(index_name, chunks_key)
                pipeline.reranking_top_k = rerank_k
                pipeline.retrieval_top_k = top_k
                st.session_state["pipeline"] = pipeline
                st.session_state["pipeline_config"] = {
                    "index_name": index_name,
                    "top_k": top_k,
                    "rerank_k": rerank_k,
                    "debug_mode": debug_mode,
                    "return_provenance": return_provenance,
                    "citation_mode": citation_mode,
                }
                prog_ph.empty()
            except Exception as exc:
                prog_ph.empty()
                st.error(f"Load failed: {exc}")

        # Status block
        if st.session_state["pipeline"] is not None:
            cfg = st.session_state["pipeline_config"]
            short = INDEX_CONFIGS[cfg["index_name"]]["short"]
            st.markdown(f"""
<div class='pipeline-status-ready'>
  <span class='sb-status-dot-green'></span>
  <div>
    <div class='ps-label'>Pipeline ready</div>
    <div class='ps-sub'>{short} · top_k={cfg['top_k']}</div>
  </div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
<div class='pipeline-status-none'>
  <span class='sb-status-dot-grey'></span>
  <div class='ps-label'>Not loaded</div>
</div>""", unsafe_allow_html=True)

        if st.session_state.get("use_user_doc") and st.session_state.get("user_chunks"):
            n = len(st.session_state["user_chunks"])
            st.divider()
            st.markdown(
                f"<div style='font-size:0.78rem;color:#1db954;font-weight:600;"
                f"padding:0.4rem 0'>📎 User doc active · {n} chunks</div>",
                unsafe_allow_html=True,
            )

    return index_name, top_k, rerank_k, debug_mode, return_provenance, citation_mode


def _loading_stages_html(stages: list, active: int) -> str:
    rows = []
    for i, label in enumerate(stages):
        if i < active:
            dot = "stage-dot-done"
            txt = "stage-text-done"
        elif i == active:
            dot = "stage-dot-active"
            txt = "stage-text-active"
        else:
            dot = "stage-dot-pending"
            txt = "stage-text-pending"
        rows.append(
            f"<div class='stage-row'>"
            f"<span class='{dot}'></span>"
            f"<span class='{txt}'>{label}</span>"
            f"</div>"
        )
    return "".join(rows)


# ── Tab 1: Ask LEXAR ───────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    "Punishment for murder — IPC §302",
    "What is culpable homicide?",
    "Bail conditions under CrPC",
    "Confession to police — admissible?",
    "What is dacoity and its punishment?",
    "Death by negligence — IPC §304A",
]

def render_qa_tab(top_k, rerank_k, debug_mode, return_provenance, citation_mode):
    # ── Hero banner
    st.markdown("""
<div class='hero-banner'>
  <div class='hero-eyebrow'>Indian Legal AI</div>
  <div class='hero-title'>Ask a Legal Question</div>
  <div class='hero-sub'>Evidence-grounded answers from IPC, CrPC, IEA &amp; more — zero hallucination.</div>
</div>""", unsafe_allow_html=True)

    # User doc banner
    if st.session_state.get("use_user_doc") and st.session_state.get("user_chunks"):
        n = len(st.session_state["user_chunks"])
        st.markdown(
            f"<div style='background:#0d1f12;border:1px solid #1a3a21;border-radius:8px;"
            f"padding:0.6rem 1rem;margin-bottom:1rem;font-size:0.85rem;color:#1db954;font-weight:600;'>"
            f"📎 User document active &nbsp;·&nbsp; <span style='color:#a7a7a7;font-weight:400'>{n} chunks will be searched</span></div>",
            unsafe_allow_html=True,
        )

    # ── Input area
    query = st.text_area(
        "query",
        placeholder="e.g.  What is the punishment for murder under IPC Section 302?",
        height=110,
        key="query_input",
        label_visibility="collapsed",
    )

    col_ask, col_clear, _sp = st.columns([1.8, 0.9, 5])
    with col_ask:
        ask_btn = st.button("⚖️  Ask LEXAR", type="primary", use_container_width=True)
    with col_clear:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state["last_result"] = None
        st.rerun()

    # ── Sample queries
    st.markdown("<div class='section-label'>Try a sample query</div>", unsafe_allow_html=True)
    chip_cols = st.columns(3)
    for i, sq in enumerate(SAMPLE_QUERIES):
        if chip_cols[i % 3].button(sq, key=f"sq_{i}", use_container_width=True):
            st.session_state["query_input"] = sq
            st.rerun()

    # ── Query history
    history = st.session_state.get("query_history", [])
    if history:
        st.markdown("<div class='section-label'>Recent</div>", unsafe_allow_html=True)
        h_cols = st.columns(len(history))
        for i, hq in enumerate(history):
            label = hq[:38] + ("…" if len(hq) > 38 else "")
            if h_cols[i].button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state["query_input"] = hq
                st.rerun()

    st.divider()

    # ── Run pipeline
    if ask_btn:
        _run_pipeline(query, top_k, rerank_k, debug_mode, return_provenance, citation_mode)

    # ── Render result
    if st.session_state.get("last_result"):
        _render_result(st.session_state["last_result"], debug_mode)


def _run_pipeline(query, top_k, rerank_k, debug_mode, return_provenance, citation_mode):
    """Run the pipeline with staged loading indicators."""
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.warning("⚠️ Load the pipeline first — click **▶ Load Pipeline** in the sidebar.")
        return
    if not query or not query.strip():
        st.warning("Enter a question before hitting Ask LEXAR.")
        return

    if st.session_state.get("use_user_doc") and st.session_state.get("user_chunks"):
        try:
            from lexar.retrieval.user_retriever import UserRetriever
            pipeline.retriever.user = UserRetriever(st.session_state["user_chunks"])
        except Exception as exc:
            st.warning(f"Could not attach user doc retriever: {exc}")

    pipeline.retrieval_top_k = top_k
    pipeline.reranking_top_k = rerank_k

    STAGES = [
        "Routing query to legal indices",
        "Retrieving relevant provisions",
        "Re-ranking evidence",
        "Generating grounded answer",
    ]

    stage_ph = st.empty()
    prog_ph  = st.progress(0)

    for i, label in enumerate(STAGES):
        stage_ph.markdown(_loading_stages_html(STAGES, active=i), unsafe_allow_html=True)
        prog_ph.progress(int((i / len(STAGES)) * 90))
        time.sleep(0.06)

    try:
        result = pipeline.answer(
            query=query.strip(),
            has_user_docs=st.session_state.get("use_user_doc", False),
            top_k=top_k,
            return_provenance=return_provenance,
            debug_mode=debug_mode,
        )
        try:
            rc = pipeline._retrieve(query.strip(), st.session_state.get("use_user_doc", False), top_k)
            ev, _ = pipeline._rerank_and_score(query.strip(), rc, rerank_k)
            result["_evidence"] = ev
        except Exception:
            result["_evidence"] = []
    except Exception as exc:
        stage_ph.empty()
        prog_ph.empty()
        st.error(f"Pipeline error: {exc}")
        return

    prog_ph.progress(100)
    time.sleep(0.15)
    stage_ph.empty()
    prog_ph.empty()

    result["_citation_mode"] = citation_mode
    result["_query"] = query.strip()

    history = st.session_state.get("query_history", [])
    if query.strip() not in history:
        history.insert(0, query.strip())
    st.session_state["query_history"] = history[:5]
    st.session_state["last_result"] = result
    st.rerun()


def _render_result(result: dict, debug_mode: bool):
    import plotly.graph_objects as go

    status     = result.get("status", "unknown")
    confidence = result.get("confidence", 0.0)
    ev_count   = result.get("evidence_count", 0)

    # ── Top metrics row
    st.markdown("<div class='section-label'>Result</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    conf_color = "#1db954" if confidence >= 0.6 else ("#f59e0b" if confidence >= 0.3 else "#ef4444")
    status_map = {
        "success":               ("✅", "Grounded"),
        "insufficient_evidence": ("⚠️", "Low Evidence"),
        "no_evidence":           ("❌", "No Evidence"),
        "generation_error":      ("🔥", "Error"),
    }
    s_icon, s_label = status_map.get(status, ("❓", status))

    c1.metric("Confidence",  f"{confidence:.0%}")
    c2.metric("Evidence",    f"{ev_count} chunks")
    c3.metric("Status",      f"{s_icon} {s_label}")
    c4.metric("Query",       f"{len(result.get('_query',''))} chars")

    st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)

    # ── Answer card
    if status == "success":
        answer       = result.get("answer", "")
        evidence_ids = result.get("evidence_ids", [])

        # Build citation HTML
        cit_html = ""
        if evidence_ids:
            cit_html = "<div class='citation-row'><span class='cit-label'>Sources</span>"
            cit_html += f"<span class='cit-primary'>{evidence_ids[0]}</span>"
            for sid in evidence_ids[1:]:
                cit_html += f"<span class='cit-secondary'>{sid}</span>"
            cit_html += "</div>"

        st.markdown(f"""
<div class='answer-card answer-card-success'>
  <div class='answer-label'>LEXAR Answer</div>
  <div class='answer-text'>{answer}</div>
  {cit_html}
</div>""", unsafe_allow_html=True)

        # Evidence details
        with st.expander("📋  Evidence Chunks", expanded=False):
            raw_ev = result.get("_evidence", [])
            if raw_ev:
                import pandas as pd
                rows = []
                for chunk in raw_ev:
                    meta = chunk.get("metadata", {})
                    rows.append({
                        "Section":      meta.get("section", chunk.get("chunk_id", "—")),
                        "Statute":      meta.get("statute", meta.get("source", "—")),
                        "Score":        f"{chunk.get('rerank_score', chunk.get('score', 0.0)):.3f}",
                        "Text Preview": chunk.get("text", "")[:160] + "…",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.caption("Enable Debug Mode for full chunk details.")

        # Provenance
        if result.get("provenance"):
            with st.expander("🔍  Token Provenance", expanded=False):
                st.json(result["provenance"])

        # Debug attention chart
        if debug_mode and result.get("debug"):
            with st.expander("🔬  Attention Distribution", expanded=False):
                debug = result["debug"]
                if isinstance(debug, dict):
                    attn = debug.get("chunk_attention_mass") or debug.get("attention_per_chunk")
                    if attn:
                        labels = list(attn.keys()) if isinstance(attn, dict) else [f"Chunk {i}" for i in range(len(attn))]
                        values = list(attn.values()) if isinstance(attn, dict) else list(attn)
                        fig = go.Figure(go.Bar(
                            x=labels, y=values,
                            marker_color="#1db954",
                            marker_line_color="#0d2818",
                            marker_line_width=1.5,
                        ))
                        fig.update_layout(
                            title=None,
                            xaxis_title="Chunk", yaxis_title="Attention Mass",
                            yaxis_range=[0, 1],
                            height=280,
                            plot_bgcolor="#111",
                            paper_bgcolor="#111",
                            font=dict(color="#b3b3b3", size=11),
                            margin=dict(l=0, r=0, t=10, b=0),
                        )
                        fig.update_xaxes(gridcolor="#1e1e1e")
                        fig.update_yaxes(gridcolor="#1e1e1e")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.json(debug)

    elif status == "insufficient_evidence":
        max_attn = result.get("max_attention", 0.0)
        required = result.get("required_threshold", 0.5)
        deficit  = result.get("deficit", 0.0)

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=max_attn,
            number={"suffix": "", "font": {"color": "#f59e0b", "size": 28}},
            title={"text": "Max Attention on Evidence", "font": {"color": "#a7a7a7", "size": 13}},
            gauge={
                "axis": {"range": [0, 1], "tickcolor": "#333", "tickwidth": 1, "tickcolor": "#333"},
                "bar": {"color": "#f59e0b", "thickness": 0.3},
                "bgcolor": "#181818",
                "bordercolor": "#282828",
                "steps": [
                    {"range": [0, required], "color": "#1a1208"},
                    {"range": [required, 1],  "color": "#0d1f12"},
                ],
                "threshold": {
                    "line": {"color": "#1db954", "width": 3},
                    "thickness": 0.75,
                    "value": required,
                },
            },
        ))
        fig.update_layout(
            height=230,
            paper_bgcolor="#111",
            font=dict(color="#b3b3b3"),
            margin=dict(l=20, r=20, t=20, b=0),
        )

        reason      = result.get("reason", "")
        suggestions = result.get("suggestions", [])
        sugg_html   = "".join(f"<li style='margin-bottom:4px;color:#a7a7a7'>{s}</li>" for s in suggestions)

        st.markdown(f"""
<div class='answer-card answer-card-warn'>
  <div class='answer-label answer-label-warn'>⚠️ Insufficient Evidence</div>
  <div class='answer-text' style='font-size:0.95rem'>
    LEXAR cannot produce a grounded answer — the retrieved evidence did not reach
    the required attention threshold (<strong style='color:#fff'>{required:.0%}</strong>).
    Current max: <strong style='color:#f59e0b'>{max_attn:.0%}</strong> &nbsp;·&nbsp;
    Deficit: <strong style='color:#ef4444'>{deficit:.0%}</strong>
  </div>
  {f"<div style='margin-top:0.8rem;font-size:0.88rem;color:#b3b3b3'>{reason}</div>" if reason else ""}
  {f"<ul style='margin:1rem 0 0 1rem;padding:0;font-size:0.88rem'>{sugg_html}</ul>" if suggestions else ""}
</div>""", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=False)

        ev_summary = result.get("evidence_summary", "")
        if ev_summary:
            with st.expander("📋  Evidence Summary"):
                st.markdown(ev_summary)

    elif status == "no_evidence":
        st.markdown("""
<div class='answer-card answer-card-error'>
  <div class='answer-label answer-label-error'>❌ No Evidence Found</div>
  <div class='answer-text' style='font-size:0.95rem'>
    No relevant legal material was found for your query.
  </div>
  <ul style='margin:1rem 0 0 1.5rem;padding:0;font-size:0.88rem;color:#a7a7a7'>
    <li>Check spelling of legal terms</li>
    <li>Try broader phrasing — e.g. "punishment for theft" not "section 379 penalty"</li>
    <li>Switch to a larger index (LEXAR Medium) in the sidebar</li>
  </ul>
</div>""", unsafe_allow_html=True)

    else:
        answer = result.get("answer", "An unknown error occurred.")
        st.markdown(f"""
<div class='answer-card answer-card-error'>
  <div class='answer-label answer-label-error'>🔥 Generation Error</div>
  <div class='answer-text' style='font-size:0.9rem'>{answer}</div>
</div>""", unsafe_allow_html=True)


# ── Tab 2: Upload & Ingest ─────────────────────────────────────────────────
def render_upload_tab():
    st.markdown("""
<div class='hero-banner'>
  <div class='hero-eyebrow'>Document Intelligence</div>
  <div class='hero-title'>Upload a Legal PDF</div>
  <div class='hero-sub'>Ingest any statute, judgment or contract and query it alongside the knowledge base.</div>
</div>""", unsafe_allow_html=True)

    col_up, col_info = st.columns([3, 2])

    with col_up:
        uploaded_file = st.file_uploader(
            "Drop your PDF here",
            type=["pdf"],
            help="Max 10 MB",
            accept_multiple_files=False,
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            size_mb = uploaded_file.size / (1024 * 1024)
            st.markdown(
                f"<div style='font-size:0.85rem;color:#a7a7a7;margin:0.5rem 0'>"
                f"<strong style='color:#fff'>{uploaded_file.name}</strong> &nbsp;·&nbsp; {size_mb:.2f} MB</div>",
                unsafe_allow_html=True,
            )
            if size_mb > 10:
                st.error("File exceeds 10 MB limit.")
                return
            if st.button("⬆️  Ingest Document", type="primary"):
                _ingest_pdf(uploaded_file)

    with col_info:
        st.markdown("""
<div style='background:#161616;border:1px solid #242424;border-radius:12px;padding:1.5rem'>
  <div style='font-size:0.67rem;letter-spacing:0.14em;text-transform:uppercase;color:#535353;font-weight:700;margin-bottom:1rem'>How it works</div>
  <div style='display:flex;gap:10px;margin-bottom:0.85rem'>
    <span style='color:#1db954;font-size:1.1rem'>1</span>
    <div>
      <div style='font-size:0.85rem;color:#fff;font-weight:600'>Extract</div>
      <div style='font-size:0.78rem;color:#6b6b6b'>Text is extracted from the PDF using pdfplumber</div>
    </div>
  </div>
  <div style='display:flex;gap:10px;margin-bottom:0.85rem'>
    <span style='color:#1db954;font-size:1.1rem'>2</span>
    <div>
      <div style='font-size:0.85rem;color:#fff;font-weight:600'>Chunk</div>
      <div style='font-size:0.78rem;color:#6b6b6b'>Document split into 300-word overlapping chunks</div>
    </div>
  </div>
  <div style='display:flex;gap:10px'>
    <span style='color:#1db954;font-size:1.1rem'>3</span>
    <div>
      <div style='font-size:0.85rem;color:#fff;font-weight:600'>Index</div>
      <div style='font-size:0.78rem;color:#6b6b6b'>FAISS in-memory index built — ready to search</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Use-in-Q&A toggle
    if st.session_state.get("user_chunks"):
        n = len(st.session_state["user_chunks"])
        st.divider()
        use = st.checkbox(
            f"Use this document in Q&A queries ({n} chunks)",
            value=st.session_state.get("use_user_doc", False),
            key="use_user_doc_toggle",
        )
        st.session_state["use_user_doc"] = use
        if use:
            st.markdown(
                "<div style='background:#0d1f12;border:1px solid #1a3a21;border-radius:8px;"
                "padding:0.65rem 1rem;font-size:0.85rem;color:#1db954;font-weight:600;margin-top:0.5rem'>"
                "✅ Document will be searched alongside the knowledge base</div>",
                unsafe_allow_html=True,
            )


def _ingest_pdf(uploaded_file):
    try:
        import pdfplumber
    except ImportError:
        st.error("pdfplumber is not installed. Run: pip install pdfplumber")
        return
    try:
        from lexar.chunking.generic_chunker import chunk_generic_text
    except ImportError:
        st.error("Could not import lexar chunking module.")
        return

    with st.status("Processing document…", expanded=True) as status_widget:
        st.write("📖 Extracting text from PDF…")
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
                tmp_pdf.write(uploaded_file.read())
                tmp_path = tmp_pdf.name
            pages_text = []
            with pdfplumber.open(tmp_path) as pdf:
                n_pages = len(pdf.pages)
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
            os.unlink(tmp_path)
            full_text = "\n\n".join(pages_text)
            if not full_text.strip():
                st.error("Could not extract text (possibly a scanned/image PDF).")
                return
            st.write(f"✅ {len(full_text):,} characters from {n_pages} pages")
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            return

        st.write("✂️ Chunking document…")
        try:
            chunks = chunk_generic_text(full_text)
            for i, chunk in enumerate(chunks):
                chunk["chunk_id"] = f"user_{i}"
                chunk.setdefault("metadata", {})
                chunk["metadata"]["source"] = "UserPDF"
                chunk["metadata"]["document"] = uploaded_file.name
        except Exception as exc:
            st.error(f"Chunking failed: {exc}")
            return
        st.write(f"✅ {len(chunks)} chunks created")
        status_widget.update(label="Done!", state="complete", expanded=False)

    st.markdown(f"""
<div class='upload-success-card'>
  <div class='upload-success-title'>✅ Document Ingested</div>
  <div class='upload-success-row'>File &nbsp;<span class='upload-success-val'>{uploaded_file.name}</span></div>
  <div class='upload-success-row'>Pages &nbsp;<span class='upload-success-val'>{n_pages}</span></div>
  <div class='upload-success-row'>Text length &nbsp;<span class='upload-success-val'>{len(full_text):,} chars</span></div>
  <div class='upload-success-row'>Chunks &nbsp;<span class='upload-success-val'>{len(chunks)}</span></div>
</div>""", unsafe_allow_html=True)

    st.session_state["user_chunks"] = chunks
    st.session_state["use_user_doc"] = True
    st.rerun()


# ── Tab 3: Evaluation Dashboard ────────────────────────────────────────────
def render_eval_tab():
    st.markdown("""
<div class='hero-banner'>
  <div class='hero-eyebrow'>Benchmarking</div>
  <div class='hero-title'>Evaluation Dashboard</div>
  <div class='hero-sub'>Precision@3, Precision@5, Recall@5 and MRR against 8 gold IPC queries.</div>
</div>""", unsafe_allow_html=True)

    missing_files = []
    if not GOLD_QUERIES_PATH.exists():   missing_files.append(str(GOLD_QUERIES_PATH))
    if not EVAL_CHUNKS_PATH.exists():    missing_files.append(str(EVAL_CHUNKS_PATH))
    if not EVAL_INDEX_PATH.exists():     missing_files.append(str(EVAL_INDEX_PATH))
    if missing_files:
        for mf in missing_files:
            st.error(f"Missing: `{Path(mf).name}`")
        return

    with open(GOLD_QUERIES_PATH) as f:
        gold_queries = json.load(f)

    col_info, col_btn = st.columns([3, 1])
    col_info.markdown(
        f"<div style='font-size:0.9rem;color:#a7a7a7'>"
        f"<strong style='color:#fff'>{len(gold_queries)} gold queries</strong> loaded from "
        f"<code style='color:#1db954;font-size:0.8rem'>evaluation/gold_queries.json</code></div>",
        unsafe_allow_html=True,
    )
    with col_btn:
        run_btn = st.button("▶  Run Evaluation", type="primary", use_container_width=True)

    if run_btn:
        _run_evaluation(gold_queries)

    if st.session_state.get("eval_results"):
        _render_eval_results(st.session_state["eval_results"])


def _run_evaluation(gold_queries: list):
    try:
        from lexar.retrieval.retriever import DenseRetriever
        from lexar.reranking.cross_encoder import LegalCrossEncoderReranker
    except ImportError:
        st.error("Could not import lexar modules. Check your installation.")
        return

    progress_area = st.empty()
    progress_bar = st.progress(0)
    log_area = st.empty()

    with progress_area.container():
        st.markdown("**Loading retriever and reranker…**")

    try:
        with open(EVAL_CHUNKS_PATH) as f:
            chunks = json.load(f)
        retriever = DenseRetriever(chunks, index_path=str(EVAL_INDEX_PATH))
        reranker = LegalCrossEncoderReranker()
    except Exception as exc:
        progress_area.empty()
        progress_bar.empty()
        st.error(f"Failed to load evaluation components: {exc}")
        return

    results_per_query = []
    precision_at_3_list = []
    precision_at_5_list = []
    recall_at_5_list = []
    mrr_list = []

    n = len(gold_queries)
    for i, item in enumerate(gold_queries):
        query = item["query"]
        relevant = set(item["relevant_sections"])

        progress_bar.progress((i + 1) / n)
        log_area.markdown(f"<span class='stage-label'>Query {i+1}/{n}: {query[:60]}…</span>", unsafe_allow_html=True)

        try:
            retrieved = retriever.retrieve(query, top_k=10)
            reranked = reranker.rerank(query, retrieved, top_k=5)
        except Exception as exc:
            results_per_query.append({
                "Query": query,
                "Relevant": ", ".join(sorted(relevant)),
                "Retrieved@5": "ERROR",
                "P@3": 0.0,
                "P@5": 0.0,
                "Recall@5": 0.0,
                "RR": 0.0,
            })
            precision_at_3_list.append(0.0)
            precision_at_5_list.append(0.0)
            recall_at_5_list.append(0.0)
            mrr_list.append(0.0)
            continue

        def extract_sections(chunks_list):
            secs = []
            for c in chunks_list:
                sec = c.get("metadata", {}).get("section")
                if sec:
                    secs.append(sec)
            return secs

        retrieved_5 = extract_sections(reranked[:5])
        retrieved_3 = extract_sections(reranked[:3])

        hits5 = len(relevant.intersection(set(retrieved_5)))
        hits3 = len(relevant.intersection(set(retrieved_3)))
        p5 = hits5 / 5
        p3 = hits3 / 3
        rec5 = hits5 / len(relevant) if relevant else 0.0

        rr = 0.0
        for rank, sec in enumerate(retrieved_5, start=1):
            if sec in relevant:
                rr = 1.0 / rank
                break

        precision_at_3_list.append(p3)
        precision_at_5_list.append(p5)
        recall_at_5_list.append(rec5)
        mrr_list.append(rr)

        results_per_query.append(
            {
                "Query": query[:55] + ("…" if len(query) > 55 else ""),
                "Relevant §": ", ".join(sorted(relevant)),
                "Retrieved §@5": ", ".join(retrieved_5),
                "P@3": round(p3, 3),
                "P@5": round(p5, 3),
                "Recall@5": round(rec5, 3),
                "RR": round(rr, 3),
            }
        )

    progress_bar.empty()
    log_area.empty()
    progress_area.empty()

    eval_results = {
        "precision_at_3": sum(precision_at_3_list) / n,
        "precision_at_5": sum(precision_at_5_list) / n,
        "recall_at_5": sum(recall_at_5_list) / n,
        "mrr": sum(mrr_list) / n,
        "per_query": results_per_query,
        "n": n,
    }
    st.session_state["eval_results"] = eval_results
    st.balloons()
    st.rerun()


def _render_eval_results(results: dict):
    import plotly.express as px
    import pandas as pd

    st.markdown("<div class='section-label'>Overall Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("P@3",       results["precision_at_3"]),
        ("P@5",       results["precision_at_5"]),
        ("Recall@5",  results["recall_at_5"]),
        ("MRR",       results["mrr"]),
    ]
    for col, (label, val) in zip([c1, c2, c3, c4], metrics):
        col.markdown(
            f"<div class='eval-card'>"
            f"<div class='eval-card-num'>{val:.3f}</div>"
            f"<div class='eval-card-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    df = pd.DataFrame(results["per_query"])
    if not df.empty:
        st.markdown("<div class='section-label' style='margin-top:2rem'>Per-Query Breakdown</div>", unsafe_allow_html=True)
        metric_df = df[["Query", "P@3", "P@5", "Recall@5", "RR"]].melt(
            id_vars="Query", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            metric_df, x="Query", y="Score", color="Metric", barmode="group", height=360,
            color_discrete_sequence=["#1db954", "#3b82f6", "#f59e0b", "#a78bfa"],
        )
        fig.update_layout(
            xaxis_tickangle=-25,
            legend_title="",
            plot_bgcolor="#111",
            paper_bgcolor="#111",
            font=dict(color="#b3b3b3", size=11),
            margin=dict(l=0, r=0, t=10, b=0),
            bargap=0.2,
        )
        fig.update_xaxes(gridcolor="#1a1a1a", tickfont=dict(size=10))
        fig.update_yaxes(gridcolor="#1a1a1a", range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-label'>Detailed Table</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── Tab 4: About ───────────────────────────────────────────────────────────
def render_about_tab():
    st.markdown("""
<div class='hero-banner'>
  <div class='hero-eyebrow'>About LEXAR</div>
  <div class='hero-title'>Legal Explainable<br>Augmented Reasoner</div>
  <div class='hero-sub'>v1.1.1 &nbsp;·&nbsp; by Garv Behl &nbsp;·&nbsp; Zero hallucination legal AI for Indian law</div>
</div>""", unsafe_allow_html=True)

    # ── Model cards grid
    st.markdown("<div class='section-label'>Models</div>", unsafe_allow_html=True)
    model_cards = [
        ("🔍", "Query Encoder", "lexar_query_encoder_v1 (fine-tuned)\nFallback: all-MiniLM-L6-v2"),
        ("📄", "Document Encoder", "sentence-transformers/all-MiniLM-L6-v2\nBuilds FAISS indexes"),
        ("📊", "Cross-Encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2\nScores (query, chunk) pairs"),
        ("🤖", "Generator", "google/flan-t5-base\nSeq2seq · T=0 · hard attention mask"),
        ("🔐", "Evidence Gate", "Threshold: 0.50 max attention mass\nRejects under-grounded answers"),
        ("📐", "FAISS Index", "IndexFlatIP (inner product / cosine)\nDeterministic, no quantization"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(model_cards):
        with cols[i % 3]:
            st.markdown(
                f"<div class='model-card'>"
                f"<div class='model-card-icon'>{icon}</div>"
                f"<div class='model-card-title'>{title}</div>"
                f"<div class='model-card-desc'>{desc.replace(chr(10), '<br>')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Pipeline flow
    st.markdown("<div class='section-label'>Pipeline Architecture</div>", unsafe_allow_html=True)
    steps = [
        ("01", "Query Router",    "Keyword matching routes to IPC / Judgment / User indexes"),
        ("02", "Dense Retrieval", "FAISS IndexFlatIP + LegalEmbedder → top-K chunks"),
        ("03", "Re-ranking",      "Cross-encoder rescores every (query, chunk) pair"),
        ("04", "Evidence Gate",   "Rejects if max attention < 0.5 — no answer silently degraded"),
        ("05", "Generation",      "flan-t5-base with hard binary attention masking on evidence"),
        ("06", "Citation",        "Token-level provenance → inline or footnote citations"),
    ]
    for num, title, desc in steps:
        st.markdown(
            f"<div style='display:flex;gap:16px;align-items:flex-start;"
            f"padding:0.9rem 0;border-bottom:1px solid #1a1a1a'>"
            f"<span style='font-size:0.7rem;font-weight:800;color:#1db954;"
            f"letter-spacing:0.1em;padding-top:2px;width:20px;flex-shrink:0'>{num}</span>"
            f"<div>"
            f"<div style='font-size:0.88rem;font-weight:700;color:#fff;margin-bottom:2px'>{title}</div>"
            f"<div style='font-size:0.8rem;color:#6b6b6b'>{desc}</div>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Available indexes
    st.markdown("<div class='section-label' style='margin-top:2rem'>Available Indexes</div>", unsafe_allow_html=True)
    for name, cfg in INDEX_CONFIGS.items():
        ok, missing = _files_exist(name)
        dot = "#1db954" if ok else "#ef4444"
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:10px;padding:0.65rem 0;"
            f"border-bottom:1px solid #1a1a1a'>"
            f"<span style='width:8px;height:8px;border-radius:50%;background:{dot};"
            f"flex-shrink:0;display:inline-block'></span>"
            f"<div>"
            f"<span class='index-badge'>{cfg['short']}</span> "
            f"<span style='font-size:0.83rem;color:#a7a7a7'>{cfg['description']}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    # ── Quick start
    st.markdown("<div class='section-label' style='margin-top:2rem'>Quick Start</div>", unsafe_allow_html=True)
    quick_steps = [
        ("Select Index",    "Choose an index in the sidebar — LEXAR Medium is recommended"),
        ("Load Pipeline",   "Click ▶ Load Pipeline (first run downloads models, ~30–60 s)"),
        ("Ask a Question",  "Type in the Ask LEXAR tab — hit the green button"),
        ("Upload a Doc",    "Upload tab → ingest PDF → enable Use in Q&A"),
        ("Benchmark",       "Evaluation tab → ▶ Run Evaluation → see P@3/P@5/MRR"),
    ]
    for i, (t, d) in enumerate(quick_steps, 1):
        st.markdown(
            f"<div style='display:flex;gap:14px;align-items:flex-start;padding:0.75rem 0;"
            f"border-bottom:1px solid #1a1a1a'>"
            f"<span style='background:#1db954;color:#000;border-radius:50%;width:22px;height:22px;"
            f"display:flex;align-items:center;justify-content:center;font-size:0.7rem;"
            f"font-weight:800;flex-shrink:0;margin-top:1px'>{i}</span>"
            f"<div><div style='font-size:0.88rem;font-weight:700;color:#fff'>{t}</div>"
            f"<div style='font-size:0.8rem;color:#6b6b6b;margin-top:2px'>{d}</div></div></div>",
            unsafe_allow_html=True,
        )


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    index_name, top_k, rerank_k, debug_mode, return_provenance, citation_mode = render_sidebar()

    tab_qa, tab_upload, tab_eval, tab_about = st.tabs([
        "⚖️  Ask LEXAR",
        "📎  Upload",
        "📊  Evaluation",
        "ℹ️  About",
    ])

    with tab_qa:
        render_qa_tab(top_k, rerank_k, debug_mode, return_provenance, citation_mode)
    with tab_upload:
        render_upload_tab()
    with tab_eval:
        render_eval_tab()
    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
