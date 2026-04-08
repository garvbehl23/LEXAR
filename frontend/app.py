"""
LEXAR Legal AI — Streamlit Frontend
====================================
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
from typing import Optional

import streamlit as st

# ── Project root on sys.path so we can import lexar / backend ──────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Page config (must be first Streamlit call) ─────────────────────────────
st.set_page_config(
    page_title="LEXAR Legal AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/garvbehl",
        "About": "LEXAR v1.1.1 — Legal Explainable Augmented Reasoner by Garv Behl",
    },
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ─── Header ─────────────────────────────────────────────── */
.lexar-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    color: white;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.lexar-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
.lexar-header p  { margin: 0; opacity: 0.75; font-size: 0.9rem; }

/* ─── Answer box ─────────────────────────────────────────── */
.answer-box {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    font-size: 1rem;
    line-height: 1.7;
    color: #14532d;
}
.answer-box-warn {
    background: #fffbeb;
    border-left: 4px solid #d97706;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    color: #78350f;
}
.answer-box-error {
    background: #fef2f2;
    border-left: 4px solid #dc2626;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin: 1rem 0;
    color: #7f1d1d;
}

/* ─── Citation chips ─────────────────────────────────────── */
.citation-primary {
    display: inline-block;
    background: #1e3a5f;
    color: white;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
}
.citation-supporting {
    display: inline-block;
    background: #64748b;
    color: white;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    margin-right: 4px;
}

/* ─── Stage indicator ────────────────────────────────────── */
.stage-label {
    font-size: 0.82rem;
    color: #6b7280;
    font-style: italic;
}

/* ─── Status badge ───────────────────────────────────────── */
.badge-success { color: #16a34a; font-weight: 700; }
.badge-warn    { color: #d97706; font-weight: 700; }
.badge-error   { color: #dc2626; font-weight: 700; }

/* ─── Card ───────────────────────────────────────────────── */
.info-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.info-card h4 { margin: 0 0 0.25rem 0; font-size: 1rem; color: #1e293b; }
.info-card p  { margin: 0; font-size: 0.85rem; color: #64748b; }

/* ─── Sidebar ────────────────────────────────────────────── */
.sidebar-section { margin-bottom: 1rem; }
.pipeline-ready  { color: #16a34a; font-weight: 600; font-size: 0.9rem; }
.pipeline-none   { color: #6b7280; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

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
        st.markdown("## ⚖️ LEXAR Legal AI")
        st.markdown("*Legal Explainable Augmented Reasoner*")
        st.divider()

        # ── Knowledge Base
        st.markdown("### 📚 Knowledge Base")
        index_name = st.selectbox(
            "Select Index",
            options=list(INDEX_CONFIGS.keys()),
            index=0,
            help="The FAISS index + corpus to search for evidence.",
        )
        st.caption(INDEX_CONFIGS[index_name]["description"])

        ok, missing = _files_exist(index_name)
        if not ok:
            st.error(f"Missing files:\n" + "\n".join(f"• `{m}`" for m in missing))

        st.divider()

        # ── Advanced Settings
        with st.expander("⚙️ Advanced Settings", expanded=False):
            top_k = st.slider(
                "Top-K Retrieval",
                min_value=3,
                max_value=20,
                value=10,
                help="Number of chunks fetched from FAISS before reranking.",
            )
            rerank_k = st.slider(
                "Reranking Top-K",
                min_value=1,
                max_value=5,
                value=3,
                help="How many top-reranked chunks are passed to the generator.",
            )
            citation_mode = st.radio(
                "Citation Mode",
                options=["inline", "footnote"],
                horizontal=True,
                help="How citations are appended to the answer.",
            )
            debug_mode = st.checkbox(
                "Debug Mode",
                value=False,
                help="Return attention weight distribution per evidence chunk.",
            )
            return_provenance = st.checkbox(
                "Return Provenance",
                value=False,
                help="Include token-level provenance in the result.",
            )

        st.divider()

        # ── Load Pipeline
        st.markdown("### 🚀 Pipeline")
        load_btn = st.button(
            "▶ Load / Reload Pipeline",
            use_container_width=True,
            type="primary",
            disabled=not ok,
        )

        if load_btn:
            chunks_key = "|".join(str(p) for p in INDEX_CONFIGS[index_name]["chunks"])
            with st.spinner("Loading models… (first run may take ~30–60 s)"):
                progress = st.empty()
                progress.markdown(
                    "<span class='stage-label'>⏳ Loading retriever…</span>",
                    unsafe_allow_html=True,
                )
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
                    progress.empty()
                    st.success("Pipeline ready!")
                except Exception as exc:
                    progress.empty()
                    st.error(f"Failed to load pipeline: {exc}")

        # ── Status badge
        if st.session_state["pipeline"] is not None:
            cfg = st.session_state["pipeline_config"]
            st.markdown(
                f"<span class='pipeline-ready'>● Pipeline ready</span>"
                f"<br><span style='font-size:0.78rem;color:#6b7280'>"
                f"{INDEX_CONFIGS[cfg['index_name']]['short']} · top_k={cfg['top_k']}"
                f"</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<span class='pipeline-none'>○ Pipeline not loaded</span>",
                unsafe_allow_html=True,
            )

        st.divider()
        if st.session_state.get("use_user_doc") and st.session_state.get("user_chunks"):
            n = len(st.session_state["user_chunks"])
            st.info(f"📎 User document active ({n} chunks)")

    return index_name, top_k, rerank_k, debug_mode, return_provenance, citation_mode


# ── Tab 1: Ask LEXAR ───────────────────────────────────────────────────────
def render_qa_tab(top_k, rerank_k, debug_mode, return_provenance, citation_mode):
    st.markdown(
        """
<div class='lexar-header'>
  <span style='font-size:2.5rem'>⚖️</span>
  <div>
    <h1>Ask LEXAR</h1>
    <p>Evidence-constrained legal question answering for Indian law</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── User doc banner
    if st.session_state.get("use_user_doc") and st.session_state.get("user_chunks"):
        st.info("📎 **User document is active.** Your uploaded PDF will also be searched.")

    # ── Query input
    query = st.text_area(
        "Your legal question",
        placeholder="e.g. What is the punishment for murder under IPC?",
        height=100,
        key="query_input",
        label_visibility="collapsed",
    )

    col_ask, col_clear, col_spacer = st.columns([2, 1, 6])
    with col_ask:
        ask_btn = st.button("⚖️  Ask LEXAR", type="primary", use_container_width=True)
    with col_clear:
        clear_btn = st.button("✕  Clear", use_container_width=True)

    if clear_btn:
        st.session_state["last_result"] = None
        st.rerun()

    # ── Query history chips
    history = st.session_state.get("query_history", [])
    if history:
        st.markdown("**Recent queries:**")
        cols = st.columns(len(history))
        for i, hq in enumerate(history):
            if cols[i].button(hq[:40] + ("…" if len(hq) > 40 else ""), key=f"hist_{i}"):
                st.session_state["query_input"] = hq
                st.rerun()

    st.divider()

    # ── Run pipeline on ask
    if ask_btn:
        _run_pipeline(query, top_k, rerank_k, debug_mode, return_provenance, citation_mode)

    # ── Render last result
    if st.session_state.get("last_result"):
        _render_result(st.session_state["last_result"], debug_mode)


def _run_pipeline(query, top_k, rerank_k, debug_mode, return_provenance, citation_mode):
    """Run the pipeline with staged loading indicators."""
    pipeline = st.session_state.get("pipeline")
    if pipeline is None:
        st.warning("⚠️ Load the pipeline first using the sidebar.")
        return
    if not query or not query.strip():
        st.warning("Please enter a question.")
        return

    # Rebuild pipeline UserRetriever if user doc is active
    if st.session_state.get("use_user_doc") and st.session_state.get("user_chunks"):
        try:
            from lexar.retrieval.user_retriever import UserRetriever
            user_ret = UserRetriever(st.session_state["user_chunks"])
            pipeline.retriever.user = user_ret
        except Exception as exc:
            st.warning(f"Could not attach user doc retriever: {exc}")

    pipeline.retrieval_top_k = top_k
    pipeline.reranking_top_k = rerank_k

    # Stage indicators
    status_box = st.empty()
    progress_bar = st.progress(0)

    def _stage(label: str, pct: int):
        status_box.markdown(
            f"<span class='stage-label'>⏳ {label}</span>",
            unsafe_allow_html=True,
        )
        progress_bar.progress(pct)
        time.sleep(0.05)  # tiny pause so user sees the transition

    _stage("Stage 1/4 — Routing query to relevant legal indices…", 5)
    _stage("Stage 2/4 — Retrieving relevant legal provisions…", 25)
    _stage("Stage 3/4 — Re-ranking evidence by relevance…", 55)
    _stage("Stage 4/4 — Generating evidence-constrained answer…", 75)

    try:
        result = pipeline.answer(
            query=query.strip(),
            has_user_docs=st.session_state.get("use_user_doc", False),
            top_k=top_k,
            return_provenance=return_provenance,
            debug_mode=debug_mode,
        )
        # Capture evidence chunks for display (retrieval + reranking are fast)
        try:
            retrieved_chunks = pipeline._retrieve(
                query.strip(), st.session_state.get("use_user_doc", False), top_k
            )
            evidence_chunks, _ = pipeline._rerank_and_score(query.strip(), retrieved_chunks, rerank_k)
            result["_evidence"] = evidence_chunks
        except Exception:
            result["_evidence"] = []
    except Exception as exc:
        status_box.empty()
        progress_bar.empty()
        st.error(f"Pipeline error: {exc}")
        return

    progress_bar.progress(100)
    time.sleep(0.2)
    status_box.empty()
    progress_bar.empty()

    # Attach citation mode (passed to generator, but we can post-process too)
    result["_citation_mode"] = citation_mode
    result["_query"] = query.strip()

    # Update history
    history = st.session_state.get("query_history", [])
    if query.strip() not in history:
        history.insert(0, query.strip())
        st.session_state["query_history"] = history[:5]

    st.session_state["last_result"] = result
    st.rerun()


def _render_result(result: dict, debug_mode: bool):
    import plotly.graph_objects as go

    status = result.get("status", "unknown")

    # ── Metrics row (always shown)
    col1, col2, col3 = st.columns(3)
    confidence = result.get("confidence", 0.0)
    evidence_count = result.get("evidence_count", 0)

    status_label = {
        "success": "✅ Grounded",
        "insufficient_evidence": "⚠️ Insufficient Evidence",
        "no_evidence": "❌ No Evidence",
        "generation_error": "🔥 Generation Error",
    }.get(status, status)

    delta_color = "normal" if status == "success" else "off"
    col1.metric("Confidence", f"{confidence:.0%}", delta_color=delta_color)
    col2.metric("Evidence Chunks", evidence_count)
    col3.metric("Status", status_label)

    st.divider()

    # ── Status-specific rendering
    if status == "success":
        answer = result.get("answer", "")
        st.markdown(
            f"<div class='answer-box'>{answer}</div>",
            unsafe_allow_html=True,
        )

        # Citations
        evidence_ids = result.get("evidence_ids", [])
        if evidence_ids:
            primary = evidence_ids[0] if evidence_ids else None
            supporting = evidence_ids[1:] if len(evidence_ids) > 1 else []

            citation_html = ""
            if primary:
                citation_html += f"<span class='citation-primary'>Primary: {primary}</span>"
            for sid in supporting:
                citation_html += f"<span class='citation-supporting'>{sid}</span>"
            if citation_html:
                st.markdown(citation_html, unsafe_allow_html=True)

        # Evidence details expander
        with st.expander("📋 Evidence Details", expanded=False):
            raw_ev = result.get("_evidence", [])
            if raw_ev:
                import pandas as pd
                rows = []
                for chunk in raw_ev:
                    meta = chunk.get("metadata", {})
                    rows.append(
                        {
                            "Section": meta.get("section", chunk.get("chunk_id", "—")),
                            "Statute": meta.get("statute", meta.get("source", "—")),
                            "Rerank Score": f"{chunk.get('rerank_score', chunk.get('score', 0.0)):.3f}",
                            "Preview": chunk.get("text", "")[:150] + "…",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("Evidence chunk details not returned (enable debug mode for full details).")

        # Provenance expander
        if result.get("provenance"):
            with st.expander("🔍 Token Provenance", expanded=False):
                prov = result["provenance"]
                st.json(prov)

        # Debug expander
        if debug_mode and result.get("debug"):
            with st.expander("🔬 Debug: Attention Distribution", expanded=False):
                debug = result["debug"]
                if isinstance(debug, dict):
                    # Try to plot chunk attention weights
                    attn = debug.get("chunk_attention_mass") or debug.get("attention_per_chunk")
                    if attn:
                        labels = list(attn.keys()) if isinstance(attn, dict) else [f"Chunk {i}" for i in range(len(attn))]
                        values = list(attn.values()) if isinstance(attn, dict) else list(attn)
                        fig = go.Figure(
                            go.Bar(
                                x=labels,
                                y=values,
                                marker_color="#1e3a5f",
                            )
                        )
                        fig.update_layout(
                            title="Attention Mass per Evidence Chunk",
                            xaxis_title="Chunk",
                            yaxis_title="Attention Mass",
                            yaxis_range=[0, 1],
                            height=300,
                            margin=dict(l=0, r=0, t=40, b=0),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.json(debug)

    elif status == "insufficient_evidence":
        st.markdown(
            "<div class='answer-box-warn'>"
            "<strong>⚠️ LEXAR cannot answer this question with sufficient evidence grounding.</strong>"
            "</div>",
            unsafe_allow_html=True,
        )

        max_attn = result.get("max_attention", 0.0)
        required = result.get("required_threshold", 0.5)
        deficit = result.get("deficit", 0.0)

        # Gauge chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=max_attn,
                title={"text": "Max Attention on Evidence"},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1},
                    "bar": {"color": "#d97706"},
                    "steps": [
                        {"range": [0, required], "color": "#fef3c7"},
                        {"range": [required, 1], "color": "#d1fae5"},
                    ],
                    "threshold": {
                        "line": {"color": "#16a34a", "width": 3},
                        "thickness": 0.8,
                        "value": required,
                    },
                },
            )
        )
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=0))
        st.plotly_chart(fig, use_container_width=False)

        st.caption(f"Deficit: **{deficit:.0%}** below the {required:.0%} threshold")

        reason = result.get("reason", "")
        if reason:
            st.info(reason)

        suggestions = result.get("suggestions", [])
        if suggestions:
            st.markdown("**📌 Suggestions:**")
            for s in suggestions:
                st.markdown(f"• {s}")

        ev_summary = result.get("evidence_summary", "")
        if ev_summary:
            with st.expander("📋 Evidence Summary"):
                st.markdown(ev_summary)

    elif status == "no_evidence":
        st.markdown(
            "<div class='answer-box-error'>"
            "<strong>❌ No relevant legal material found for your query.</strong>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Suggestions:**\n"
            "- Check spelling of legal terms\n"
            "- Try broader phrasing (e.g. 'punishment for theft' instead of 'section 379 penalty')\n"
            "- Switch to a larger index (e.g. LEXAR Medium) in the sidebar"
        )

    else:
        # generation_error or unknown
        answer = result.get("answer", "An unknown error occurred.")
        st.markdown(
            f"<div class='answer-box-error'>🔥 {answer}</div>",
            unsafe_allow_html=True,
        )


# ── Tab 2: Upload & Ingest ─────────────────────────────────────────────────
def render_upload_tab():
    st.markdown("## 📎 Upload a Legal PDF")
    st.markdown(
        "Upload any Indian legal document (statute, judgment, contract) "
        "to query it alongside the selected knowledge base."
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Maximum file size: 10 MB",
        accept_multiple_files=False,
    )

    if uploaded_file is not None:
        size_mb = uploaded_file.size / (1024 * 1024)
        col1, col2 = st.columns(2)
        col1.markdown(
            f"<div class='info-card'>"
            f"<h4>📄 {uploaded_file.name}</h4>"
            f"<p>Size: {size_mb:.2f} MB</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

        if size_mb > 10:
            st.error("File exceeds 10 MB limit. Please upload a smaller document.")
            return

        if st.button("⬆️  Ingest Document", type="primary"):
            _ingest_pdf(uploaded_file)

    # ── Use-in-Q&A toggle
    if st.session_state.get("user_chunks"):
        n = len(st.session_state["user_chunks"])
        use = st.checkbox(
            f"☑ Use this document in Q&A queries ({n} chunks)",
            value=st.session_state.get("use_user_doc", False),
            key="use_user_doc_toggle",
        )
        st.session_state["use_user_doc"] = use
        if use:
            st.success("✅ Document will be searched during Q&A.")
        else:
            st.info("○ Document loaded but not active. Check the box to activate.")


def _ingest_pdf(uploaded_file):
    """Process the uploaded PDF: extract text, chunk, store in session state."""
    try:
        import pdfplumber
    except ImportError:
        st.error("pdfplumber is not installed. Run: pip install pdfplumber")
        return

    try:
        from lexar.chunking.generic_chunker import chunk_generic_text
    except ImportError:
        st.error("Could not import lexar chunking module. Check your installation.")
        return

    with st.status("Processing document…", expanded=True) as status_widget:
        st.write("📖 Extracting text from PDF…")
        try:
            # Save to temp file for pdfplumber
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
                st.error("Could not extract text from PDF (possibly scanned/image-only).")
                return

            st.write(f"✅ Extracted {len(full_text):,} characters from {n_pages} pages.")
        except Exception as exc:
            st.error(f"Text extraction failed: {exc}")
            return

        st.write("✂️ Chunking document…")
        try:
            chunks = chunk_generic_text(full_text)
            # Enrich metadata
            for i, chunk in enumerate(chunks):
                chunk["chunk_id"] = f"user_{i}"
                if "metadata" not in chunk:
                    chunk["metadata"] = {}
                chunk["metadata"]["source"] = "UserPDF"
                chunk["metadata"]["document"] = uploaded_file.name
        except Exception as exc:
            st.error(f"Chunking failed: {exc}")
            return

        st.write(f"✅ Created {len(chunks)} chunks.")
        status_widget.update(label="Document processed!", state="complete", expanded=False)

    # Success card
    st.markdown(
        f"<div class='info-card'>"
        f"<h4>✅ Document Ingested</h4>"
        f"<p><strong>File:</strong> {uploaded_file.name}</p>"
        f"<p><strong>Pages:</strong> {n_pages} &nbsp;|&nbsp; "
        f"<strong>Text length:</strong> {len(full_text):,} chars &nbsp;|&nbsp; "
        f"<strong>Chunks:</strong> {len(chunks)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.session_state["user_chunks"] = chunks
    st.session_state["use_user_doc"] = True
    st.rerun()


# ── Tab 3: Evaluation Dashboard ────────────────────────────────────────────
def render_eval_tab():
    st.markdown("## 📊 Evaluation Dashboard")
    st.markdown(
        "Run the gold-query evaluation suite against the IPC corpus. "
        "Measures **Precision@3**, **Precision@5**, **Recall@5**, and **MRR**."
    )

    if not GOLD_QUERIES_PATH.exists():
        st.error(f"Gold queries file not found: `{GOLD_QUERIES_PATH}`")
        return
    if not EVAL_CHUNKS_PATH.exists():
        st.error(f"IPC chunks file not found: `{EVAL_CHUNKS_PATH}`")
        return
    if not EVAL_INDEX_PATH.exists():
        st.error(f"IPC FAISS index not found: `{EVAL_INDEX_PATH}`")
        return

    with open(GOLD_QUERIES_PATH) as f:
        gold_queries = json.load(f)

    st.info(f"**{len(gold_queries)} gold queries** loaded from `evaluation/gold_queries.json`.")

    run_btn = st.button("▶ Run Evaluation", type="primary")

    if run_btn:
        _run_evaluation(gold_queries)

    # Render previous results
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

    st.markdown("### 📈 Overall Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@3", f"{results['precision_at_3']:.3f}")
    c2.metric("Precision@5", f"{results['precision_at_5']:.3f}")
    c3.metric("Recall@5",    f"{results['recall_at_5']:.3f}")
    c4.metric("MRR",         f"{results['mrr']:.3f}")

    # Bar chart per query
    df = pd.DataFrame(results["per_query"])
    if not df.empty:
        st.markdown("### 📊 Per-Query Metrics")
        metric_df = df[["Query", "P@3", "P@5", "Recall@5", "RR"]].melt(
            id_vars="Query", var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            metric_df,
            x="Query",
            y="Score",
            color="Metric",
            barmode="group",
            height=380,
            color_discrete_sequence=["#1e3a5f", "#3b82f6", "#22c55e", "#f59e0b"],
        )
        fig.update_layout(
            xaxis_tickangle=-30,
            legend_title="",
            margin=dict(l=0, r=0, t=20, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Detailed Breakdown")
        st.dataframe(df, use_container_width=True, hide_index=True)


# ── Tab 4: About ───────────────────────────────────────────────────────────
def render_about_tab():
    st.markdown("## ℹ️ About LEXAR")
    st.markdown(
        """
**LEXAR** (Legal Explainable Augmented Reasoner, v1.1.1) is a production-oriented 
Retrieval-Augmented Generation system for **Indian legal question answering** by Garv Behl.

Its defining property is **architectural hallucination prevention**: generation is 
constrained by hard binary attention masking so the decoder can only attend to retrieved 
legal chunks and the query. No answer is produced without evidence.
"""
    )

    st.divider()

    # Pipeline architecture
    st.markdown("### 🏗️ Pipeline Architecture")
    st.markdown(
        """
```
User Query
    │
    ▼
┌─────────────┐
│ Query Router│  Keyword routing → IPC / Judgment / User docs
└──────┬──────┘
       │
    ┌──┴──────────────────────────────────────┐
    │           MultiIndexRetriever            │
    │  IPCRetriever  JudgmentRetriever  User   │
    │  (FAISS IndexFlatIP + LegalEmbedder)     │
    └──────────────────┬──────────────────────┘
                       │  top-K chunks
                       ▼
            ┌───────────────────┐
            │  Cross-Encoder    │  ms-marco-MiniLM-L-6-v2
            │  Reranker         │  → top-K reranked with scores
            └─────────┬─────────┘
                      │  evidence chunks
                      ▼
          ┌─────────────────────────┐
          │  EvidenceSufficiencyGate│  Rejects if max attention < 0.5
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │    LexarGenerator        │  flan-t5-base
          │  Hard attention masking  │  no parametric memory leakage
          │  Token provenance        │  every token → source chunk
          └───────────┬─────────────┘
                      │
                      ▼
          ┌─────────────────────────┐
          │   CitationRenderer       │  inline or footnote
          └─────────────────────────┘
                      │
                      ▼
               Grounded Answer
```
"""
    )

    st.divider()

    st.markdown("### 🤖 Model Cards")
    cols = st.columns(2)
    model_cards = [
        ("🔍 Query Encoder", "lexar_query_encoder_v1 (fine-tuned)\nFallback: all-MiniLM-L6-v2"),
        ("📄 Document Encoder", "sentence-transformers/all-MiniLM-L6-v2\nUsed to build FAISS indexes"),
        ("📊 Cross-Encoder Reranker", "cross-encoder/ms-marco-MiniLM-L-6-v2\nScores (query, chunk) pairs"),
        ("🤖 Generator", "google/flan-t5-base\nSeq2seq, deterministic (T=0), hard attention masking"),
        ("🔐 Evidence Gate", "Threshold: 0.50 max attention mass\nRejects under-grounded answers"),
        ("📐 FAISS Index Type", "IndexFlatIP (inner product / cosine)\nDeterministic, no quantization"),
    ]
    for i, (title, desc) in enumerate(model_cards):
        with cols[i % 2]:
            st.markdown(
                f"<div class='info-card'><h4>{title}</h4><p>{desc.replace(chr(10), '<br>')}</p></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    st.markdown("### 🚀 Quick-Start Guide")
    st.markdown(
        """
1. **Select Index** — choose from the sidebar (LEXAR Medium recommended for general queries)
2. **Load Pipeline** — click "▶ Load / Reload Pipeline" (first load downloads models, ~30–60 s)
3. **Ask a Question** — type your legal question in the "Ask LEXAR" tab and click the button
4. **Upload a Document** — go to the "Upload" tab to add a private PDF; enable "Use in Q&A" to include it
5. **Run Evaluation** — go to the "Evaluation" tab to benchmark retrieval quality on gold queries

**Sample queries:**
- What is the punishment for murder under IPC?
- Define culpable homicide not amounting to murder.
- What are the provisions for bail under CrPC?
- Is a confession to police admissible as evidence?
- What is dacoity and what is its punishment?
"""
    )

    st.divider()
    st.markdown("### 📂 Available Indexes")
    for name, cfg in INDEX_CONFIGS.items():
        exists_ok, missing = _files_exist(name)
        status_icon = "✅" if exists_ok else "❌"
        st.markdown(f"**{status_icon} {name}** — {cfg['description']}")
        if not exists_ok:
            for m in missing:
                st.caption(f"  Missing: `{m}`")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    index_name, top_k, rerank_k, debug_mode, return_provenance, citation_mode = render_sidebar()

    tab_qa, tab_upload, tab_eval, tab_about = st.tabs(
        ["⚖️ Ask LEXAR", "📎 Upload & Ingest", "📊 Evaluation", "ℹ️ About"]
    )

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
