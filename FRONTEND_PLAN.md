# LEXAR Legal AI — Frontend Plan

## Overview

The LEXAR frontend is a **Streamlit** single-page application (`frontend/app.py`) that exposes the full LEXAR pipeline to a non-technical user. It connects **directly** to the Python library (no HTTP hop required) and gives the user rich feedback at every pipeline stage.

---

## Stack Choice

| Concern | Decision | Reason |
|---|---|---|
| Framework | **Streamlit** | Pure Python, runs in the same process as LEXAR, zero JS required |
| Charts | **Plotly Express** | Interactive bar/gauge charts for confidence + evaluation metrics |
| PDF handling | **pdfplumber** (already a dep) | Direct import, no extra package |
| HTTP (optional upload proxy) | **requests** | Only needed if FastAPI backend is running separately |

---

## App Structure

```
frontend/
  app.py               ← single Streamlit entry point
  requirements.txt     ← streamlit + plotly
```

Run: `streamlit run frontend/app.py`

---

## Page Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│ SIDEBAR                     │ MAIN CONTENT                              │
│                             │                                           │
│  ⚖️  LEXAR Legal AI         │  [Tab: Ask LEXAR]  [Upload]  [Eval]  [ℹ] │
│  ─────────────────          │                                           │
│  📚 Knowledge Base          │  (active tab content)                     │
│     [ Index selector ▼ ]   │                                           │
│                             │                                           │
│  ⚙  Pipeline Settings      │                                           │
│  ▶ Advanced (expander)      │                                           │
│     Top-K Retrieval         │                                           │
│     Reranking Top-K         │                                           │
│     Citation Mode           │                                           │
│     Debug Mode              │                                           │
│     Return Provenance       │                                           │
│                             │                                           │
│  [ ▶ Load Pipeline ]        │                                           │
│  ● Pipeline ready           │                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Sidebar: Configuration Panel

### Section 1 — Knowledge Base

| Widget | Type | Values | Default |
|---|---|---|---|
| Index | `selectbox` | LEXAR Medium · IPC Only · IPC + CrPC · IPC + CrPC + IEA | LEXAR Medium |
| Description | static text | Contextual description of selected index | — |

**Index → Files mapping:**

| Display Name | Corpus | FAISS Index | Chunk File(s) |
|---|---|---|---|
| **LEXAR Medium** ⭐ | Multi-law | `lexar_medium.index` | `lexar_medium_chunks.json` |
| IPC Only | Indian Penal Code | `ipc.index` | `ipc_chunks.json` |
| IPC + CrPC | IPC + Criminal Procedure | `ipc_crpc.index` | `ipc_chunks.json` + `crpc_chunks.json` |
| IPC + CrPC + IEA | IPC + CrPC + Evidence Act | `ipc_crpc_iea.index` | All three above + `iea_1872_chunks.json` |

### Section 2 — Advanced Settings (collapsed by default)

| Widget | Type | Range / Options | Default |
|---|---|---|---|
| Top-K Retrieval | `slider` | 3 – 20 | 10 |
| Reranking Top-K | `slider` | 1 – 5 | 3 |
| Citation Mode | `radio` | inline · footnote | inline |
| Debug Mode | `checkbox` | — | off |
| Return Provenance | `checkbox` | — | off |

### Section 3 — Load Pipeline

- **`[ ▶ Load Pipeline ]`** button — instantiates `LexarPipeline` and caches it with `@st.cache_resource`
- While loading: `st.spinner("Loading models… (one-time ~30s)")` with a multi-step text progress
- After load: green `● Pipeline ready` badge + index name shown
- If index files are missing: `st.error()` with a helpful message pointing to build scripts

---

## Tab 1: Ask LEXAR (Q&A)

### Layout

```
╔══════════════════════════════════════════════════════════════╗
║  Ask a legal question                                        ║
║  ┌────────────────────────────────────────────────────────┐  ║
║  │  e.g., What is the punishment for murder under IPC?   │  ║
║  └────────────────────────────────────────────────────────┘  ║
║  [ ⚖  Ask LEXAR ]    [ ✕ Clear ]                            ║
╚══════════════════════════════════════════════════════════════╝

                    ↓ answer area ↓

┌──────────────────────────────────────────────────────────────┐
│  Metrics row                                                 │
│  [ Confidence: 0.84 ]  [ Evidence: 3 chunks ]  [ ✅ Success ]│
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  Answer                                                      │
│  Section 302 of IPC provides for punishment of murder…      │
│  [Primary: IPC §302]  [Supporting: IPC §300, §304]          │
└──────────────────────────────────────────────────────────────┘

▶ Evidence Details (expandable)
▶ Debug Information (expandable, only if debug_mode=True)
```

### Loading States (sequential, replaced as pipeline advances)

| Stage | UI |
|---|---|
| Query submitted | `st.info("Processing your query…")` |
| Stage 1: Routing | Spinner: *"Routing query to relevant legal indices…"* |
| Stage 2: Retrieval | Spinner + progress bar 0→50%: *"Retrieving relevant legal provisions…"* |
| Stage 3: Re-ranking | Spinner + progress bar 50→75%: *"Re-ranking evidence by relevance…"* |
| Stage 4: Generation | Spinner + progress bar 75→100%: *"Generating evidence-constrained answer…"* |
| Done | Clear loading UI; render result |

### Answer Display — States

#### ✅ Status: `success`

```
st.success(answer_text)

Metrics (3 columns):
  col1: st.metric("Confidence", f"{confidence:.0%}")   ← coloured green/amber/red
  col2: st.metric("Evidence Chunks", evidence_count)
  col3: st.metric("Status", "✅ Grounded")

─── Evidence Details (st.expander) ───
  st.dataframe with columns:
    Section | Statute | Rerank Score | Text Preview (first 120 chars)
  
  Citations block:
    Primary: IPC §302
    Supporting: IPC §300, IPC §304

─── Debug Panel (st.expander, only if debug_mode) ───
  Attention distribution per chunk (Plotly bar chart)
  Token provenance table
```

#### ⚠️ Status: `insufficient_evidence`

```
st.warning("⚠️ LEXAR cannot answer this question with sufficient evidence.")

Metric: max_attention vs required threshold (Plotly gauge)

st.info(reason)

Deficit: {deficit:.2%} below threshold

📌 Suggestions:
  • "Try rephrasing with specific section numbers"
  • ...

📋 Evidence Summary:
  (styled text block)
```

#### ❌ Status: `no_evidence`

```
st.error("No relevant legal material found for your query.")
st.info("Suggestions: check spelling, try broader terms, switch to a larger index")
```

#### 🔥 Generation Error

```
st.error(answer_text)
```

### Query History

- Last 5 queries stored in `st.session_state.query_history`
- Shown as clickable chips below the input box: click to re-run

---

## Tab 2: Upload & Ingest

### Layout

```
╔══════════════════════════════════════════════════════╗
║  Upload a Legal PDF                                  ║
║  ┌──────────────────────────────────────────────┐   ║
║  │   📎  Drag and drop PDF here, or Browse      │   ║
║  │   Max file size: 10 MB                       │   ║
║  └──────────────────────────────────────────────┘   ║
╚══════════════════════════════════════════════════════╝

[File info card: name | size | pages (after extraction)]
[ ⬆  Ingest Document ]

     ── loading ──
     Extracting text from PDF…   [spinner]
     Chunking document…           [spinner]
     ── done ──

┌─────────────────────────────────────────────────┐
│  ✅ Document Ingested                            │
│  Document ID: 8f3c…                             │
│  Chunks created: 47                             │
│  Text length: 12,304 chars                      │
└─────────────────────────────────────────────────┘

☑ Use this document in Q&A queries
  (activates UserRetriever in next pipeline call)
```

### Loading States

| Step | UI |
|---|---|
| File selected | Show file name + size immediately |
| Ingest clicked | `st.spinner("Extracting text from PDF…")` |
| Chunking | Update spinner text: `"Chunking document…"` |
| Done | `st.success(...)` card with stats |
| Error | `st.error(...)` with message |

### User Doc Q&A Mode

When the "Use this document in Q&A" checkbox is enabled:
- `st.session_state.user_chunks` holds ingested chunks
- A `UserRetriever(chunks)` is created in-memory
- A `st.info("📎 User document active in pipeline")` banner is shown on the Q&A tab
- To disable: uncheck the box → banner disappears

---

## Tab 3: Evaluation Dashboard

### Layout

```
Pipeline: LEXAR Medium | IPC index: ipc_chunks.json

[ ▶ Run Evaluation ]  (uses evaluation/gold_queries.json — 8 queries)

─── Progress ───
Query 1/8: "What is the punishment for murder…"  ██████░░░ 62%

─── Results ───
┌────────────┬────────────┬───────────┬──────┐
│ Precision@3│ Precision@5│  Recall@5 │  MRR │
│  0.83      │  0.75      │   0.88    │ 0.92 │
└────────────┴────────────┴───────────┴──────┘

Plotly grouped bar chart: per-metric per-query

─── Per-Query Breakdown ───
st.dataframe with columns:
  Query | Relevant Sections | Retrieved Sections | P@3 | P@5 | Recall | RR
```

### Loading States

- `st.progress(i/total)` updated after each query
- `st.status(f"Query {i+1}/{total}")` live status box
- After completion: `st.balloons()` + metrics displayed

---

## Tab 4: About

### Sections

#### System Overview

```
┌──────────────────────────────────────────────────────────────┐
│  LEXAR v1.1.1 — Legal Explainable Augmented Reasoner         │
│  by Garv Behl                                                │
│                                                              │
│  Architecture: Query → Routing → Retrieval → Reranking →     │
│                Evidence-Constrained Generation → Citation    │
└──────────────────────────────────────────────────────────────┘
```

#### Model Cards (3-column grid)

| Card | Info |
|---|---|
| 🔍 Query Encoder | `lexar_query_encoder_v1` (fine-tuned) |
| 📄 Document Encoder | `all-MiniLM-L6-v2` |
| 📊 Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| 🤖 Generator | `google/flan-t5-base` |
| 🔐 Evidence Gate | Threshold: 0.5 attention mass |

#### Key Principles

1. **No generation without evidence** — retrieval is mandatory
2. **Hard attention masking** — decoder cannot see anything outside retrieved chunks
3. **Explicit refusals** — insufficient evidence returns a structured explanation, not a hallucinated answer
4. **Token-level provenance** — every generated token is attributed to a specific chunk

#### Quick-Start Guide

Step-by-step instructions embedded in the tab.

---

## Global UX Behaviours

### Error Guard

Before any Q&A action, check:
1. Pipeline is loaded → else `st.warning("Load pipeline first")`
2. Query is non-empty → else `st.warning("Enter a question")`
3. Index files exist → else `st.error("Index files missing; run build scripts")`

### Session State Keys

| Key | Type | Description |
|---|---|---|
| `pipeline` | `LexarPipeline` | Cached pipeline (via `@st.cache_resource`) |
| `pipeline_config` | `dict` | Index name + settings used for current pipeline |
| `user_chunks` | `list[dict]` | Chunks from user-uploaded PDF |
| `use_user_doc` | `bool` | Whether to activate UserRetriever |
| `last_result` | `dict` | Last answer result for re-display |
| `query_history` | `list[str]` | Last 5 queries |
| `eval_results` | `dict` | Last evaluation run results |

### Performance Notes

- Pipeline models are loaded **once** with `@st.cache_resource` — first load takes ~30–60s; subsequent queries are fast
- Models download automatically from HuggingFace Hub on first run
- The fine-tuned query encoder (`data/models/lexar_query_encoder_v1`) must exist locally; if missing, falls back to base `all-MiniLM-L6-v2`

---

## File: `frontend/requirements.txt`

```
streamlit>=1.32.0
plotly>=5.18.0
```

(All other deps — transformers, faiss, sentence-transformers, pdfplumber — come from pyproject.toml)
