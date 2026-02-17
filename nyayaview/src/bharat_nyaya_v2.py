"""
Bharat Nyaya Console v2 - Main Application
============================================
A research-grade Legal Intelligence Terminal powered by LEXAR architecture.

Run with: streamlit run bharat_nyaya_v2.py --server.port 8502
"""

import streamlit as st
from datetime import datetime
import json
from typing import Optional, Dict, List, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import services
from services.lexar_service import LexarService, RetrievedChunk, AnalysisResult
from services.grok_service import GrokService, GrokResponse, OpenAIService

# ==============================================================================
# Page Configuration
# ==============================================================================

st.set_page_config(
    page_title="Bharat Nyaya Console | LEXAR",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Custom CSS
# ==============================================================================

def render_custom_css():
    """Inject custom CSS for courtroom theme"""
    
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400;700&family=Cinzel:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
    
    :root {
        --primary: #4d0f0f;
        --sandalwood: #EADBC8;
        --antique-gold: #C6A75E;
        --background-dark: #201212;
        --background-darker: #1a0d0d;
    }
    
    .stApp {
        background: radial-gradient(circle at center, #2a1818 0%, #1a0d0d 100%);
    }
    
    /* Hide Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom header */
    .court-header {
        background: var(--background-dark);
        border-bottom: 2px solid var(--primary);
        padding: 1rem 2rem;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .court-title {
        font-family: 'Cinzel', serif;
        font-size: 1.5rem;
        color: var(--antique-gold);
        letter-spacing: 0.2em;
        margin: 0;
    }
    
    .court-subtitle {
        font-size: 0.625rem;
        color: rgba(198, 167, 94, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.3em;
        margin-top: 0.25rem;
    }
    
    /* Mode badge */
    .mode-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border: 2px solid var(--antique-gold);
        color: var(--antique-gold);
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-size: 0.75rem;
    }
    
    .mode-badge.advocate {
        background: rgba(198, 167, 94, 0.2);
    }
    
    .mode-badge.bench {
        background: var(--primary);
    }
    
    /* Parchment document */
    .parchment {
        background-color: #EADBC8;
        background-image: repeating-linear-gradient(
            90deg, transparent, transparent 2px,
            rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px
        );
        padding: 3rem;
        color: #201212;
        font-family: 'Noto Serif', serif;
        min-height: 500px;
        position: relative;
        box-shadow: 0 0 50px rgba(0,0,0,0.5);
        border-radius: 4px;
    }
    
    .parchment h2 {
        font-family: 'Cinzel', serif;
        text-align: center;
        border-bottom: 2px solid rgba(32, 18, 18, 0.2);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .parchment-section {
        margin-bottom: 1.5rem;
    }
    
    .parchment-section h3 {
        font-size: 0.875rem;
        font-weight: bold;
        text-transform: uppercase;
        border-bottom: 1px solid rgba(32, 18, 18, 0.1);
        padding-bottom: 0.25rem;
        margin-bottom: 0.75rem;
    }
    
    .parchment-section p {
        text-align: justify;
        line-height: 1.6;
    }
    
    .court-seal {
        position: absolute;
        bottom: 2rem;
        right: 2rem;
        width: 70px;
        height: 70px;
        border: 3px double #991b1b;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #991b1b;
        font-weight: bold;
        font-size: 0.4rem;
        text-align: center;
        transform: rotate(-15deg);
        opacity: 0.4;
    }
    
    /* Evidence panel */
    .evidence-panel {
        background: var(--background-dark);
        border: 1px solid var(--primary);
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
    
    .evidence-chunk {
        background: rgba(77, 15, 15, 0.3);
        border-left: 3px solid var(--antique-gold);
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
        border-radius: 0 4px 4px 0;
    }
    
    .evidence-chunk.passed {
        border-left-color: #22c55e;
    }
    
    .evidence-chunk.failed {
        border-left-color: #ef4444;
    }
    
    /* Metrics */
    .metric-card {
        background: rgba(77, 15, 15, 0.2);
        border: 1px solid var(--primary);
        padding: 1rem;
        text-align: center;
        border-radius: 4px;
    }
    
    .metric-label {
        font-size: 0.625rem;
        color: var(--antique-gold);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: bold;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        font-variant-numeric: tabular-nums;
    }
    
    .metric-value.risk {
        color: #ef4444;
    }
    
    .metric-value.confidence {
        color: #3b82f6;
    }
    
    /* Gauge meter */
    .gauge {
        width: 80px;
        height: 40px;
        border-radius: 40px 40px 0 0;
        border: 2px solid var(--antique-gold);
        position: relative;
        overflow: hidden;
        background: #311c1c;
        margin: 0.5rem auto;
    }
    
    .gauge-needle {
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 2px;
        height: 35px;
        background: var(--antique-gold);
        transform-origin: bottom center;
    }
    
    /* Statutory record */
    .statute-card {
        background: rgba(77, 15, 15, 0.2);
        border: 1px solid var(--primary);
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        border-radius: 4px;
    }
    
    .statute-title {
        font-size: 0.625rem;
        color: var(--antique-gold);
        text-transform: uppercase;
        font-weight: bold;
    }
    
    .statute-content {
        font-weight: bold;
        color: white;
        margin-top: 0.25rem;
    }
    
    .precedent-card {
        border-left: 2px solid var(--antique-gold);
        padding-left: 0.75rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        padding: 0.5rem 0 0.5rem 0.75rem;
    }
    
    .precedent-card:hover {
        background: rgba(77, 15, 15, 0.2);
    }
    
    .precedent-name {
        font-weight: bold;
        font-size: 0.875rem;
    }
    
    .precedent-citation {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.5);
        font-style: italic;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--primary) !important;
        border: 2px solid var(--antique-gold) !important;
        color: var(--antique-gold) !important;
        font-weight: bold !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    
    .stButton > button:hover {
        background: rgba(77, 15, 15, 0.8) !important;
        border-color: var(--sandalwood) !important;
    }
    
    /* Text areas */
    .stTextArea textarea {
        background: #120808 !important;
        border: 1px solid var(--primary) !important;
        color: var(--antique-gold) !important;
        font-family: 'Noto Serif', serif !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(198, 167, 94, 0.5) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(77, 15, 15, 0.2) !important;
        border: 1px solid var(--primary) !important;
        color: var(--antique-gold) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        background: rgba(77, 15, 15, 0.2);
        padding: 0.5rem;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary) !important;
        color: var(--antique-gold) !important;
        border-top: 2px solid var(--antique-gold);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--background-dark);
        border-right: 1px solid var(--primary);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.8);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #120808 !important;
        border: 1px solid var(--primary) !important;
        color: var(--antique-gold) !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        background: #120808 !important;
        border: 1px solid var(--primary) !important;
        color: var(--antique-gold) !important;
    }
    
    /* Status indicators */
    .status-live {
        color: #22c55e;
        font-weight: bold;
    }
    
    .status-processing {
        color: #fbbf24;
        font-weight: bold;
    }
    
    .status-offline {
        color: #6b7280;
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


# ==============================================================================
# Session State Initialization
# ==============================================================================

def init_session_state():
    """Initialize all session state variables"""
    
    defaults = {
        # User settings
        "user_mode": "Advocate",  # "Advocate" or "Bench"
        "api_key": "",
        "llm_provider": "Grok",  # "Grok" or "OpenAI"
        
        # Case context
        "current_case_type": "Civil",  # "Civil" or "Criminal"
        "current_query": "",
        "case_number": "W.P. (Civil) No. 1234/2024",
        
        # Analysis results
        "analysis_result": None,  # AnalysisResult object
        "evidence_chunks": [],  # List[RetrievedChunk]
        "generation_output": None,
        
        # Metrics
        "metrics": {
            "evidentiary_strength": 0.0,
            "citation_validity": 0.0,
            "procedural_compliance": 0.0,
            "constitutional_risk": 0.0,
            "judicial_confidence": 0.0,
        },
        
        # UI State
        "proceedings_visible": False,
        "is_processing": False,
        "error_message": None,
        "success_message": None,
        
        # Document content
        "document_content": {
            "issues": [],
            "statutory_position": "",
            "reasoning": "",
            "conclusion": "",
            "order": "",
        },
        
        # Statutory context
        "statutory_context": {
            "primary_statute": "Not Analyzed",
            "section": "",
            "supporting_provisions": [],
            "precedents": [],
        },
        
        # Criminal appeal specific
        "criminal_appeal": {
            "charge_sheet": "",
            "witnesses": [],
            "sentencing": "",
        },
        
        # Proceedings history (like chat history)
        "proceedings_history": [],  # List of past analyses
        "selected_proceeding_id": None,  # Currently selected history item
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==============================================================================
# UI Components
# ==============================================================================

def render_header():
    """Render the application header"""
    
    mode = st.session_state.user_mode
    mode_class = "advocate" if mode == "Advocate" else "bench"
    date_str = datetime.now().strftime("%d %B, %Y")
    
    header_html = f"""
    <div class="court-header">
        <div>
            <h1 class="court-title">⚖️ BHARAT NYAYA CONSOLE</h1>
            <p class="court-subtitle">Judicial Evidence Analysis System • LEXAR Architecture</p>
        </div>
        <div style="display: flex; align-items: center; gap: 2rem;">
            <div style="text-align: right;">
                <div style="font-size: 0.75rem; color: rgba(198, 167, 94, 0.7); text-transform: uppercase;">
                    Original Jurisdiction
                </div>
                <div style="color: white; font-weight: bold;">{date_str}</div>
            </div>
            <div class="mode-badge {mode_class}">{mode} Mode</div>
        </div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)


def render_metrics_gauge(label: str, value: float, color: str = "#C6A75E"):
    """Render a single metric with gauge"""
    
    # Calculate needle rotation (-90 to 90 degrees)
    rotation = (value / 100) * 180 - 90
    
    gauge_html = f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="gauge">
            <div class="gauge-needle" style="transform: rotate({rotation}deg); background: {color};"></div>
        </div>
        <div class="metric-value" style="color: {color};">{value:.1f}%</div>
    </div>
    """
    
    st.markdown(gauge_html, unsafe_allow_html=True)


def render_metrics_row():
    """Render all metrics in a row"""
    
    metrics = st.session_state.metrics
    cols = st.columns(5)
    
    metric_configs = [
        ("evidentiary_strength", "Evidence", "#C6A75E"),
        ("citation_validity", "Citations", "#C6A75E"),
        ("procedural_compliance", "Procedure", "#C6A75E"),
        ("constitutional_risk", "Risk", "#ef4444"),
        ("judicial_confidence", "Confidence", "#3b82f6"),
    ]
    
    for col, (key, label, color) in zip(cols, metric_configs):
        with col:
            value = metrics.get(key, 0.0)
            render_metrics_gauge(label, value, color)


def render_parchment_document():
    """Render the parchment document with analysis results"""
    
    doc = st.session_state.document_content
    mode = st.session_state.user_mode
    case_number = st.session_state.case_number
    
    # Check if we have content
    has_content = doc.get("issues") or doc.get("reasoning")
    
    if not has_content:
        # Empty state
        st.markdown("""
        <div class="parchment" style="text-align: center;">
            <h2>IN THE SUPREME COURT OF INDIA</h2>
            <p style="color: rgba(32, 18, 18, 0.5); margin-top: 4rem;">
                Enter a legal question above and click ANALYZE<br>
                to begin evidence-constrained analysis
            </p>
            <div style="font-size: 4rem; color: rgba(32, 18, 18, 0.2); margin-top: 2rem;">⚖️</div>
            <div class="court-seal">
                SUPREME COURT<br>OF INDIA<br>OFFICIAL SEAL
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Format issues
    issues = doc.get("issues", [])
    issues_html = "".join([f"<li style='margin-bottom: 0.5rem;'>{issue}</li>" for issue in issues])
    
    # Mode-specific rendering
    if mode == "Bench":
        # Bench mode: Judgment format with findings and orders
        document_html = f"""
        <div class="parchment">
            <h2>IN THE SUPREME COURT OF INDIA</h2>
            <p style="text-align: center; font-weight: bold; letter-spacing: 0.2em;">
                {"CRIMINAL APPELLATE" if st.session_state.current_case_type == "Criminal" else "CIVIL APPELLATE"} JURISDICTION
            </p>
            <p style="text-align: center; font-style: italic; font-size: 0.875rem;">
                {case_number}
            </p>
            
            <div class="parchment-section">
                <h3>I. Issues for Determination</h3>
                <ol style="padding-left: 1.5rem;">{issues_html}</ol>
            </div>
            
            <div class="parchment-section">
                <h3>II. Findings</h3>
                <p>{doc.get('conclusion', 'Analysis pending...')}</p>
            </div>
            
            <div class="parchment-section">
                <h3>III. Order</h3>
                <p>{doc.get('order', 'The matter is disposed accordingly.')}</p>
            </div>
            
            <div class="court-seal">
                SUPREME COURT<br>OF INDIA<br>OFFICIAL SEAL
            </div>
        </div>
        """
    else:
        # Advocate mode: Full detail with citations and reasoning
        document_html = f"""
        <div class="parchment">
            <h2>IN THE SUPREME COURT OF INDIA</h2>
            <p style="text-align: center; font-weight: bold; letter-spacing: 0.2em;">
                {"CRIMINAL APPELLATE" if st.session_state.current_case_type == "Criminal" else "CIVIL APPELLATE"} JURISDICTION
            </p>
            <p style="text-align: center; font-style: italic; font-size: 0.875rem;">
                {case_number}
            </p>
            
            <div class="parchment-section">
                <h3>I. Issues Framed</h3>
                <ol style="padding-left: 1.5rem;">{issues_html}</ol>
            </div>
            
            <div class="parchment-section">
                <h3>II. Statutory Position</h3>
                <p>{doc.get('statutory_position', 'The relevant statutory provisions are being analyzed...')}</p>
            </div>
            
            <div class="parchment-section">
                <h3>III. Judicial Reasoning</h3>
                <p>{doc.get('reasoning', 'The court observes...')}</p>
            </div>
            
            <div class="parchment-section">
                <h3>IV. Conclusion</h3>
                <p>{doc.get('conclusion', 'In view of the foregoing analysis...')}</p>
            </div>
            
            <div class="parchment-section">
                <h3>V. Order</h3>
                <p>{doc.get('order', 'The petition is disposed of accordingly.')}</p>
            </div>
            
            <div class="court-seal">
                SUPREME COURT<br>OF INDIA<br>OFFICIAL SEAL
            </div>
        </div>
        """
    
    st.markdown(document_html, unsafe_allow_html=True)


def render_evidence_panel():
    """Render the Record of Proceedings / Evidence panel"""
    
    chunks = st.session_state.evidence_chunks
    
    if not chunks:
        st.info("📋 No evidence retrieved yet. Submit a query to retrieve evidence.")
        return
    
    st.markdown("### 📋 RECORD OF PROCEEDINGS")
    st.markdown(f"*{len(chunks)} evidence chunks retrieved*")
    st.markdown("---")
    
    for i, chunk in enumerate(chunks, 1):
        # Handle both dict and dataclass
        if hasattr(chunk, 'text'):
            text = chunk.text
            source = chunk.source
            section = chunk.section
            score = chunk.score
            attention = getattr(chunk, 'attention_weight', 0.0)
            gating = getattr(chunk, 'gating_status', 'PASS')
        else:
            text = chunk.get('text', '')
            source = chunk.get('source', 'Unknown')
            section = chunk.get('section', '')
            score = chunk.get('score', 0.0)
            attention = chunk.get('attention_weight', 0.0)
            gating = chunk.get('gating_status', 'PASS')
        
        gating_color = "#22c55e" if gating == "PASS" else "#ef4444"
        gating_icon = "✓" if gating == "PASS" else "✗"
        
        with st.expander(f"Evidence {i}: {source[:50]}...", expanded=(i <= 2)):
            # Metadata row
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Score:** {score:.3f}")
            with col2:
                st.markdown(f"**Attention:** {attention:.3f}")
            with col3:
                st.markdown(f"<span style='color:{gating_color};font-weight:bold;'>Gating: {gating_icon} {gating}</span>", unsafe_allow_html=True)
            
            st.markdown(f"**Section:** {section}")
            st.markdown("---")
            
            # Content
            st.markdown(f"<div style='background:rgba(77,15,15,0.2);padding:1rem;border-radius:4px;font-size:0.875rem;'>{text}</div>", unsafe_allow_html=True)


def render_statutory_record():
    """Render the statutory record in sidebar"""
    
    ctx = st.session_state.statutory_context
    
    # Primary statute
    st.markdown(f"""
    <div class="statute-card">
        <div class="statute-title">PRIMARY STATUTE</div>
        <div class="statute-content">{ctx.get('primary_statute', 'Not Analyzed')}</div>
        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.6); margin-top: 0.25rem;">{ctx.get('section', '')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Supporting provisions
    provisions = ctx.get('supporting_provisions', [])
    if provisions:
        st.markdown("**Supporting Provisions**")
        for provision in provisions:
            name = provision.get('name', '') if isinstance(provision, dict) else str(provision)
            st.markdown(f"- {name}")
    
    # Precedents
    precedents = ctx.get('precedents', [])
    if precedents:
        st.markdown("---")
        st.markdown("**🏛️ Landmark Judgments**")
        for precedent in precedents:
            if isinstance(precedent, dict):
                name = precedent.get('name', '')
                citation = precedent.get('citation', '')
            else:
                name = str(precedent)
                citation = ""
            
            st.markdown(f"""
            <div class="precedent-card">
                <div class="precedent-name">{name}</div>
                <div class="precedent-citation">{citation}</div>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# Proceedings History (Chat History Feature)
# ==============================================================================

def save_to_proceedings_history(query: str, result):
    """Save current analysis to proceedings history"""
    from datetime import datetime
    import uuid
    
    proceeding = {
        "id": str(uuid.uuid4())[:8],
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "title": query[:50] + "..." if len(query) > 50 else query,
        "case_type": st.session_state.current_case_type,
        "case_number": st.session_state.case_number,
        "mode": st.session_state.user_mode,
        "metrics": st.session_state.metrics.copy(),
        "document_content": st.session_state.document_content.copy(),
        "evidence_count": len(st.session_state.evidence_chunks),
        "dominant_statute": result.dominant_statute,
    }
    
    # Add to beginning of history (most recent first)
    st.session_state.proceedings_history.insert(0, proceeding)
    
    # Keep only last 20 proceedings
    if len(st.session_state.proceedings_history) > 20:
        st.session_state.proceedings_history = st.session_state.proceedings_history[:20]


def load_proceeding(proceeding_id: str):
    """Load a past proceeding into current view"""
    for proc in st.session_state.proceedings_history:
        if proc["id"] == proceeding_id:
            st.session_state.current_query = proc["query"]
            st.session_state.current_case_type = proc["case_type"]
            st.session_state.case_number = proc["case_number"]
            st.session_state.metrics = proc["metrics"]
            st.session_state.document_content = proc["document_content"]
            st.session_state.selected_proceeding_id = proceeding_id
            # Note: We don't restore evidence_chunks to avoid memory issues
            st.session_state.evidence_chunks = []
            st.session_state.proceedings_visible = False
            return True
    return False


def delete_proceeding(proceeding_id: str):
    """Delete a proceeding from history"""
    st.session_state.proceedings_history = [
        p for p in st.session_state.proceedings_history 
        if p["id"] != proceeding_id
    ]


def clear_all_proceedings():
    """Clear all proceedings history"""
    st.session_state.proceedings_history = []
    st.session_state.selected_proceeding_id = None


def render_proceedings_history():
    """Render the proceedings history panel in sidebar"""
    
    history = st.session_state.proceedings_history
    
    if not history:
        st.caption("No previous proceedings")
        return
    
    for proc in history:
        # Format timestamp
        from datetime import datetime
        try:
            ts = datetime.fromisoformat(proc["timestamp"])
            time_str = ts.strftime("%d %b, %H:%M")
        except:
            time_str = "Unknown"
        
        # Determine if selected
        is_selected = proc["id"] == st.session_state.selected_proceeding_id
        
        # Create a container for each proceeding
        with st.container():
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # Display as a clickable item
                label = f"📜 {proc['title']}"
                if is_selected:
                    label = f"▶️ {proc['title']}"
                
                if st.button(
                    label,
                    key=f"proc_{proc['id']}",
                    use_container_width=True,
                    type="primary" if is_selected else "secondary"
                ):
                    load_proceeding(proc["id"])
                    st.rerun()
                
                # Show metadata
                st.caption(f"{proc['case_type']} • {time_str}")
            
            with col2:
                if st.button("🗑️", key=f"del_proc_{proc['id']}"):
                    delete_proceeding(proc["id"])
                    st.rerun()


# ==============================================================================
# Backend Integration
# ==============================================================================

def run_analysis(query: str) -> bool:
    """
    Run the full analysis pipeline.
    Returns True on success, False on failure.
    """
    
    st.session_state.is_processing = True
    st.session_state.error_message = None
    
    try:
        # Step 1: Run LEXAR retrieval and analysis
        lexar = LexarService()
        result = lexar.analyze(
            query=query,
            case_type=st.session_state.current_case_type
        )
        
        # Store analysis results
        st.session_state.analysis_result = result
        st.session_state.evidence_chunks = result.retrieved_chunks
        st.session_state.metrics = {
            "evidentiary_strength": result.evidentiary_strength,
            "citation_validity": result.citation_validity,
            "procedural_compliance": result.procedural_compliance,
            "constitutional_risk": result.constitutional_risk,
            "judicial_confidence": result.judicial_confidence,
        }
        
        # Step 2: Generate response with LLM if API key available
        api_key = st.session_state.api_key.strip()
        mode = st.session_state.user_mode
        
        if api_key:
            if st.session_state.llm_provider == "Grok":
                llm = GrokService(api_key=api_key)
            else:
                llm = OpenAIService(api_key=api_key)
            
            # Generate mode-appropriate response
            if mode == "Advocate":
                response = llm.generate_advocate_response(
                    query=query,
                    evidence_chunks=result.retrieved_chunks
                )
            else:
                response = llm.generate_bench_response(
                    query=query,
                    evidence_chunks=result.retrieved_chunks
                )
            
            st.session_state.generation_output = response
            
            # Use generated content for document
            st.session_state.document_content = {
                "issues": result.issues_framed,
                "statutory_position": result.statutory_position,
                "reasoning": response.reasoning if hasattr(response, 'reasoning') else response.content,
                "conclusion": response.conclusion if hasattr(response, 'conclusion') else result.conclusion,
                "order": response.order if hasattr(response, 'order') else result.order,
            }
        else:
            # Use LEXAR's direct output
            st.session_state.document_content = {
                "issues": result.issues_framed,
                "statutory_position": result.statutory_position,
                "reasoning": result.judicial_reasoning,
                "conclusion": result.conclusion,
                "order": result.order,
            }
        
        # Update statutory context
        st.session_state.statutory_context = {
            "primary_statute": result.dominant_statute,
            "section": "",
            "supporting_provisions": [],
            "precedents": [{"name": c, "citation": ""} for c in result.citations[:3]],
        }
        
        # Save to proceedings history
        save_to_proceedings_history(query, result)
        
        st.session_state.is_processing = False
        st.session_state.success_message = "Analysis complete"
        return True
        
    except Exception as e:
        st.session_state.is_processing = False
        st.session_state.error_message = f"Analysis failed: {str(e)}"
        import traceback
        traceback.print_exc()
        return False


def toggle_proceedings():
    """Toggle the Record of Proceedings visibility"""
    st.session_state.proceedings_visible = not st.session_state.proceedings_visible


def switch_mode(new_mode: str):
    """Switch between Advocate and Bench mode"""
    st.session_state.user_mode = new_mode


def validate_api_key(key: str, provider: str) -> bool:
    """Validate API key format"""
    if not key or len(key) < 10:
        return False
    
    if provider == "Grok":
        return key.startswith("xai-") or len(key) > 20
    elif provider == "OpenAI":
        return key.startswith("sk-") or len(key) > 20
    
    return True


def clear_analysis():
    """Clear all analysis state"""
    st.session_state.analysis_result = None
    st.session_state.evidence_chunks = []
    st.session_state.generation_output = None
    st.session_state.proceedings_visible = False
    st.session_state.metrics = {k: 0.0 for k in st.session_state.metrics}
    st.session_state.document_content = {
        "issues": [],
        "statutory_position": "",
        "reasoning": "",
        "conclusion": "",
        "order": "",
    }
    st.session_state.statutory_context = {
        "primary_statute": "Not Analyzed",
        "section": "",
        "supporting_provisions": [],
        "precedents": [],
    }


# ==============================================================================
# Sidebar
# ==============================================================================

def render_sidebar():
    """Render the sidebar with settings and statutory record"""
    
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # LLM Settings
        st.markdown("### 🤖 LLM Provider")
        
        provider = st.selectbox(
            "Provider",
            ["Grok", "OpenAI"],
            index=0 if st.session_state.llm_provider == "Grok" else 1,
            key="llm_provider_select"
        )
        st.session_state.llm_provider = provider
        
        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            value=st.session_state.api_key,
            key="api_key_input",
            placeholder=f"{'xai-xxx...' if provider == 'Grok' else 'sk-xxx...'}"
        )
        st.session_state.api_key = api_key
        
        if api_key:
            if validate_api_key(api_key, provider):
                st.markdown('<span style="color:#22c55e;">✓ API Key Set</span>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Invalid API key format")
        else:
            st.caption(f"Using LEXAR simulation mode")
        
        st.markdown("---")
        
        # Case Settings
        st.markdown("### 📋 Case Settings")
        
        case_type = st.selectbox(
            "Case Type",
            ["Civil", "Criminal"],
            index=0 if st.session_state.current_case_type == "Civil" else 1,
            key="case_type_select"
        )
        st.session_state.current_case_type = case_type
        
        case_number = st.text_input(
            "Case Number",
            value=st.session_state.case_number,
            key="case_number_input"
        )
        st.session_state.case_number = case_number
        
        st.markdown("---")
        
        # Mode Toggle
        st.markdown("### 👤 User Mode")
        
        col1, col2 = st.columns(2)
        with col1:
            advocate_selected = st.session_state.user_mode == "Advocate"
            if st.button(
                "⚖️ Advocate",
                use_container_width=True,
                type="primary" if advocate_selected else "secondary"
            ):
                switch_mode("Advocate")
                st.rerun()
        
        with col2:
            bench_selected = st.session_state.user_mode == "Bench"
            if st.button(
                "🏛️ Bench",
                use_container_width=True,
                type="primary" if bench_selected else "secondary"
            ):
                switch_mode("Bench")
                st.rerun()
        
        # Mode description
        if st.session_state.user_mode == "Advocate":
            st.caption("📝 Full citations, precedent analysis, detailed reasoning")
        else:
            st.caption("📜 Judgment format: findings and orders only")
        
        st.markdown("---")
        
        # Statutory Record (only show if we have analysis)
        if st.session_state.statutory_context.get("primary_statute", "Not Analyzed") != "Not Analyzed":
            st.markdown("### 📚 Statutory Record")
            render_statutory_record()
        
        st.markdown("---")
        
        # System status
        st.markdown("### 🖥️ System Status")
        
        lexar = LexarService()
        lexar_status = "🟢 Pipeline Ready" if lexar.is_available else "🟡 Simulation Mode"
        llm_status = f"🟢 {provider}" if api_key else "⚪ Not Connected"
        
        st.markdown(f"""
        - **LEXAR:** {lexar_status}
        - **LLM:** {llm_status}
        - **Mode:** {st.session_state.user_mode}
        - **Case:** {st.session_state.current_case_type}
        """)
        
        st.markdown("---")
        
        # Proceedings History (like ChatGPT chat history)
        st.markdown("### 📜 Record of Proceedings")
        
        # New Proceeding button
        if st.button("➕ New Proceeding", use_container_width=True):
            clear_analysis()
            st.session_state.selected_proceeding_id = None
            st.rerun()
        
        # Show history
        render_proceedings_history()
        
        # Clear all button (if there's history)
        if st.session_state.proceedings_history:
            st.markdown("---")
            if st.button("🗑️ Clear All History", use_container_width=True):
                clear_all_proceedings()
                st.rerun()


# ==============================================================================
# Main Content
# ==============================================================================

def render_main_content():
    """Render the main content area"""
    
    # Query input section
    st.markdown("### ⚖️ QUESTION OF LAW PRESENTED")
    
    query = st.text_area(
        "State the issue for judicial determination...",
        height=100,
        key="query_input",
        label_visibility="collapsed",
        placeholder="Enter the legal question or issue to be analyzed...\n\nExample: Whether the conviction under Section 302 IPC is sustainable when the prosecution has failed to establish the chain of custody of the murder weapon?"
    )
    st.session_state.current_query = query
    
    # Action buttons row
    col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
    
    with col1:
        analyze_disabled = not query.strip() or st.session_state.is_processing
        if st.button(
            "🔍 ANALYZE",
            use_container_width=True,
            disabled=analyze_disabled,
            type="primary"
        ):
            with st.spinner("Analyzing with LEXAR..."):
                success = run_analysis(query)
                if success:
                    st.rerun()
    
    with col2:
        has_evidence = len(st.session_state.evidence_chunks) > 0
        proceedings_label = "📋 Hide Proceedings" if st.session_state.proceedings_visible else "📋 Show Proceedings"
        if st.button(
            proceedings_label,
            use_container_width=True,
            disabled=not has_evidence
        ):
            toggle_proceedings()
            st.rerun()
    
    with col3:
        if st.button(
            "🔄 Clear",
            use_container_width=True
        ):
            clear_analysis()
            st.rerun()
    
    # Show messages
    if st.session_state.error_message:
        st.error(f"❌ {st.session_state.error_message}")
        st.session_state.error_message = None
    
    if st.session_state.success_message:
        st.success(f"✓ {st.session_state.success_message}")
        st.session_state.success_message = None
    
    st.markdown("---")
    
    # Metrics row (only show if we have metrics)
    if st.session_state.metrics.get("evidentiary_strength", 0) > 0:
        render_metrics_row()
        st.markdown("---")
    
    # Two-column layout: Document + Evidence
    if st.session_state.proceedings_visible and st.session_state.evidence_chunks:
        doc_col, evidence_col = st.columns([2, 1])
        
        with doc_col:
            render_parchment_document()
        
        with evidence_col:
            render_evidence_panel()
    else:
        render_parchment_document()


def render_criminal_appeal_section():
    """Render Criminal Appeal specific section"""
    
    if st.session_state.current_case_type != "Criminal":
        return
    
    # Only show if we have analysis
    if not st.session_state.analysis_result:
        return
    
    st.markdown("---")
    st.markdown("### ⚖️ CRIMINAL APPEAL RECORD")
    
    tabs = st.tabs(["📋 Charge Sheet", "👥 Witnesses", "⚖️ Sentencing"])
    
    with tabs[0]:
        charge_sheet = st.text_area(
            "Charge Sheet Summary",
            value=st.session_state.criminal_appeal.get("charge_sheet", ""),
            height=150,
            placeholder="Enter charge sheet details or summarize the allegations...",
            key="charge_sheet_input"
        )
        st.session_state.criminal_appeal["charge_sheet"] = charge_sheet
    
    with tabs[1]:
        st.markdown("**Witness List**")
        
        col1, col2 = st.columns([4, 1])
        with col1:
            witness_input = st.text_input(
                "Add Witness",
                placeholder="Name - Role - Key testimony",
                key="witness_input",
                label_visibility="collapsed"
            )
        with col2:
            if st.button("➕ Add", key="add_witness_btn"):
                if witness_input:
                    if "witnesses" not in st.session_state.criminal_appeal:
                        st.session_state.criminal_appeal["witnesses"] = []
                    st.session_state.criminal_appeal["witnesses"].append(witness_input)
                    st.rerun()
        
        witnesses = st.session_state.criminal_appeal.get("witnesses", [])
        if witnesses:
            for i, witness in enumerate(witnesses):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{i+1}.** {witness}")
                with col2:
                    if st.button("🗑️", key=f"del_witness_{i}"):
                        st.session_state.criminal_appeal["witnesses"].pop(i)
                        st.rerun()
        else:
            st.caption("No witnesses added yet")
    
    with tabs[2]:
        sentencing = st.text_area(
            "Sentencing Analysis",
            value=st.session_state.criminal_appeal.get("sentencing", ""),
            height=150,
            placeholder="Enter sentencing considerations, mitigating/aggravating factors...",
            key="sentencing_input"
        )
        st.session_state.criminal_appeal["sentencing"] = sentencing


# ==============================================================================
# Main Application Entry Point
# ==============================================================================

def main():
    """Main application entry point"""
    
    # Initialize session state
    init_session_state()
    
    # Inject custom CSS
    render_custom_css()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Render main content
    render_main_content()
    
    # Render Criminal Appeal section if applicable
    render_criminal_appeal_section()


if __name__ == "__main__":
    main()
