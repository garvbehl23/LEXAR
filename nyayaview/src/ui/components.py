"""
UI Components Module
=====================
Reusable Streamlit components for the Bharat Nyaya Console.

Contains:
- Header component
- Evidence panel
- Metrics display
- Document viewer
- Settings panel
"""

import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime


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
        min-height: 600px;
        position: relative;
        box-shadow: 0 0 50px rgba(0,0,0,0.5);
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
        bottom: 3rem;
        right: 3rem;
        width: 80px;
        height: 80px;
        border: 4px double #991b1b;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #991b1b;
        font-weight: bold;
        font-size: 0.5rem;
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
    }
    
    .evidence-chunk {
        background: rgba(77, 15, 15, 0.3);
        border-left: 3px solid var(--antique-gold);
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        font-size: 0.875rem;
    }
    
    .evidence-chunk.passed {
        border-left-color: #22c55e;
    }
    
    .evidence-chunk.failed {
        border-left-color: #ef4444;
    }
    
    .evidence-source {
        font-weight: bold;
        color: var(--antique-gold);
        margin-bottom: 0.25rem;
    }
    
    .evidence-score {
        font-size: 0.75rem;
        color: rgba(255, 255, 255, 0.6);
    }
    
    /* Metrics */
    .metric-container {
        background: rgba(77, 15, 15, 0.2);
        border: 1px solid var(--primary);
        padding: 1rem;
        text-align: center;
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
        margin: 0 auto;
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
    }
    
    /* Text areas */
    .stTextArea textarea {
        background: #120808 !important;
        border: 1px solid var(--primary) !important;
        color: var(--antique-gold) !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: rgba(77, 15, 15, 0.2) !important;
        border: 1px solid var(--primary) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(77, 15, 15, 0.2);
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--primary);
        color: var(--antique-gold);
        border-top: 2px solid var(--antique-gold);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--background-dark);
        border-right: 1px solid var(--primary);
    }
    
    /* Status indicator */
    .status-live {
        color: #22c55e;
        font-weight: bold;
    }
    
    .status-processing {
        color: #fbbf24;
        font-weight: bold;
    }
    
    /* Footer metrics bar */
    .metrics-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: var(--background-dark);
        border-top: 2px solid var(--primary);
        padding: 1rem 2rem;
        display: flex;
        gap: 2rem;
        z-index: 100;
    }
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def render_header(mode: str, date_str: str):
    """Render the application header"""
    
    mode_class = "advocate" if mode == "Advocate" else "bench"
    
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


def render_evidence_panel(chunks: List[Any], is_visible: bool = True):
    """Render the Record of Proceedings / Evidence panel"""
    
    if not is_visible:
        return
    
    st.markdown("### 📋 RECORD OF PROCEEDINGS")
    st.markdown("---")
    
    if not chunks:
        st.info("No evidence retrieved yet. Submit a query to retrieve evidence.")
        return
    
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
        
        with st.expander(f"Evidence {i}: {source[:40]}...", expanded=(i <= 2)):
            st.markdown(f"""
            <div style="font-size: 0.875rem;">
                <strong style="color: #C6A75E;">Source:</strong> {source}<br>
                <strong style="color: #C6A75E;">Section:</strong> {section}<br>
                <strong style="color: #C6A75E;">Relevance Score:</strong> {score:.3f}<br>
                <strong style="color: #C6A75E;">Attention Weight:</strong> {attention:.3f}<br>
                <strong style="color: {gating_color};">Gating Status:</strong> {gating}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown(f"<div style='font-size: 0.875rem; color: rgba(255,255,255,0.8);'>{text}</div>", unsafe_allow_html=True)


def render_metrics_gauge(label: str, value: float, color: str = "#C6A75E"):
    """Render a single metric with gauge"""
    
    # Calculate needle rotation (-90 to 90 degrees)
    rotation = (value / 100) * 180 - 90
    
    gauge_html = f"""
    <div class="metric-container">
        <div class="metric-label">{label}</div>
        <div class="gauge">
            <div class="gauge-needle" style="transform: rotate({rotation}deg); background: {color};"></div>
        </div>
        <div class="metric-value" style="color: {color};">{value:.1f}</div>
    </div>
    """
    
    st.markdown(gauge_html, unsafe_allow_html=True)


def render_metrics_row(metrics: Dict[str, float]):
    """Render all metrics in a row"""
    
    cols = st.columns(5)
    
    metric_configs = [
        ("evidentiary_strength", "Evidentiary Strength %", "#C6A75E"),
        ("citation_validity", "Citation Validity", "#C6A75E"),
        ("procedural_compliance", "Procedural Compliance", "#C6A75E"),
        ("constitutional_risk", "Constitutional Risk", "#ef4444"),
        ("judicial_confidence", "Judicial Confidence", "#3b82f6"),
    ]
    
    for col, (key, label, color) in zip(cols, metric_configs):
        with col:
            value = metrics.get(key, 0.0)
            render_metrics_gauge(label, value, color)


def render_parchment_document(
    case_number: str,
    issues: List[str],
    statutory_position: str,
    reasoning: str,
    conclusion: str,
    order: str,
    mode: str = "Bench"
):
    """Render the parchment document with analysis results"""
    
    # Format issues
    issues_html = "".join([f"<li>{issue}</li>" for issue in issues])
    
    # Mode-specific rendering
    if mode == "Bench":
        # Bench mode: Hide detailed reasoning, show structured judgment
        document_html = f"""
        <div class="parchment">
            <h2>IN THE SUPREME COURT OF INDIA</h2>
            <p style="text-align: center; font-weight: bold; letter-spacing: 0.2em;">
                CIVIL / CRIMINAL APPELLATE JURISDICTION
            </p>
            <p style="text-align: center; font-style: italic; font-size: 0.875rem;">
                {case_number}
            </p>
            
            <div class="parchment-section">
                <h3>I. Issues Framed</h3>
                <ol style="padding-left: 1.5rem;">{issues_html}</ol>
            </div>
            
            <div class="parchment-section">
                <h3>II. Findings</h3>
                <p>{conclusion}</p>
            </div>
            
            <div class="parchment-section">
                <h3>III. Order</h3>
                <p>{order}</p>
            </div>
            
            <div class="court-seal">
                SUPREME COURT<br>OF INDIA<br>OFFICIAL SEAL
            </div>
        </div>
        """
    else:
        # Advocate mode: Full detail with citations
        document_html = f"""
        <div class="parchment">
            <h2>IN THE SUPREME COURT OF INDIA</h2>
            <p style="text-align: center; font-weight: bold; letter-spacing: 0.2em;">
                CIVIL / CRIMINAL APPELLATE JURISDICTION
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
                <p>{statutory_position}</p>
            </div>
            
            <div class="parchment-section">
                <h3>III. Judicial Reasoning</h3>
                <p>{reasoning}</p>
            </div>
            
            <div class="parchment-section">
                <h3>IV. Conclusion</h3>
                <p>{conclusion}</p>
            </div>
            
            <div class="parchment-section">
                <h3>V. Order</h3>
                <p>{order}</p>
            </div>
            
            <div class="court-seal">
                SUPREME COURT<br>OF INDIA<br>OFFICIAL SEAL
            </div>
        </div>
        """
    
    st.markdown(document_html, unsafe_allow_html=True)


def render_statutory_record(
    primary_statute: str,
    section: str,
    supporting_provisions: List[Dict],
    precedents: List[Dict]
):
    """Render the statutory record sidebar"""
    
    st.markdown("### 📚 STATUTORY RECORD")
    st.markdown("---")
    
    # Primary statute
    st.markdown(f"""
    <div class="statute-card">
        <div class="statute-title">PRIMARY STATUTE</div>
        <div class="statute-content">{primary_statute}</div>
        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.6); margin-top: 0.25rem;">{section}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Supporting provisions
    st.markdown("#### Supporting Provisions")
    for provision in supporting_provisions:
        with st.expander(provision.get('name', 'Provision')):
            st.write(provision.get('content', ''))
    
    st.markdown("---")
    
    # Precedents
    st.markdown("#### 🏛️ Landmark Judgments")
    for precedent in precedents:
        active = precedent.get('active', False)
        border_color = "#C6A75E" if active else "rgba(198, 167, 94, 0.3)"
        text_color = "white" if active else "rgba(255, 255, 255, 0.6)"
        
        st.markdown(f"""
        <div class="precedent-card" style="border-left-color: {border_color};">
            <div class="precedent-name" style="color: {text_color};">{precedent.get('name', '')}</div>
            <div class="precedent-citation">{precedent.get('citation', '')}</div>
        </div>
        """, unsafe_allow_html=True)


def render_query_input():
    """Render the Question of Law input section"""
    
    st.markdown("### ⚖️ QUESTION OF LAW PRESENTED")
    
    query = st.text_area(
        "State the issue for judicial determination...",
        height=100,
        key="query_input",
        label_visibility="collapsed",
        placeholder="Enter the legal question or issue to be analyzed..."
    )
    
    return query


def render_loading_state():
    """Render loading animation"""
    
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <div style="font-size: 2rem; color: #C6A75E; margin-bottom: 1rem;">⚖️</div>
        <div style="color: #C6A75E; font-weight: bold;">Analyzing with LEXAR Architecture...</div>
        <div style="color: rgba(255,255,255,0.5); font-size: 0.875rem; margin-top: 0.5rem;">
            Retrieving evidence • Reranking • Generating response
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_error_alert(message: str):
    """Render error alert"""
    st.error(f"❌ {message}")


def render_success_alert(message: str):
    """Render success alert"""
    st.success(f"✓ {message}")
