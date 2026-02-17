"""
Bharat Nyaya Console v4 - Pure Streamlit
==========================================
Simplified version using native Streamlit components.

Run with: streamlit run bharat_nyaya_v4.py --server.port 8502
"""

import streamlit as st
from datetime import datetime
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
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# Custom CSS - Injected Once
# ==============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Cormorant+Garaiant:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --maroon-deep: #1a0a0a;
    --maroon-primary: #2d1111;
    --maroon-accent: #4d1f1f;
    --maroon-border: #6b2a2a;
    --gold-primary: #c9a227;
    --gold-light: #e8c547;
    --gold-muted: #8b7355;
    --parchment: #f4e4c1;
    --parchment-dark: #e8d4a8;
    --text-light: #f5f5f5;
    --text-muted: #a89f8f;
}

.stApp {
    background: linear-gradient(180deg, #1a0a0a 0%, #0d0505 100%) !important;
}

#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}
div[data-testid="stToolbar"] {display: none;}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    max-width: 100% !important;
}

/* Hide default header */
div[data-testid="stHeader"] {
    display: none;
}

/* Button Styles */
.stButton > button {
    background: #4d1f1f !important;
    border: 1px solid #6b2a2a !important;
    color: #c9a227 !important;
    font-weight: 600 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: #c9a227 !important;
    color: #1a0a0a !important;
    border-color: #c9a227 !important;
}

/* Text inputs */
.stTextArea textarea, .stTextInput > div > div > input {
    background: #120808 !important;
    border: 1px solid #4d1f1f !important;
    color: #f5f5f5 !important;
    font-family: 'Inter', sans-serif !important;
}

.stTextArea textarea::placeholder, .stTextInput input::placeholder {
    color: #8b7355 !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: #120808 !important;
    border: 1px solid #4d1f1f !important;
    color: #c9a227 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #2d1111;
    gap: 2px;
    padding: 4px;
    border-radius: 4px;
}

.stTabs [data-baseweb="tab"] {
    background: #1a0a0a;
    color: #a89f8f;
    border: 1px solid #4d1f1f;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.5rem 1rem;
    border-radius: 2px;
}

.stTabs [aria-selected="true"] {
    background: #4d1f1f !important;
    color: #c9a227 !important;
    border-color: #c9a227 !important;
}

/* Expander */
.stExpander {
    background: #1a0a0a !important;
    border: 1px solid #4d1f1f !important;
    border-radius: 4px;
}

/* Slider */
.stSlider > div > div > div {
    background: #c9a227 !important;
}

/* Text colors */
h1, h2, h3, h4, h5, h6 {
    color: #c9a227 !important;
    font-family: 'Cinzel', serif !important;
}

p, .stMarkdown, label {
    color: #f5f5f5 !important;
}

/* Custom header bar */
.header-bar {
    background: linear-gradient(180deg, #2d1111 0%, #1a0a0a 100%);
    border-bottom: 2px solid #4d1f1f;
    padding: 0.75rem 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: -1rem -1rem 1rem -1rem;
}

.header-title {
    font-family: 'Cinzel', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #c9a227;
    letter-spacing: 0.15em;
}

.header-subtitle {
    font-size: 0.65rem;
    color: #a89f8f;
    text-transform: uppercase;
    letter-spacing: 0.2em;
}

.header-date {
    font-family: 'Cinzel', serif;
    font-size: 0.9rem;
    color: #f5f5f5;
    text-align: center;
}

.header-jurisdiction {
    font-size: 0.6rem;
    color: #a89f8f;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}

/* Parchment document */
.parchment {
    background: linear-gradient(135deg, #f4e4c1 0%, #e8d4a8 100%);
    border-radius: 4px;
    padding: 2rem 2.5rem;
    color: #2d1f1f;
    position: relative;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 0 100px rgba(139, 69, 19, 0.1);
    margin: 1rem 0;
}

.parchment-title {
    font-family: 'Cinzel', serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #1a0a0a !important;
    letter-spacing: 0.1em;
    text-align: center;
    margin-bottom: 0.5rem;
}

.parchment-subtitle {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #4d1f1f !important;
    text-transform: uppercase;
    text-align: center;
}

.parchment-case {
    font-size: 0.85rem;
    font-style: italic;
    color: #6b4a4a !important;
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 1.5rem;
}

.section-header {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4d1f1f !important;
    border-bottom: 1px solid rgba(77, 31, 31, 0.3);
    padding-bottom: 0.25rem;
    margin-bottom: 0.5rem;
}

.section-content {
    font-family: 'Georgia', serif;
    font-size: 1rem;
    line-height: 1.7;
    color: #2d1f1f !important;
    text-align: justify;
    margin-bottom: 1.5rem;
}

.court-seal {
    position: absolute;
    bottom: 1.5rem;
    right: 1.5rem;
    width: 80px;
    height: 80px;
    border: 2px double #8b4a4a;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #8b4a4a;
    font-family: 'Cinzel', serif;
    font-size: 0.4rem;
    text-align: center;
    font-weight: 600;
    transform: rotate(-15deg);
    opacity: 0.4;
}

/* Metrics bar */
.metrics-bar {
    background: linear-gradient(180deg, #2d1111 0%, #1a0a0a 100%);
    border-top: 2px solid #4d1f1f;
    padding: 1rem;
    margin: 1rem -1rem -1rem -1rem;
}

.metric-card {
    background: #1a0a0a;
    border: 1px solid #4d1f1f;
    border-radius: 4px;
    padding: 0.75rem;
    text-align: center;
}

.metric-label {
    font-size: 0.55rem;
    color: #a89f8f;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
}

.metric-gold { color: #c9a227; }
.metric-danger { color: #ef4444; }
.metric-info { color: #60a5fa; }

/* Nav panel */
.nav-panel {
    background: #2d1111;
    border: 1px solid #4d1f1f;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.nav-title {
    font-family: 'Cinzel', serif;
    font-size: 0.8rem;
    color: #c9a227;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* Statute card */
.statute-card {
    background: #4d1f1f;
    border: 1px solid #6b2a2a;
    border-radius: 4px;
    padding: 0.75rem;
    margin-bottom: 1rem;
}

.statute-label {
    font-size: 0.6rem;
    color: #c9a227;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.statute-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #f5f5f5;
}

.statute-section {
    font-size: 0.75rem;
    color: #a89f8f;
}

/* Judgment card */
.judgment-item {
    border-left: 2px solid #c9a227;
    padding-left: 0.75rem;
    margin-bottom: 0.75rem;
}

.judgment-name {
    font-size: 0.8rem;
    font-weight: 600;
    color: #f5f5f5;
}

.judgment-cite {
    font-size: 0.7rem;
    color: #a89f8f;
    font-style: italic;
}

/* Mode toggle */
.mode-btn-active {
    background: #c9a227 !important;
    color: #1a0a0a !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #2d1111 !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# Session State Initialization
# ==============================================================================

def init_session_state():
    defaults = {
        "user_mode": "ADVOCATE",
        "case_type": "constitutional",
        "case_number": "Special Leave Petition (Civil) No. 4920 of 2024",
        "current_query": "",
        "active_nav": "fir",
        "analysis_result": None,
        "metrics": {
            "evidentiary_strength": 74.2,
            "citation_validity": 91.8,
            "procedural_compliance": 82.5,
            "constitutional_risk": 12.4,
            "judicial_confidence": 96.0,
        },
        "document_content": {
            "matter": """This petition concerns the interpretation of fundamental rights 
under Part III of the Constitution, specifically the interplay between 
procedural fairness and evidentiary standards in high-stakes 
technological crimes. The Petitioner seeks a writ of certiorari against 
the high court's previous determination.""",
            "issues": [
                "Whether the integrity of hash-values in digital forensic logs constitutes a 'primary fact' under the Indian Evidence Act.",
                "The scope of judicial review in matters involving automated algorithmic sentencing recommendations.",
            ],
            "reasoning": """The Court has carefully examined the submissions made by learned counsel 
for both parties. Having regard to the constitutional provisions and the 
precedents cited, the following observations are made...""",
        },
        "statutory_context": {
            "primary_statute": "The Bharatiya Nyaya Sanhita, 2023",
            "section": "Section 113: Terrorism",
            "supporting_provisions": ["Art. 21 - Right to Life", "Sec. 65B Evidence Act"],
            "precedents": [
                {"name": "K.S. Puttaswamy v. Union of India", "citation": "2017 (10) SCC 1"},
                {"name": "Maneka Gandhi v. Union of India", "citation": "1978 AIR 597"},
                {"name": "Kesavananda Bharati Case", "citation": "1973 (4) SCC 225"},
            ],
        },
        "llm_provider": "Groq",
        "api_key": "",
        "model_name": "llama-3.1-70b-versatile",
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==============================================================================
# Case Templates
# ==============================================================================

CASE_TEMPLATES = {
    "constitutional": {
        "title": "CONSTITUTIONAL MATTER",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "CIVIL / CRIMINAL APPELLATE JURISDICTION",
        "jurisdiction": "Article 142 Jurisdiction",
    },
    "criminal": {
        "title": "CRIMINAL APPEAL",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "CRIMINAL APPELLATE JURISDICTION",
        "jurisdiction": "Section 374 CrPC",
    },
    "civil": {
        "title": "CIVIL APPEAL",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "CIVIL APPELLATE JURISDICTION",
        "jurisdiction": "Order XLI CPC",
    },
    "writ": {
        "title": "WRIT PETITION",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "ORIGINAL JURISDICTION",
        "jurisdiction": "Article 32/226",
    },
    "slp": {
        "title": "SPECIAL LEAVE PETITION",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "SPECIAL LEAVE JURISDICTION",
        "jurisdiction": "Article 136",
    },
}

# ==============================================================================
# Header
# ==============================================================================

date_str = datetime.now().strftime("%A, %d %B %Y").upper()
template = CASE_TEMPLATES[st.session_state.case_type]

st.markdown(f"""
<div class="header-bar">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <span style="font-size: 2rem;">🏛️</span>
        <div>
            <div class="header-title">BHARAT NYAYA CONSOLE</div>
            <div class="header-subtitle">Judicial Evidence Analysis System</div>
        </div>
    </div>
    <div style="text-align: center;">
        <div class="header-jurisdiction">{template['jurisdiction']}</div>
        <div class="header-date">{date_str}</div>
    </div>
    <div style="display: flex; align-items: center; gap: 1rem;">
        <div style="font-size: 0.75rem; color: #22c55e; font-weight: 600;">
            🟢 BENCH LIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# Mode Toggle Row
# ==============================================================================

col_spacer, col_mode1, col_mode2, col_seal = st.columns([5, 0.8, 0.8, 1.5])

with col_mode1:
    adv_type = "primary" if st.session_state.user_mode == "ADVOCATE" else "secondary"
    if st.button("⚖️ ADVOCATE", key="adv_btn", type=adv_type):
        st.session_state.user_mode = "ADVOCATE"
        st.rerun()

with col_mode2:
    bench_type = "primary" if st.session_state.user_mode == "BENCH" else "secondary"
    if st.button("🏛️ BENCH", key="bench_btn", type=bench_type):
        st.session_state.user_mode = "BENCH"
        st.rerun()

with col_seal:
    if st.button("📜 SEAL & EXPORT", key="seal_btn"):
        st.toast("Decree sealed and exported!", icon="📜")

# ==============================================================================
# Main Layout
# ==============================================================================

left_col, main_col, right_col = st.columns([1.2, 3, 1.5])

# ==============================================================================
# Left Sidebar
# ==============================================================================

with left_col:
    st.markdown('<div class="nav-panel">', unsafe_allow_html=True)
    st.markdown('<div class="nav-title">📋 RECORD OF PROCEEDINGS</div>', unsafe_allow_html=True)
    
    nav_items = [
        ("fir", "📋 FIR / CASE NUMBER"),
        ("parties", "👥 PARTIES"),
        ("annexures", "📎 ANNEXURES"),
        ("forensic", "🔬 FORENSIC REPORTS"),
        ("evidence", "📁 EVIDENCE LOG"),
    ]
    
    for nav_id, label in nav_items:
        btn_type = "primary" if st.session_state.active_nav == nav_id else "secondary"
        if st.button(label, key=f"nav_{nav_id}", use_container_width=True, type=btn_type):
            st.session_state.active_nav = nav_id
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Content based on selected nav
    active = st.session_state.active_nav
    
    if active == "fir":
        st.text_input("FIR Number", placeholder="FIR-2024-XXX", key="fir_input")
        st.text_input("Police Station", placeholder="Enter PS name", key="ps_input")
        st.date_input("Date Filed", key="date_input")
    elif active == "parties":
        st.text_input("Petitioner", placeholder="Name of Petitioner", key="pet_input")
        st.text_input("Respondent", placeholder="Name of Respondent", key="resp_input")
    elif active == "annexures":
        st.file_uploader("Upload Annexure", type=["pdf", "docx"], key="annex_upload")
    elif active == "forensic":
        st.file_uploader("Upload Report", type=["pdf"], key="forensic_upload")
    elif active == "evidence":
        st.caption("Evidence will appear here after analysis")
    
    # LLM Settings
    with st.expander("⚙️ LLM SETTINGS"):
        provider = st.selectbox("Provider", ["Groq", "OpenAI", "Anthropic"], key="provider_sel")
        st.session_state.llm_provider = provider
        
        api_key = st.text_input(f"{provider} API Key", type="password", key="api_input")
        st.session_state.api_key = api_key
        
        if provider == "Groq":
            models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
        elif provider == "OpenAI":
            models = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
        else:
            models = ["claude-3-opus", "claude-3-sonnet"]
        
        model = st.selectbox("Model", models, key="model_sel")
        st.session_state.model_name = model
        
        temp = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1, key="temp_sl")
        st.session_state.temperature = temp

# ==============================================================================
# Main Content
# ==============================================================================

with main_col:
    # Case Type Tabs
    tab_const, tab_crim, tab_civil, tab_writ, tab_slp = st.tabs([
        "CONSTITUTIONAL", "CRIMINAL", "CIVIL", "WRIT", "SLP"
    ])
    
    case_map = {
        "CONSTITUTIONAL": "constitutional",
        "CRIMINAL": "criminal", 
        "CIVIL": "civil",
        "WRIT": "writ",
        "SLP": "slp"
    }
    
    # Query Input
    st.markdown("##### ⚖️ QUESTION OF LAW PRESENTED")
    
    query_col, btn_col = st.columns([4, 1])
    
    with query_col:
        query = st.text_area(
            "Query",
            value=st.session_state.current_query,
            height=80,
            placeholder="State the issue for judicial determination...",
            label_visibility="collapsed",
            key="query_input"
        )
        st.session_state.current_query = query
    
    with btn_col:
        st.markdown(f"*{template['jurisdiction']}*")
        if st.button("⚖️ PLACE BEFORE BENCH", key="analyze_btn", type="primary", use_container_width=True):
            if query.strip():
                with st.spinner("Analyzing with LEXAR..."):
                    try:
                        lexar = LexarService()
                        result = lexar.analyze(query=query, case_type=st.session_state.case_type)
                        st.session_state.analysis_result = result
                        st.session_state.metrics = {
                            "evidentiary_strength": result.evidentiary_strength,
                            "citation_validity": result.citation_validity,
                            "procedural_compliance": result.procedural_compliance,
                            "constitutional_risk": result.constitutional_risk,
                            "judicial_confidence": result.judicial_confidence,
                        }
                        st.session_state.document_content["reasoning"] = result.judicial_reasoning
                        st.success("Analysis complete!")
                    except Exception as e:
                        st.error(f"Analysis error: {e}")
    
    # Parchment Document
    doc = st.session_state.document_content
    issues_html = "".join([f"<li style='color: #2d1f1f; margin-bottom: 0.5rem;'>{issue}</li>" for issue in doc["issues"]])
    
    st.markdown(f"""
    <div class="parchment">
        <h1 class="parchment-title">{template['court']}</h1>
        <p class="parchment-subtitle">{template['subtitle']}</p>
        <p class="parchment-case">{st.session_state.case_number}</p>
        
        <h4 class="section-header">I. THE MATTER</h4>
        <p class="section-content">{doc['matter']}</p>
        
        <h4 class="section-header">II. ISSUES FRAMED</h4>
        <ol style="padding-left: 1.5rem; margin-bottom: 1.5rem;">
            {issues_html}
        </ol>
        
        <h4 class="section-header">III. JUDICIAL REASONING</h4>
        <p class="section-content">{doc['reasoning']}</p>
        
        <div class="court-seal">
            SUPREME COURT<br>OF INDIA<br>•<br>OFFICIAL SEAL
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# Right Sidebar
# ==============================================================================

with right_col:
    st.markdown("#### STATUTORY RECORD")
    
    ctx = st.session_state.statutory_context
    
    st.markdown(f"""
    <div class="statute-card">
        <div class="statute-label">PRIMARY STATUTE</div>
        <div class="statute-title">{ctx['primary_statute']}</div>
        <div class="statute-section">{ctx['section']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**SUPPORTING PROVISIONS**")
    for prov in ctx["supporting_provisions"]:
        with st.expander(prov):
            st.write("Detailed provision text...")
    
    st.markdown("---")
    st.markdown("**LANDMARK JUDGMENTS**")
    
    for j in ctx["precedents"]:
        st.markdown(f"""
        <div class="judgment-item">
            <div class="judgment-name">{j['name']}</div>
            <div class="judgment-cite">{j['citation']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("📚 CONSULT AI ARCHIVE", key="archive_btn", use_container_width=True):
        st.info("Searching precedent archive...")

# ==============================================================================
# Metrics Bar
# ==============================================================================

st.markdown('<div class="metrics-bar">', unsafe_allow_html=True)

m = st.session_state.metrics
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">EVIDENTIARY STRENGTH</div>
        <div class="metric-value metric-gold">{m['evidentiary_strength']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CITATION VALIDITY</div>
        <div class="metric-value metric-gold">{m['citation_validity']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">PROCEDURAL COMPLIANCE</div>
        <div class="metric-value metric-gold">{m['procedural_compliance']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">CONSTITUTIONAL RISK</div>
        <div class="metric-value metric-danger">{m['constitutional_risk']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with m5:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">JUDICIAL CONFIDENCE</div>
        <div class="metric-value metric-info">{m['judicial_confidence']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
