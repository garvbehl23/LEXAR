"""
Bharat Nyaya Console v3 - Production UI
=========================================
Exact replica of the reference judicial evidence analysis system.

Run with: streamlit run bharat_nyaya_v3.py --server.port 8502
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import json
from typing import Optional, Dict, List, Any
import sys
import os
import uuid

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
# Session State Initialization
# ==============================================================================

def init_session_state():
    """Initialize all session state variables"""
    
    defaults = {
        # User settings
        "user_mode": "ADVOCATE",
        "show_settings": False,
        
        # LLM Configuration
        "llm_provider": "Groq",
        "api_key": "",
        "model_name": "llama-3.1-70b-versatile",
        "temperature": 0.3,
        "max_tokens": 2048,
        "system_prompt": "You are a legal research assistant specialized in Indian law.",
        
        # Case context
        "case_type": "constitutional",
        "case_number": "Special Leave Petition (Civil) No. 4920 of 2024",
        "current_query": "",
        
        # Left sidebar navigation
        "active_nav": "fir",
        
        # Analysis results
        "analysis_result": None,
        "evidence_chunks": [],
        "metrics": {
            "evidentiary_strength": 74.2,
            "citation_validity": 91.8,
            "procedural_compliance": 82.5,
            "constitutional_risk": 12.4,
            "judicial_confidence": 96.0,
        },
        
        # Document content
        "document_content": {
            "matter": "",
            "issues": [],
            "reasoning": "",
            "conclusion": "",
            "order": "",
        },
        
        # Statutory context
        "statutory_context": {
            "primary_statute": "The Bharatiya Nyaya Sanhita, 2023",
            "section": "Section 113: Terrorism",
            "supporting_provisions": [
                "Art. 21 - Right to Life",
                "Sec. 65B Evidence Act",
            ],
            "precedents": [
                {"name": "K.S. Puttaswamy v. Union of India", "citation": "2017 (10) SCC 1"},
                {"name": "Maneka Gandhi v. Union of India", "citation": "1978 AIR 597"},
                {"name": "Kesavananda Bharati Case", "citation": "1973 (4) SCC 225"},
            ],
        },
        
        # Proceedings history
        "proceedings_history": [],
        "selected_proceeding_id": None,
        
        # FIR/Case data
        "fir_data": {"fir_number": "", "police_station": "", "date_filed": ""},
        "parties": {"petitioner": "", "respondent": ""},
        "annexures": [],
        "forensic_reports": [],
        "evidence_log": [],
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ==============================================================================
# Case Type Templates
# ==============================================================================

CASE_TEMPLATES = {
    "constitutional": {
        "title": "CONSTITUTIONAL MATTER",
        "jurisdiction": "Article 142 Jurisdiction",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "CIVIL / CRIMINAL APPELLATE JURISDICTION",
        "default_statute": "The Bharatiya Nyaya Sanhita, 2023",
        "default_section": "Section 113: Terrorism",
    },
    "criminal": {
        "title": "CRIMINAL APPEAL", 
        "jurisdiction": "Section 374 CrPC",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "CRIMINAL APPELLATE JURISDICTION",
        "default_statute": "Bharatiya Nyaya Sanhita, 2023",
        "default_section": "Section 103: Murder",
    },
    "civil": {
        "title": "CIVIL APPEAL",
        "jurisdiction": "Order XLI CPC",
        "court": "IN THE SUPREME COURT OF INDIA", 
        "subtitle": "CIVIL APPELLATE JURISDICTION",
        "default_statute": "Code of Civil Procedure, 1908",
        "default_section": "Order XLI Rule 1",
    },
    "writ": {
        "title": "WRIT PETITION",
        "jurisdiction": "Article 32/226",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "ORIGINAL JURISDICTION",
        "default_statute": "Constitution of India",
        "default_section": "Article 32 - Writ Remedies",
    },
    "slp": {
        "title": "SPECIAL LEAVE PETITION",
        "jurisdiction": "Article 136",
        "court": "IN THE SUPREME COURT OF INDIA",
        "subtitle": "SPECIAL LEAVE JURISDICTION",
        "default_statute": "Constitution of India",
        "default_section": "Article 136 - Special Leave",
    },
}


# ==============================================================================
# Custom CSS
# ==============================================================================

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');
    
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
        background: linear-gradient(180deg, #1a0a0a 0%, #0d0505 100%);
    }
    
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Streamlit Component Overrides */
    .stButton > button {
        background: #4d1f1f !important;
        border: 1px solid #6b2a2a !important;
        color: #c9a227 !important;
        font-weight: 600 !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    
    .stButton > button:hover {
        background: #c9a227 !important;
        color: #1a0a0a !important;
        border-color: #c9a227 !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4d1f1f 0%, #6b2a2a 100%) !important;
        border: 1px solid #c9a227 !important;
    }
    
    .stTextArea textarea, .stTextInput > div > div > input {
        background: #120808 !important;
        border: 1px solid #4d1f1f !important;
        color: #c9a227 !important;
        font-family: 'Cormorant Garamond', serif !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #8b7355 !important;
    }
    
    .stSelectbox > div > div {
        background: #120808 !important;
        border: 1px solid #4d1f1f !important;
        color: #c9a227 !important;
    }
    
    [data-testid="stSidebar"] {
        background: #2d1111 !important;
        border-right: 1px solid #4d1f1f;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #f5f5f5;
    }
    
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
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.5rem 0.75rem;
        border-radius: 2px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4d1f1f !important;
        color: #c9a227 !important;
        border-color: #c9a227 !important;
    }
    
    .stExpander {
        background: #1a0a0a;
        border: 1px solid #4d1f1f;
        border-radius: 4px;
    }
    
    .stExpander summary {
        color: #f5f5f5;
    }
    
    .stSlider > div > div > div {
        background: #c9a227 !important;
    }
    
    h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #c9a227 !important;
        font-family: 'Cinzel', serif !important;
    }
    
    p, .stMarkdown p, .stMarkdown {
        color: #f5f5f5;
    }
    
    .stCaption {
        color: #a89f8f !important;
    }
    
    hr {
        border-color: #4d1f1f;
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# Header Component using components.html
# ==============================================================================

def render_header():
    """Render the main header using components.html for proper styling"""
    
    date_str = datetime.now().strftime("%A, %d %B %Y").upper()
    mode = st.session_state.user_mode
    
    adv_style = "background: #c9a227; color: #1a0a0a;" if mode == "ADVOCATE" else "background: transparent; color: #a89f8f;"
    bench_style = "background: #c9a227; color: #1a0a0a;" if mode == "BENCH" else "background: transparent; color: #a89f8f;"
    
    header_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: transparent; }}
            .header {{
                background: linear-gradient(180deg, #2d1111 0%, #1a0a0a 100%);
                border-bottom: 2px solid #4d1f1f;
                padding: 0.75rem 1.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-family: 'Inter', -apple-system, sans-serif;
            }}
            .header-left {{
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            .emblem {{
                font-size: 2rem;
                color: #c9a227;
            }}
            .title {{
                font-family: 'Cinzel', serif;
                font-size: 1.25rem;
                font-weight: 600;
                color: #c9a227;
                letter-spacing: 0.15em;
            }}
            .subtitle {{
                font-size: 0.625rem;
                color: #a89f8f;
                text-transform: uppercase;
                letter-spacing: 0.2em;
            }}
            .header-center {{
                text-align: center;
            }}
            .jurisdiction {{
                font-size: 0.625rem;
                color: #a89f8f;
                text-transform: uppercase;
                letter-spacing: 0.15em;
            }}
            .date {{
                font-family: 'Cinzel', serif;
                font-size: 0.875rem;
                color: #f5f5f5;
                font-weight: 500;
            }}
            .header-right {{
                display: flex;
                align-items: center;
                gap: 1rem;
            }}
            .mode-toggle {{
                display: flex;
                border: 1px solid #6b2a2a;
                border-radius: 4px;
                overflow: hidden;
            }}
            .mode-btn {{
                padding: 0.5rem 1rem;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                border: none;
                cursor: pointer;
            }}
            .seal-btn {{
                background: linear-gradient(135deg, #8b2a2a 0%, #6b1a1a 100%);
                border: 1px solid #c9a227;
                color: #c9a227;
                padding: 0.5rem 1rem;
                font-family: 'Cinzel', serif;
                font-size: 0.7rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="header-left">
                <span class="emblem">🏛️</span>
                <div>
                    <div class="title">BHARAT NYAYA CONSOLE</div>
                    <div class="subtitle">Judicial Evidence Analysis System</div>
                </div>
            </div>
            <div class="header-center">
                <div class="jurisdiction">ORIGINAL JURISDICTION</div>
                <div class="date">{date_str}</div>
            </div>
            <div class="header-right">
                <div class="mode-toggle">
                    <span class="mode-btn" style="{adv_style}">ADVOCATE</span>
                    <span class="mode-btn" style="{bench_style}">BENCH</span>
                </div>
                <button class="seal-btn">
                    📜 SEAL & EXPORT DECREE
                </button>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(header_html, height=70)


# ==============================================================================
# Parchment Document using components.html
# ==============================================================================

def render_parchment():
    """Render the parchment document using components.html"""
    
    template = CASE_TEMPLATES[st.session_state.case_type]
    doc = st.session_state.document_content
    
    # Format issues
    issues_html = ""
    if doc.get("issues") and len(doc["issues"]) > 0:
        for i, issue in enumerate(doc["issues"], 1):
            issues_html += f"<li>{issue}</li>"
    else:
        issues_html = """
        <li>Whether the integrity of hash-values in digital forensic logs constitutes a 'primary fact' under the Indian Evidence Act.</li>
        <li>The scope of judicial review in matters involving automated algorithmic sentencing recommendations.</li>
        """
    
    # Default content
    matter_text = doc.get("matter") or """This petition concerns the interpretation of fundamental rights 
under Part III of the Constitution, specifically the interplay between 
procedural fairness and evidentiary standards in high-stakes 
technological crimes. The Petitioner seeks a writ of certiorari against 
the high court's previous determination."""
    
    reasoning_text = doc.get("reasoning") or """The Court has carefully examined the submissions made by learned counsel 
for both parties. Having regard to the constitutional provisions and the 
precedents cited, the following observations are made..."""

    parchment_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;500;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: transparent; font-family: 'Cormorant Garamond', serif; }}
            .parchment {{
                background: linear-gradient(135deg, #f4e4c1 0%, #e8d4a8 100%);
                border-radius: 4px;
                padding: 2rem 2.5rem;
                color: #2d1f1f;
                position: relative;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 0 100px rgba(139, 69, 19, 0.1);
                min-height: 500px;
            }}
            .parchment::before {{
                content: '';
                position: absolute;
                top: 0; left: 0; right: 0; bottom: 0;
                background: repeating-linear-gradient(0deg, transparent, transparent 28px, rgba(139, 69, 19, 0.03) 28px, rgba(139, 69, 19, 0.03) 29px);
                pointer-events: none;
                border-radius: 4px;
            }}
            .parchment-header {{
                text-align: center;
                margin-bottom: 1.5rem;
                position: relative;
                z-index: 1;
            }}
            .parchment-title {{
                font-family: 'Cinzel', serif;
                font-size: 1.5rem;
                font-weight: 700;
                color: #1a0a0a;
                letter-spacing: 0.1em;
                margin-bottom: 0.5rem;
            }}
            .parchment-subtitle {{
                font-size: 0.875rem;
                font-weight: 600;
                letter-spacing: 0.2em;
                color: #4d1f1f;
                text-transform: uppercase;
            }}
            .parchment-case {{
                font-size: 0.85rem;
                font-style: italic;
                color: #6b4a4a;
                margin-top: 0.5rem;
            }}
            .parchment-section {{
                margin-bottom: 1.5rem;
                position: relative;
                z-index: 1;
            }}
            .section-title {{
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                color: #4d1f1f;
                margin-bottom: 0.5rem;
                padding-bottom: 0.25rem;
                border-bottom: 1px solid rgba(77, 31, 31, 0.2);
            }}
            .section-content {{
                font-size: 1rem;
                line-height: 1.7;
                text-align: justify;
                color: #2d1f1f;
            }}
            .issues-list {{
                padding-left: 1.5rem;
                margin: 0.5rem 0;
            }}
            .issues-list li {{
                margin-bottom: 0.5rem;
                line-height: 1.6;
            }}
            .court-seal {{
                position: absolute;
                bottom: 2rem;
                right: 2rem;
                width: 90px;
                height: 90px;
                border: 3px double #8b4a4a;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #8b4a4a;
                font-family: 'Cinzel', serif;
                font-size: 0.45rem;
                text-align: center;
                font-weight: 600;
                letter-spacing: 0.05em;
                transform: rotate(-15deg);
                opacity: 0.4;
                background: radial-gradient(circle, transparent 30%, rgba(139, 74, 74, 0.05) 100%);
            }}
        </style>
    </head>
    <body>
        <div class="parchment">
            <div class="parchment-header">
                <h1 class="parchment-title">{template['court']}</h1>
                <p class="parchment-subtitle">{template['subtitle']}</p>
                <p class="parchment-case">{st.session_state.case_number}</p>
            </div>
            
            <div class="parchment-section">
                <h4 class="section-title">I. The Matter</h4>
                <p class="section-content">{matter_text}</p>
            </div>
            
            <div class="parchment-section">
                <h4 class="section-title">II. Issues Framed</h4>
                <ol class="issues-list">{issues_html}</ol>
            </div>
            
            <div class="parchment-section">
                <h4 class="section-title">III. Judicial Reasoning</h4>
                <p class="section-content">{reasoning_text}</p>
            </div>
            
            <div class="court-seal">
                SUPREME COURT<br>OF INDIA<br>•<br>OFFICIAL SEAL
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(parchment_html, height=550, scrolling=True)


# ==============================================================================
# Metrics Bar using components.html
# ==============================================================================

def render_metrics_bar():
    """Render the metrics bar using components.html"""
    
    m = st.session_state.metrics
    
    metrics_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: transparent; font-family: 'Inter', sans-serif; }}
            .metrics-bar {{
                background: linear-gradient(180deg, #2d1111 0%, #1a0a0a 100%);
                border-top: 2px solid #4d1f1f;
                padding: 1rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .bench-status {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            .bench-label {{
                font-size: 0.7rem;
                color: #a89f8f;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            }}
            .bench-live {{
                color: #22c55e;
                font-weight: 600;
                font-size: 0.75rem;
            }}
            .metrics-container {{
                display: flex;
                gap: 3rem;
            }}
            .metric-item {{
                text-align: center;
            }}
            .metric-label {{
                font-size: 0.55rem;
                color: #a89f8f;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                margin-bottom: 0.5rem;
            }}
            .metric-gauge {{
                width: 60px;
                height: 30px;
                border: 2px solid #c9a227;
                border-radius: 30px 30px 0 0;
                border-bottom: none;
                position: relative;
                margin: 0 auto 0.25rem;
                background: linear-gradient(to right, #22c55e 0%, #c9a227 50%, #ef4444 100%);
                opacity: 0.3;
            }}
            .metric-value {{
                font-size: 1.25rem;
                font-weight: 700;
            }}
            .gold {{ color: #c9a227; }}
            .danger {{ color: #ef4444; }}
            .info {{ color: #60a5fa; }}
        </style>
    </head>
    <body>
        <div class="metrics-bar">
            <div class="bench-status">
                <span class="bench-label">BENCH STATUS</span>
                <span class="bench-live">LIVE</span>
            </div>
            <div class="metrics-container">
                <div class="metric-item">
                    <div class="metric-label">EVIDENTIARY STRENGTH %</div>
                    <div class="metric-gauge"></div>
                    <div class="metric-value gold">{m['evidentiary_strength']:.1f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">CITATION VALIDITY</div>
                    <div class="metric-gauge"></div>
                    <div class="metric-value gold">{m['citation_validity']:.1f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">PROCEDURAL COMPLIANCE</div>
                    <div class="metric-gauge"></div>
                    <div class="metric-value gold">{m['procedural_compliance']:.1f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">CONSTITUTIONAL RISK INDEX</div>
                    <div class="metric-gauge"></div>
                    <div class="metric-value danger">{m['constitutional_risk']:.1f}</div>
                </div>
                <div class="metric-item">
                    <div class="metric-label">JUDICIAL CONFIDENCE INDEX</div>
                    <div class="metric-gauge"></div>
                    <div class="metric-value info">{m['judicial_confidence']:.1f}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    components.html(metrics_html, height=100)


# ==============================================================================
# Sidebar Components
# ==============================================================================

def render_left_sidebar():
    """Render the left sidebar"""
    
    st.markdown("### 📋 RECORD OF PROCEEDINGS")
    
    nav_items = [
        ("fir", "📋 FIR / CASE NUMBER"),
        ("parties", "👥 PARTIES"),
        ("annexures", "📎 ANNEXURES"),
        ("forensic", "🔬 FORENSIC REPORTS"),
        ("evidence", "📁 EVIDENCE LOG"),
    ]
    
    for nav_id, label in nav_items:
        is_active = st.session_state.active_nav == nav_id
        if st.button(label, key=f"nav_{nav_id}", use_container_width=True, 
                    type="primary" if is_active else "secondary"):
            st.session_state.active_nav = nav_id
            st.rerun()
    
    st.markdown("---")
    
    # Content based on selected nav
    active = st.session_state.active_nav
    
    if active == "fir":
        st.text_input("FIR Number", placeholder="FIR-2024-XXX", key="fir_num_input")
        st.text_input("Police Station", placeholder="Enter PS name", key="ps_input")
        st.date_input("Date Filed", key="fir_date_input")
        
    elif active == "parties":
        st.text_input("Petitioner", placeholder="Name of Petitioner", key="pet_input")
        st.text_input("Respondent", placeholder="Name of Respondent", key="resp_input")
        if st.button("➕ Add Intervener", key="add_interv"):
            st.info("Intervener added")
            
    elif active == "annexures":
        uploaded = st.file_uploader("Upload Annexure", type=["pdf", "docx"], key="annex_up")
        if uploaded:
            st.success(f"Uploaded: {uploaded.name}")
        st.caption("No annexures uploaded yet")
        
    elif active == "forensic":
        uploaded = st.file_uploader("Upload Report", type=["pdf"], key="forensic_up")
        if uploaded:
            st.success(f"Uploaded: {uploaded.name}")
        st.caption("No forensic reports yet")
        
    elif active == "evidence":
        st.caption("Evidence will appear here after analysis")
    
    st.markdown("---")
    st.markdown("**BENCH STATUS**")
    st.markdown("🟢 **LIVE**")


def render_right_sidebar():
    """Render the statutory record sidebar"""
    
    ctx = st.session_state.statutory_context
    template = CASE_TEMPLATES[st.session_state.case_type]
    
    st.markdown("### STATUTORY RECORD")
    
    # Primary Statute Card
    st.markdown(f"""
    <div style="background: #4d1f1f; border: 1px solid #6b2a2a; border-radius: 4px; padding: 0.75rem; margin-bottom: 1rem;">
        <div style="font-size: 0.6rem; color: #c9a227; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.25rem;">PRIMARY STATUTE</div>
        <div style="font-size: 0.9rem; font-weight: 600; color: #f5f5f5;">{template['default_statute']}</div>
        <div style="font-size: 0.75rem; color: #a89f8f; margin-top: 0.25rem;">{template['default_section']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**SUPPORTING PROVISIONS**")
    
    for provision in ctx.get("supporting_provisions", []):
        with st.expander(provision):
            st.write("Detailed provision text would appear here...")
    
    st.markdown("---")
    
    st.markdown("**LANDMARK JUDGMENTS**")
    
    for judgment in ctx.get("precedents", []):
        st.markdown(f"""
        <div style="border-left: 2px solid #c9a227; padding-left: 0.75rem; margin-bottom: 0.75rem;">
            <div style="font-size: 0.8rem; font-weight: 600; color: #f5f5f5;">{judgment['name']}</div>
            <div style="font-size: 0.7rem; color: #a89f8f; font-style: italic;">{judgment['citation']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if st.button("📚 CONSULT AI PRECEDENT ARCHIVE", use_container_width=True):
        st.info("Searching precedent archive...")


def render_settings():
    """Render LLM settings panel"""
    
    st.markdown("---")
    st.markdown("### ⚙️ LLM CONFIGURATION")
    
    provider = st.selectbox(
        "Provider",
        ["Groq", "OpenAI", "Anthropic"],
        index=["Groq", "OpenAI", "Anthropic"].index(st.session_state.llm_provider),
        key="provider_select"
    )
    st.session_state.llm_provider = provider
    
    api_key = st.text_input(
        f"{provider} API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder=f"Enter {provider} API key...",
        key="api_key_input"
    )
    st.session_state.api_key = api_key
    
    if api_key:
        st.success("✓ API Key Set")
    
    # Model selection based on provider
    if provider == "Groq":
        models = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    elif provider == "OpenAI":
        models = ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
    else:
        models = ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
    
    model = st.selectbox("Model", models, key="model_select")
    st.session_state.model_name = model
    
    temp = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.1, key="temp_slider")
    st.session_state.temperature = temp
    
    tokens = st.number_input("Max Tokens", 256, 8192, st.session_state.max_tokens, 256, key="tokens_input")
    st.session_state.max_tokens = tokens
    
    st.markdown("**System Prompt**")
    sys_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=80,
        label_visibility="collapsed",
        key="sys_prompt_input"
    )
    st.session_state.system_prompt = sys_prompt


# ==============================================================================
# Main Content Components
# ==============================================================================

def render_case_tabs():
    """Render case type selector tabs"""
    
    tabs = st.tabs([
        "CONSTITUTIONAL MATTER",
        "CRIMINAL APPEAL", 
        "CIVIL APPEAL",
        "WRIT PETITION",
        "SLP"
    ])
    
    case_types = ["constitutional", "criminal", "civil", "writ", "slp"]
    
    for i, tab in enumerate(tabs):
        with tab:
            if st.session_state.case_type != case_types[i]:
                st.session_state.case_type = case_types[i]
                st.rerun()


def render_query_section():
    """Render the question of law input section"""
    
    template = CASE_TEMPLATES[st.session_state.case_type]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("##### ⚖️ QUESTION OF LAW PRESENTED")
        query = st.text_area(
            "Question",
            value=st.session_state.current_query,
            height=80,
            placeholder="State the issue for judicial determination...",
            label_visibility="collapsed",
            key="main_query"
        )
        st.session_state.current_query = query
    
    with col2:
        st.markdown(f"*{template['jurisdiction']}*")
        
        if st.button("⚖️ PLACE BEFORE BENCH", key="place_btn", use_container_width=True, type="primary"):
            if query.strip():
                with st.spinner("Analyzing with LEXAR..."):
                    run_analysis(query)
                st.rerun()


def render_mode_toggle():
    """Render advocate/bench mode toggle"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("⚖️ ADVOCATE", key="adv_btn", use_container_width=True,
                    type="primary" if st.session_state.user_mode == "ADVOCATE" else "secondary"):
            st.session_state.user_mode = "ADVOCATE"
            st.rerun()
    
    with col2:
        if st.button("🏛️ BENCH", key="bench_btn", use_container_width=True,
                    type="primary" if st.session_state.user_mode == "BENCH" else "secondary"):
            st.session_state.user_mode = "BENCH"
            st.rerun()


# ==============================================================================
# Backend Integration
# ==============================================================================

def run_analysis(query: str) -> bool:
    """Run the full analysis pipeline"""
    
    try:
        lexar = LexarService()
        result = lexar.analyze(
            query=query,
            case_type=st.session_state.case_type
        )
        
        # Store results
        st.session_state.analysis_result = result
        st.session_state.evidence_chunks = result.retrieved_chunks
        st.session_state.metrics = {
            "evidentiary_strength": result.evidentiary_strength,
            "citation_validity": result.citation_validity,
            "procedural_compliance": result.procedural_compliance,
            "constitutional_risk": result.constitutional_risk,
            "judicial_confidence": result.judicial_confidence,
        }
        
        # Update document content
        st.session_state.document_content = {
            "matter": f"This petition concerns {query}. The matter has been examined with reference to the constitutional provisions and statutory framework applicable to the case.",
            "issues": result.issues_framed if result.issues_framed else [
                "Whether the constitutional validity of the impugned provision is sustainable.",
                "The scope of judicial review in the present circumstances."
            ],
            "reasoning": result.judicial_reasoning,
            "conclusion": result.conclusion,
            "order": result.order,
        }
        
        # Update statutory context
        if result.dominant_statute:
            st.session_state.statutory_context["primary_statute"] = result.dominant_statute
        
        if result.citations:
            st.session_state.statutory_context["precedents"] = [
                {"name": c, "citation": ""} for c in result.citations[:3]
            ]
        
        return True
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# Main Application
# ==============================================================================

def main():
    """Main application entry point"""
    
    init_session_state()
    inject_css()
    
    # Header
    render_header()
    
    # Mode toggle row
    mode_col1, mode_col2, mode_col3, mode_col4 = st.columns([6, 1, 1, 2])
    with mode_col3:
        render_mode_toggle()
    
    # Main 3-column layout
    left_col, main_col, right_col = st.columns([1.2, 3, 1.5])
    
    with left_col:
        render_left_sidebar()
        render_settings()
    
    with main_col:
        # Case type tabs
        render_case_tabs()
        
        # Query section
        render_query_section()
        
        # Parchment document
        render_parchment()
    
    with right_col:
        render_right_sidebar()
    
    # Metrics bar at bottom
    render_metrics_bar()


if __name__ == "__main__":
    main()
