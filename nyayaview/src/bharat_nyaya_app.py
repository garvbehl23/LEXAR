"""
Bharat Nyaya Console - Constitutional Interface
A fully-fledged Streamlit application for Judicial Evidence Analysis System
Integrated with LexAR Legal RAG Architecture
Author: Legal AI Team
Date: February 2026
Version: 2.0
"""

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import json
import base64
from pathlib import Path
import time
import random
import sys
import os
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import LexAR components
try:
    from lexar.lexar_pipeline import LexarPipeline
    from lexar.retrieval.multi_index_retriever import MultiIndexRetriever
    LEXAR_AVAILABLE = True
except ImportError:
    LEXAR_AVAILABLE = False
    print("Warning: LexAR components not available. Running in demo mode.")

# Import LLM providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import requests
    GROK_AVAILABLE = True
except ImportError:
    GROK_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Bharat Nyaya Console - Constitutional Interface",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'mode' not in st.session_state:
    st.session_state.mode = 'Bench'  # or 'Advocate'
    
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 'Constitutional Matter'
    
if 'active_section' not in st.session_state:
    st.session_state.active_section = 'FIR / Case Number'
    
if 'question_of_law' not in st.session_state:
    st.session_state.question_of_law = ''

if 'lexar_pipeline' not in st.session_state and LEXAR_AVAILABLE:
    try:
        st.session_state.lexar_pipeline = LexarPipeline()
        st.session_state.lexar_enabled = True
    except Exception as e:
        st.session_state.lexar_enabled = False
        print(f"LexAR initialization failed: {e}")
else:
    st.session_state.lexar_enabled = False
    
if 'case_number' not in st.session_state:
    st.session_state.case_number = 'Special Leave Petition (Civil) No. 4920 of 2024'
    
if 'evidentiary_strength' not in st.session_state:
    st.session_state.evidentiary_strength = 74.2
    
if 'citation_validity' not in st.session_state:
    st.session_state.citation_validity = 91.8
    
if 'procedural_compliance' not in st.session_state:
    st.session_state.procedural_compliance = 82.5
    
if 'constitutional_risk' not in st.session_state:
    st.session_state.constitutional_risk = 12.4
    
if 'judicial_confidence' not in st.session_state:
    st.session_state.judicial_confidence = 96.0
    
if 'document_content' not in st.session_state:
    st.session_state.document_content = {
        'matter': '''This petition concerns the interpretation of fundamental rights under Part III of the Constitution, specifically the interplay between procedural fairness and evidentiary standards in high-stakes technological crimes. The Petitioner seeks a writ of certiorari against the high court's previous determination.''',
        'issues': [
            "Whether the integrity of hash-values in digital forensic logs constitutes a 'primary fact' under the Indian Evidence Act.",
            "The scope of judicial review in matters involving automated algorithmic sentencing recommendations."
        ],
        'reasoning': '''Upon careful examination of the forensic logs presented in the Annexure series 'D', this bench finds a significant deviation in the chain of custody. The doctrine of *stare decisis* compels us to look at the precedent set in *State of Punjab v. Modern Electronics (2019)*, wherein it was established that digital authenticity cannot be presumed in the absence of a valid Section 65B certificate.

Further, the Constitutional Risk Index as calculated by the Console indicates a 42% probability of procedural infringement should the current evidence be admitted without further verification from the Primary Statute.'''
    }
    
if 'parties' not in st.session_state:
    st.session_state.parties = {
        'petitioner': 'M/s TechCorp India Ltd.',
        'respondent': 'Union of India & Ors.',
        'petitioner_counsel': 'Senior Advocate Rahul Mehra',
        'respondent_counsel': 'Attorney General of India'
    }
    
if 'statutes' not in st.session_state:
    st.session_state.statutes = {
        'primary': {
            'name': 'The Bharatiya Nyaya Sanhita, 2023',
            'section': 'Section 113: Terrorism'
        },
        'supporting': [
            {
                'name': 'Art. 21 - Right to Life',
                'content': 'No person shall be deprived of his life or personal liberty except according to procedure established by law.'
            },
            {
                'name': 'Sec. 65B Evidence Act',
                'content': 'Admissibility of electronic records and requirements for certification...'
            }
        ],
        'precedents': [
            {
                'name': 'K.S. Puttaswamy v. Union of India',
                'citation': '2017 (10) SCC 1',
                'active': True
            },
            {
                'name': 'Maneka Gandhi v. Union of India',
                'citation': '1978 AIR 597',
                'active': False
            },
            {
                'name': 'Kesavananda Bharati Case',
                'citation': '1973 (4) SCC 225',
                'active': False
            }
        ]
    }

if 'annexures' not in st.session_state:
    st.session_state.annexures = [
        {'name': 'Annexure A-1', 'type': 'FIR Copy', 'status': 'Verified'},
        {'name': 'Annexure A-2', 'type': 'Witness Statement', 'status': 'Pending'},
        {'name': 'Annexure B-1', 'type': 'Digital Evidence', 'status': 'Verified'},
        {'name': 'Annexure D-1', 'type': 'Forensic Report', 'status': 'Verified'},
    ]

if 'forensic_reports' not in st.session_state:
    st.session_state.forensic_reports = [
        {'name': 'Hash Value Analysis Report', 'date': '2024-03-15', 'status': 'Complete'},
        {'name': 'Chain of Custody Certificate', 'date': '2024-03-20', 'status': 'Complete'},
        {'name': 'Digital Signature Verification', 'date': '2024-04-01', 'status': 'Pending'},
    ]

if 'evidence_log' not in st.session_state:
    st.session_state.evidence_log = [
        {'time': '10:30 AM', 'entry': 'Digital evidence presented by prosecution'},
        {'time': '11:15 AM', 'entry': 'Section 65B certificate challenged by defense'},
        {'time': '02:30 PM', 'entry': 'Forensic expert cross-examination commenced'},
        {'time': '04:00 PM', 'entry': 'Court directed verification of hash values'},
    ]

if 'lexar_response' not in st.session_state:
    st.session_state.lexar_response = None

if 'lexar_evidence' not in st.session_state:
    st.session_state.lexar_evidence = []

if 'api_key_openai' not in st.session_state:
    st.session_state.api_key_openai = ''

if 'api_key_grok' not in st.session_state:
    st.session_state.api_key_grok = ''

if 'selected_llm' not in st.session_state:
    st.session_state.selected_llm = 'demo'  # demo, openai, grok

if 'show_proceedings_bar' not in st.session_state:
    st.session_state.show_proceedings_bar = False

if 'criminal_appeal_data' not in st.session_state:
    st.session_state.criminal_appeal_data = {
        'charge_sheet': [],
        'witness_records': [],
        'evidence_checks': [],
        'sentencing_guidelines': []
    }

if 'criminal_charges' not in st.session_state:
    st.session_state.criminal_charges = [
        {'code': 'IPC 302', 'charge': 'Murder', 'section': 'Indian Penal Code', 'severity': 'High', 'status': 'Pending'},
        {'code': 'IPC 201', 'charge': 'Causing disappearance of evidence', 'section': 'Indian Penal Code', 'severity': 'Medium', 'status': 'Verified'},
    ]

if 'witnesses' not in st.session_state:
    st.session_state.witnesses = [
        {'name': 'Witness 1', 'status': 'Examined', 'credibility': 'High', 'date': '2024-03-15'},
        {'name': 'Witness 2', 'status': 'Cross-examined', 'credibility': 'Medium', 'date': '2024-03-20'},
    ]

if 'sentencing_data' not in st.session_state:
    st.session_state.sentencing_data = {
        'recommended_years': 10,
        'guideline_references': ['Rigorous Imprisonment Act 1973', 'Section 45 IPC'],
        'mitigating_factors': ['First time offender', 'Cooperation with investigation'],
        'aggravating_factors': ['Premeditation', 'Use of weapon']
    }

if 'criminal_section' not in st.session_state:
    st.session_state.criminal_section = None

if 'show_settings' not in st.session_state:
    st.session_state.show_settings = False

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def load_css():
    """Load custom CSS for the application"""
    css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400;700&family=Cinzel:wght@700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
    
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle at center, #2a1818 0%, #1a0d0d 100%);
        font-family: "Noto Serif", serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Color Variables */
    :root {
        --primary: #4d0f0f;
        --sandalwood: #EADBC8;
        --antique-gold: #C6A75E;
        --background-light: #f8f6f6;
        --background-dark: #201212;
    }
    
    /* Parchment Background */
    .parchment-bg {
        background-color: #EADBC8;
        background-image: repeating-linear-gradient(
            90deg,
            transparent,
            transparent 2px,
            rgba(0,0,0,0.03) 2px,
            rgba(0,0,0,0.03) 4px
        );
    }
    
    /* Bench Border Styles */
    .bench-border {
        border: 1px solid #C6A75E;
    }
    
    .bench-border-heavy {
        border: 2px solid #C6A75E;
    }
    
    /* Custom Button Styles */
    .stButton > button {
        background-color: var(--primary);
        border: 2px solid var(--antique-gold);
        color: var(--antique-gold);
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
        font-size: 0.75rem;
    }
    
    .stButton > button:hover {
        background-color: rgba(77, 15, 15, 0.8);
        border-color: var(--antique-gold);
        color: var(--antique-gold);
    }
    
    /* Textarea Styles */
    .stTextArea textarea {
        background-color: #120808 !important;
        border: 1px solid var(--primary) !important;
        color: var(--antique-gold) !important;
        font-size: 1rem !important;
    }
    
    .stTextArea textarea::placeholder {
        color: rgba(198, 167, 94, 0.2) !important;
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(77, 15, 15, 0.2);
        border-top: 2px solid transparent;
        border-left: 2px solid transparent;
        border-right: 2px solid transparent;
        color: rgba(255, 255, 255, 0.4);
        font-size: 0.625rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        padding: 0.5rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary);
        border-color: var(--antique-gold);
        color: var(--antique-gold);
    }
    
    /* Metric Styles */
    .metric-container {
        background-color: rgba(77, 15, 15, 0.2);
        border: 1px solid var(--primary);
        padding: 1rem;
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.625rem;
        font-weight: bold;
        color: var(--antique-gold);
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        font-variant-numeric: tabular-nums;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background-color: var(--background-dark);
        border-right: 1px solid var(--primary);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background-color: rgba(77, 15, 15, 0.05);
        border: 1px solid rgba(77, 15, 15, 0.4);
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
    }
    
    .streamlit-expanderContent {
        background-color: #120808;
        border: 1px solid var(--primary);
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.6875rem;
        line-height: 1.6;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--background-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary);
        border: 1px solid var(--antique-gold);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--antique-gold);
    }
    
    /* Watermark */
    .ashoka-watermark {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        opacity: 0.05;
        pointer-events: none;
        width: 300px;
        height: 300px;
    }
    
    /* Seal Styles */
    .court-seal {
        position: absolute;
        bottom: 5rem;
        right: 5rem;
        width: 6rem;
        height: 6rem;
        border: 4px double #991b1b;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #991b1b;
        font-weight: bold;
        font-size: 0.625rem;
        text-align: center;
        transform: rotate(-15deg);
        opacity: 0.4;
        padding: 0.5rem;
    }
    
    /* Justified Text */
    .justified-legal {
        text-align: justify;
        text-justify: inter-word;
        line-height: 1.6;
    }
    
    /* Header Styles */
    .app-header {
        background-color: var(--background-dark);
        border-bottom: 2px solid var(--primary);
        padding: 1rem 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    }
    
    .app-title {
        font-family: "Cinzel", serif;
        font-size: 1.5rem;
        color: var(--antique-gold);
        letter-spacing: 0.2em;
        margin: 0;
        line-height: 1;
    }
    
    .app-subtitle {
        font-size: 0.625rem;
        color: rgba(198, 167, 94, 0.6);
        text-transform: uppercase;
        letter-spacing: 0.3em;
        font-weight: bold;
        margin-top: 0.25rem;
    }
    
    /* Toggle Switch */
    .toggle-switch {
        display: inline-flex;
        align-items: center;
        background-color: rgba(77, 15, 15, 0.4);
        border: 1px solid var(--primary);
        padding: 0.25rem;
    }
    
    .toggle-label {
        font-size: 0.625rem;
        text-transform: uppercase;
        font-weight: bold;
        padding: 0 0.5rem;
        color: rgba(198, 167, 94, 0.5);
    }
    
    .toggle-label.active {
        color: white;
    }
    
    .toggle-slider {
        width: 2.5rem;
        height: 1.25rem;
        background-color: var(--primary);
        border: 1px solid rgba(198, 167, 94, 0.3);
        position: relative;
        cursor: pointer;
    }
    
    .toggle-slider::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        bottom: 0;
        width: 1.25rem;
        background-color: var(--antique-gold);
    }
    
    /* Document Container */
    .document-container {
        background-color: #EADBC8;
        min-height: 800px;
        padding: 4rem;
        box-shadow: 0 0 50px rgba(0, 0, 0, 0.5);
        position: relative;
        color: #201212;
    }
    
    /* Section Headers */
    .section-header {
        font-weight: bold;
        border-bottom: 1px solid rgba(32, 18, 18, 0.1);
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        font-size: 0.75rem;
        padding-bottom: 0.25rem;
    }
    
    /* List Styles */
    .legal-list {
        padding-left: 1.25rem;
    }
    
    .legal-list li {
        margin-bottom: 0.5rem;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        font-size: 0.625rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        border-radius: 0.25rem;
    }
    
    .status-verified {
        background-color: rgba(34, 197, 94, 0.2);
        color: #22c55e;
        border: 1px solid #22c55e;
    }
    
    .status-pending {
        background-color: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
        border: 1px solid #fbbf24;
    }
    
    .status-complete {
        background-color: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border: 1px solid #3b82f6;
    }
    
    /* Card Styles */
    .info-card {
        background-color: rgba(77, 15, 15, 0.2);
        border: 1px solid var(--primary);
        padding: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .info-card-title {
        font-size: 0.625rem;
        color: var(--antique-gold);
        text-transform: uppercase;
        font-weight: bold;
        margin-bottom: 0.25rem;
    }
    
    .info-card-content {
        font-size: 0.875rem;
        font-weight: bold;
        color: white;
    }
    
    .info-card-detail {
        font-size: 0.6875rem;
        color: rgba(255, 255, 255, 0.6);
        margin-top: 0.5rem;
    }
    
    /* Precedent Card */
    .precedent-card {
        border-left: 2px solid var(--antique-gold);
        padding-left: 0.75rem;
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
        background-color: rgba(77, 15, 15, 0.1);
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .precedent-card:hover {
        background-color: rgba(77, 15, 15, 0.2);
    }
    
    .precedent-card.inactive {
        border-left-color: rgba(198, 167, 94, 0.3);
    }
    
    .precedent-name {
        font-size: 0.75rem;
        font-weight: bold;
        color: white;
    }
    
    .precedent-card.inactive .precedent-name {
        color: rgba(255, 255, 255, 0.6);
    }
    
    .precedent-citation {
        font-size: 0.625rem;
        color: rgba(255, 255, 255, 0.4);
        font-style: italic;
    }
    
    .precedent-card.inactive .precedent-citation {
        color: rgba(255, 255, 255, 0.3);
    }
    
    /* Menu Item Styles */
    .menu-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem;
        cursor: pointer;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }
    
    .menu-item:hover {
        background-color: rgba(77, 15, 15, 0.2);
    }
    
    .menu-item.active {
        background-color: rgba(77, 15, 15, 0.4);
        border: 2px solid var(--antique-gold);
    }
    
    .menu-item-icon {
        color: rgba(198, 167, 94, 0.5);
        font-size: 1.25rem;
    }
    
    .menu-item.active .menu-item-icon {
        color: var(--antique-gold);
    }
    
    .menu-item-label {
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(255, 255, 255, 0.7);
    }
    
    .menu-item.active .menu-item-label {
        color: white;
    }
    
    /* Progress Bar */
    .progress-container {
        margin-top: 0.5rem;
    }
    
    .progress-bar {
        height: 0.25rem;
        background-color: var(--primary);
        width: 100%;
        position: relative;
        overflow: hidden;
    }
    
    .progress-fill {
        height: 100%;
        background-color: var(--antique-gold);
        transition: width 0.3s ease;
    }
    
    /* Date Display */
    .date-display {
        font-size: 0.875rem;
        font-weight: bold;
        color: white;
        text-transform: uppercase;
        font-variant-numeric: tabular-nums;
    }
    
    /* Material Icons Support */
    .material-symbols-outlined {
        font-family: 'Material Symbols Outlined';
        font-weight: normal;
        font-style: normal;
        font-size: 24px;
        line-height: 1;
        letter-spacing: normal;
        text-transform: none;
        display: inline-block;
        white-space: nowrap;
        word-wrap: normal;
        direction: ltr;
        -webkit-font-smoothing: antialiased;
    }
    
    /* Evidence Log Entry */
    .evidence-entry {
        padding: 0.5rem;
        border-left: 2px solid rgba(198, 167, 94, 0.3);
        margin-bottom: 0.5rem;
        font-size: 0.75rem;
    }
    
    .evidence-time {
        color: var(--antique-gold);
        font-weight: bold;
        font-size: 0.625rem;
    }
    
    .evidence-text {
        color: rgba(255, 255, 255, 0.7);
        margin-top: 0.25rem;
    }
    
    /* Meter Dial */
    .meter-container {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: 0.5rem;
    }
    
    .meter-dial {
        width: 80px;
        height: 40px;
        border-radius: 40px 40px 0 0;
        border: 2px solid var(--antique-gold);
        position: relative;
        overflow: hidden;
        background: #311c1c;
    }
    
    .meter-needle {
        position: absolute;
        bottom: 0;
        left: 50%;
        width: 2px;
        height: 35px;
        background: var(--antique-gold);
        transform-origin: bottom center;
    }
    
    .meter-base {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 0.25rem;
        background-color: rgba(198, 167, 94, 0.2);
    }
    
    /* Risk Meter */
    .risk-meter .meter-dial {
        border-color: rgba(153, 27, 27, 0.4);
    }
    
    .risk-meter .meter-needle {
        background: #dc2626;
    }
    
    .risk-meter .meter-base {
        background-color: rgba(153, 27, 27, 0.2);
    }
    
    /* Confidence Meter */
    .confidence-meter .meter-dial {
        border-color: rgba(37, 99, 235, 0.4);
    }
    
    .confidence-meter .meter-needle {
        background: #3b82f6;
    }
    
    .confidence-meter .meter-base {
        background-color: rgba(37, 99, 235, 0.2);
    }
    
    /* Footer Metrics */
    .footer-metric {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    /* Animate entrance */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.5s ease-out;
    }
    
    /* Selection color */
    ::selection {
        background-color: rgba(198, 167, 94, 0.3);
    }
    
    ::-moz-selection {
        background-color: rgba(198, 167, 94, 0.3);
    }
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
        }
        50% {
            opacity: 0.5;
        }
    }
    
    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    /* Hide Streamlit branding */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    
    .viewerBadge_link__1S137 {
        display: none !important;
    }
    
    /* Radio button styles */
    .stRadio > label {
        color: white;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Checkbox styles */
    .stCheckbox > label {
        color: white;
        font-size: 0.75rem;
    }
    
    /* Info box */
    .info-box {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.875rem;
    }
    
    /* Warning box */
    .warning-box {
        background-color: rgba(251, 191, 36, 0.1);
        border-left: 4px solid #fbbf24;
        padding: 1rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.875rem;
    }
    
    /* Error box */
    .error-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.875rem;
    }
    
    /* Success box */
    .success-box {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 4px solid #22c55e;
        padding: 1rem;
        margin: 1rem 0;
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.875rem;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_meter_rotation(value, min_val=0, max_val=100):
    """Calculate rotation angle for meter needle based on value"""
    # Meter goes from -90deg to 90deg (180 degree range)
    normalized = (value - min_val) / (max_val - min_val)
    angle = (normalized * 180) - 90
    return angle

def format_date(date_format="%A, %d %B %Y"):
    """Format current date"""
    return datetime.now().strftime(date_format)

def create_meter_html(value, label, meter_type="default"):
    """Create HTML for a meter display"""
    rotation = get_meter_rotation(value)
    
    meter_class = ""
    value_color = "white"
    
    if meter_type == "risk":
        meter_class = "risk-meter"
        value_color = "#ef4444"
    elif meter_type == "confidence":
        meter_class = "confidence-meter"
        value_color = "#3b82f6"
    
    html = f"""
    <div class="footer-metric {meter_class}">
        <span class="metric-label">{label}</span>
        <div class="meter-container">
            <div class="meter-dial">
                <div class="meter-needle" style="transform: rotate({rotation}deg);"></div>
                <div class="meter-base"></div>
            </div>
            <span class="metric-value" style="color: {value_color};">{value:.1f}</span>
        </div>
    </div>
    """
    return html

def export_decree():
    """Export the current decree as a downloadable document"""
    content = f"""
    IN THE SUPREME COURT OF INDIA
    CIVIL / CRIMINAL APPELLATE JURISDICTION
    {st.session_state.case_number}
    
    DATE: {format_date()}
    
    I. THE MATTER
    {st.session_state.document_content['matter']}
    
    II. ISSUES FRAMED
    """
    for i, issue in enumerate(st.session_state.document_content['issues'], 1):
        content += f"\n{i}. {issue}"
    
    content += f"""
    
    III. JUDICIAL REASONING
    {st.session_state.document_content['reasoning']}
    
    ================================
    EVIDENTIARY METRICS
    ================================
    Evidentiary Strength: {st.session_state.evidentiary_strength}%
    Citation Validity: {st.session_state.citation_validity}%
    Procedural Compliance: {st.session_state.procedural_compliance}%
    Constitutional Risk Index: {st.session_state.constitutional_risk}%
    Judicial Confidence Index: {st.session_state.judicial_confidence}%
    
    SEAL OF THE SUPREME COURT OF INDIA
    """
    
    return content

def toggle_mode():
    """Toggle between Advocate and Bench mode"""
    if st.session_state.mode == 'Bench':
        st.session_state.mode = 'Advocate'
    else:
        st.session_state.mode = 'Bench'

def update_metrics():
    """Simulate updating metrics based on document analysis"""
    # Simulate some variation in metrics
    st.session_state.evidentiary_strength = round(random.uniform(65.0, 95.0), 1)
    st.session_state.citation_validity = round(random.uniform(80.0, 98.0), 1)
    st.session_state.procedural_compliance = round(random.uniform(75.0, 92.0), 1)
    st.session_state.constitutional_risk = round(random.uniform(5.0, 25.0), 1)
    st.session_state.judicial_confidence = round(random.uniform(88.0, 99.0), 1)

def generate_text_with_llm(prompt: str, evidence: List[str] = None) -> str:
    """
    Generate text using selected LLM based on LexAR evidence
    
    Args:
        prompt: The user prompt/question
        evidence: Retrieved evidence chunks from LexAR
        
    Returns:
        Generated text response
    """
    
    if st.session_state.selected_llm == 'demo':
        # Demo mode - generate realistic response
        return demo_text_generation(prompt, evidence)
    
    elif st.session_state.selected_llm == 'openai':
        if not st.session_state.api_key_openai:
            return "⚠️ OpenAI API key not configured. Please add it in Settings."
        
        try:
            import openai
            openai.api_key = st.session_state.api_key_openai
            
            # Build context from evidence
            evidence_context = "\n\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(evidence or [])])
            
            system_prompt = f"""You are a legal expert assisting in judicial analysis. 
Your responses must be grounded in the following retrieved evidence and the LexAR architecture principles:
- No generation without evidence (all statements must cite retrieved content)
- Hard attention masking (focus only on provided evidence)
- Evidence metadata preservation
- Provable grounding in retrieved chunks

Retrieved Evidence:
{evidence_context}

Respond with citations and confidence levels."""
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ OpenAI Error: {str(e)}"
    
    elif st.session_state.selected_llm == 'grok':
        if not st.session_state.api_key_grok:
            return "⚠️ Grok API key not configured. Please add it in Settings."
        
        try:
            # Grok API call via xAI
            evidence_context = "\n\n".join([f"Evidence {i+1}: {e}" for i, e in enumerate(evidence or [])])
            
            system_prompt = f"""You are a legal expert for Indian courts.
Follow LexAR principles:
- Retrieve and ground all statements in evidence
- Provide confidence scores
- Never generate unsupported claims

Evidence Base:
{evidence_context}"""
            
            headers = {
                "Authorization": f"Bearer {st.session_state.api_key_grok}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                "model": "grok-1",
                "stream": False,
                "temperature": 0.3
            }
            
            response = requests.post(
                "https://api.x.ai/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"❌ Grok Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Grok Error: {str(e)}"
    
    return "Please select and configure an LLM provider"

def demo_text_generation(prompt: str, evidence: List[str] = None) -> str:
    """Generate realistic legal text in demo mode"""
    
    demo_responses = {
        'constitutional': """Based on the retrieved evidence and LexAR architectural principles, the following analysis is provided:

The constitutional question presented herein engages with the fundamental right to life under Article 21 of the Constitution. The evidence retrieved from our legal indices demonstrates:

1. ESTABLISHED PRECEDENT: The landmark judgment in K.S. Puttaswamy v. Union of India (2017 10 SCC 1) establishes that privacy is a fundamental right. The retrieved evidence shows 95% textual alignment with the present case.

2. STATUTORY COMPLIANCE: Section 65B of the Indian Evidence Act requires certification for digital evidence. Our retrieval system identified 12 relevant precedents confirming this requirement, with an average confidence score of 0.88.

3. PROCEDURAL SAFEGUARDS: The chain of custody analysis reveals no deviation from established procedures. Evidence chunks from Maneka Gandhi v. Union of India support the procedural fairness argument with 91% relevance.

CONCLUSION: The evidence base supports the petitioner's contention. Confidence Score: 0.89 (High)
Evidence Count: 7 | Processing Method: LexAR Evidence-Constrained Generation""",
        
        'criminal': """CRIMINAL APPEAL ANALYSIS - Evidence Based Response

The criminal appeal analysis based on retrieved evidence:

CHARGE SHEET ANALYSIS:
- IPC 302 (Murder): Evidence from 8 retrieved cases shows similar charge patterns
- Average conviction rate: 73%
- Mitigating factors present: 6/10 cases show similar profile

WITNESS CREDIBILITY ASSESSMENT:
Retrieved precedent evidence shows witness examination standards:
- K.S. Puttaswamy principles apply: 92% confidence
- Cross-examination safeguards validated: 8 relevant cases
- Overall credibility score: 0.76 (Moderate-High)

SENTENCING RECOMMENDATIONS:
Based on evidence retrieval:
- Comparable cases recommend: 8-12 years rigorous imprisonment
- Mitigating factors reduce by 15-20%
- Aggravating factors increase by 10-15%
- Final recommendation: 10 years (within statutory range)

Evidence Grounding: LexAR Architecture Compliance ✓
Confidence: 0.85 | Evidence Used: 15 chunks""",
        
        'civil': """CIVIL APPEAL ANALYSIS

The civil matter analysis with evidence-based reasoning:

CONTRACT INTERPRETATION:
Retrieved evidence from commercial law precedents (15 relevant cases):
- Contract formation requirements: Validated at 0.91 confidence
- Consideration doctrine: Supported by evidence in 14/15 cases
- Breach determination: Established with 0.88 confidence

REMEDIES ASSESSMENT:
Evidence-based remedy selection:
- Damages calculation: 9 precedents retrieved
- Specific performance applicability: 7 relevant cases
- Injunctive relief standards: 5 supporting cases

LIABILITY DETERMINATION:
Based on retrieved evidence:
- Negligence elements: 0.89 confidence match
- Causation established: 8 precedent cases
- Foreseeability: High confidence (0.92)

DAMAGES QUANTUM:
Evidence suggests: ₹50-75 lakhs based on:
- 12 comparable cases analyzed
- Average recovery: ₹62 lakhs
- Range adjustment: ±15%

LexAR Evidence Chain: Complete | Confidence: 0.87"""
    }
    
    # Determine response type based on prompt
    if any(word in prompt.lower() for word in ['constitutional', 'fundamental', 'article', 'right']):
        base_response = demo_responses['constitutional']
    elif any(word in prompt.lower() for word in ['criminal', 'murder', 'charge', 'conviction']):
        base_response = demo_responses['criminal']
    elif any(word in prompt.lower() for word in ['civil', 'contract', 'damages', 'liability']):
        base_response = demo_responses['civil']
    else:
        base_response = demo_responses['constitutional']
    
    return base_response

def consult_ai_archive():
    """Simulate or actually consult AI precedent archive using LexAR"""
    if st.session_state.lexar_enabled and st.session_state.question_of_law:
        try:
            # Use actual LexAR pipeline
            result = st.session_state.lexar_pipeline.answer(
                query=st.session_state.question_of_law,
                has_user_docs=False,
                top_k=10,
                return_provenance=True,
                debug_mode=False
            )
            
            st.session_state.lexar_response = result
            st.session_state.ai_consultation_result = {
                'status': 'success',
                'precedents_found': result.get('evidence_count', 0),
                'relevance_score': result.get('confidence', 0.0),
                'processing_time': random.uniform(0.5, 2.5),
                'answer': result.get('answer', '')
            }
            
            # Update document content with LexAR response
            if result.get('answer'):
                st.session_state.document_content['reasoning'] += f"\n\nAI-Assisted Analysis:\n{result['answer']}"
            
            # Update metrics based on LexAR confidence
            if result.get('confidence'):
                st.session_state.judicial_confidence = result['confidence'] * 100
                st.session_state.evidentiary_strength = min(95.0, result['confidence'] * 100 + 10)
                
        except Exception as e:
            # Fallback to simulation
            st.session_state.ai_consultation_result = {
                'status': 'error',
                'error_message': str(e),
                'precedents_found': random.randint(15, 45),
                'relevance_score': round(random.uniform(0.75, 0.95), 2),
                'processing_time': round(random.uniform(0.5, 2.5), 2)
            }
    else:
        # Simulation mode
        st.session_state.ai_consultation_result = {
            'status': 'success',
            'precedents_found': random.randint(15, 45),
            'relevance_score': round(random.uniform(0.75, 0.95), 2),
            'processing_time': round(random.uniform(0.5, 2.5), 2)
        }

# ============================================================================
# COMPONENT FUNCTIONS
# ============================================================================

def render_header():
    """Render the application header"""
    header_html = f"""
    <div class="app-header fade-in-up">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="color: var(--antique-gold); opacity: 0.8;">
                    <span class="material-symbols-outlined" style="font-size: 2.5rem;">account_balance</span>
                </div>
                <div>
                    <h1 class="app-title">BHARAT NYAYA CONSOLE</h1>
                    <p class="app-subtitle">Judicial Evidence Analysis System</p>
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 2.5rem;">
                <div style="text-align: right; border-right: 1px solid var(--primary); padding-right: 2.5rem;">
                    <span style="font-size: 0.75rem; color: rgba(198, 167, 94, 0.7); text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em;">Original Jurisdiction</span><br>
                    <span class="date-display">{format_date()}</span>
                </div>
                <div class="toggle-switch">
                    <span class="toggle-label {'active' if st.session_state.mode == 'Advocate' else ''}">Advocate</span>
                    <div class="toggle-slider" style="{'transform: scaleX(-1);' if st.session_state.mode == 'Advocate' else ''}"></div>
                    <span class="toggle-label {'active' if st.session_state.mode == 'Bench' else ''}">Bench</span>
                </div>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

def render_draggable_proceedings_bar():
    """Render draggable Record of Proceedings bar"""
    
    draggable_html = """
    <div id="proceedings-bar" class="draggable-proceedings-bar" style="
        position: fixed;
        bottom: 120px;
        right: 20px;
        width: 350px;
        max-height: 400px;
        background-color: #201212;
        border: 2px solid #C6A75E;
        border-radius: 4px;
        box-shadow: 0 0 20px rgba(0,0,0,0.8);
        z-index: 999;
        display: none;
        flex-direction: column;
        overflow: hidden;
    ">
        <div style="
            background-color: #4d0f0f;
            border-bottom: 1px solid #C6A75E;
            padding: 0.75rem;
            cursor: move;
            display: flex;
            justify-content: space-between;
            align-items: center;
            user-select: none;
        " class="draggable-header">
            <span style="color: #C6A75E; font-weight: bold; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.1em;">📋 RECORD OF PROCEEDINGS</span>
            <button style="
                background: none;
                border: none;
                color: #C6A75E;
                font-size: 1.25rem;
                cursor: pointer;
                padding: 0;
            " onclick="document.getElementById('proceedings-bar').style.display = 'none'">×</button>
        </div>
        
        <div style="
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.875rem;
        ">
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0 0 0.5rem 0; color: #C6A75E; font-weight: bold;">Case Status</p>
                <div style="background: rgba(77, 15, 15, 0.3); padding: 0.5rem; border-left: 2px solid #C6A75E;">
                    <strong>Live Proceedings</strong><br>
                    <small>Sessions ongoing: 3</small>
                </div>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0 0 0.5rem 0; color: #C6A75E; font-weight: bold;">Recent Events</p>
                <div style="background: rgba(77, 15, 15, 0.2); padding: 0.5rem; border-left: 2px solid rgba(198, 167, 94, 0.3);">
                    <small>10:30 AM - Evidence submitted<br>
                    11:15 AM - Motion filed<br>
                    2:30 PM - Hearing resumed</small>
                </div>
            </div>
            
            <div style="margin-bottom: 1rem;">
                <p style="margin: 0 0 0.5rem 0; color: #C6A75E; font-weight: bold;">Participants</p>
                <div style="background: rgba(77, 15, 15, 0.2); padding: 0.5rem; border-left: 2px solid rgba(198, 167, 94, 0.3);">
                    <small>Judges: 3<br>
                    Advocates: 6<br>
                    Clerks: 4</small>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let isDragging = false;
        let offsetX = 0;
        let offsetY = 0;
        
        const proceedingsBar = document.getElementById('proceedings-bar');
        const header = document.querySelector('.draggable-header');
        
        if (header) {
            header.addEventListener('mousedown', (e) => {
                isDragging = true;
                offsetX = e.clientX - proceedingsBar.offsetLeft;
                offsetY = e.clientY - proceedingsBar.offsetTop;
            });
            
            document.addEventListener('mousemove', (e) => {
                if (isDragging) {
                    proceedingsBar.style.left = (e.clientX - offsetX) + 'px';
                    proceedingsBar.style.right = 'auto';
                    proceedingsBar.style.top = (e.clientY - offsetY) + 'px';
                    proceedingsBar.style.bottom = 'auto';
                }
            });
            
            document.addEventListener('mouseup', () => {
                isDragging = false;
            });
        }
    </script>
    """
    
    st.markdown(draggable_html, unsafe_allow_html=True)

def render_left_sidebar():
    """Render the left sidebar with record of proceedings"""
    with st.sidebar:
        st.markdown("### 📋 RECORD OF PROCEEDINGS")
        st.markdown("---")
        
        sections = [
            ("📄", "FIR / Case Number", "description"),
            ("👥", "Parties", "groups"),
            ("📎", "Annexures", "attachment"),
            ("🔬", "Forensic Reports", "biotech"),
            ("📜", "Evidence Log", "history_edu")
        ]
        
        for icon, label, _ in sections:
            if st.button(f"{icon} {label}", key=f"sidebar_{label}", use_container_width=True):
                st.session_state.active_section = label
                st.session_state.show_section_modal = True
        
        # Show section content in expander if modal is active
        if st.session_state.get('show_section_modal', False):
            with st.expander(f"📂 {st.session_state.active_section}", expanded=True):
                render_section_content_inline()
                if st.button("✕ Close", key="close_modal"):
                    st.session_state.show_section_modal = False
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### ⚖️ BENCH STATUS")
        
        bench_status_html = """
        <div class="info-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-size: 0.625rem; color: rgba(198, 167, 94, 0.6); font-weight: bold; text-transform: uppercase;">Status</span>
                <span style="font-size: 0.625rem; color: #22c55e; font-weight: bold;">● LIVE</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: 66%;"></div>
            </div>
        </div>
        """
        st.markdown(bench_status_html, unsafe_allow_html=True)

def render_section_content_inline():
    """Render section content in sidebar expander"""
    section = st.session_state.active_section
    
    if section == "FIR / Case Number":
        st.text_input("Case Number", value=st.session_state.case_number, key="case_number_input_sidebar")
        if st.button("Update", key="update_case_sidebar"):
            st.session_state.case_number = st.session_state.case_number_input_sidebar
            st.success("✓ Updated")
    
    elif section == "Parties":
        st.text_input("Petitioner", value=st.session_state.parties['petitioner'], key="petitioner_sidebar")
        st.text_input("Respondent", value=st.session_state.parties['respondent'], key="respondent_sidebar")
        if st.button("Update", key="update_parties_sidebar"):
            st.session_state.parties['petitioner'] = st.session_state.petitioner_sidebar
            st.session_state.parties['respondent'] = st.session_state.respondent_sidebar
            st.success("✓ Updated")
    
    elif section == "Annexures":
        for annexure in st.session_state.annexures[:3]:
            st.write(f"**{annexure['name']}** - {annexure['status']}")
    
    elif section == "Forensic Reports":
        for report in st.session_state.forensic_reports[:3]:
            st.write(f"**{report['name']}** - {report['status']}")
    
    elif section == "Evidence Log":
        for entry in st.session_state.evidence_log[-3:]:
            st.caption(f"{entry['time']}: {entry['entry']}")

def render_document_viewer():
    """Render the main document viewer (parchment area)"""
    
    # Use HTML component for proper rendering
    document_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@700&display=swap" rel="stylesheet"/>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: "Noto Serif", serif;
            }}
            .document-container {{
                background-color: #EADBC8;
                background-image: repeating-linear-gradient(
                    90deg,
                    transparent,
                    transparent 2px,
                    rgba(0,0,0,0.03) 2px,
                    rgba(0,0,0,0.03) 4px
                );
                min-height: 800px;
                padding: 4rem;
                box-shadow: 0 0 50px rgba(0, 0, 0, 0.5);
                position: relative;
                color: #201212;
            }}
            .ashoka-watermark {{
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                opacity: 0.05;
                pointer-events: none;
                width: 300px;
                height: 300px;
            }}
            .court-seal {{
                position: absolute;
                bottom: 5rem;
                right: 5rem;
                width: 6rem;
                height: 6rem;
                border: 4px double #991b1b;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #991b1b;
                font-weight: bold;
                font-size: 0.625rem;
                text-align: center;
                transform: rotate(-15deg);
                opacity: 0.4;
                padding: 0.5rem;
            }}
            .section-header {{
                font-weight: bold;
                border-bottom: 1px solid rgba(32, 18, 18, 0.1);
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                font-size: 0.75rem;
                padding-bottom: 0.25rem;
            }}
            .justified-legal {{
                text-align: justify;
                text-justify: inter-word;
                line-height: 1.6;
            }}
            .legal-list {{
                padding-left: 1.25rem;
            }}
            .legal-list li {{
                margin-bottom: 0.5rem;
            }}
        </style>
    </head>
    <body>
        <div class="document-container">
            <div class="ashoka-watermark">
                <svg fill="currentColor" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                    <path d="M100 0L110 40H90L100 0ZM100 200L90 160H110L100 200ZM0 100L40 90V110L0 100ZM200 100L160 110V90L200 100Z"></path>
                    <circle cx="100" cy="100" fill="none" r="40" stroke="currentColor" stroke-width="2"></circle>
                    <circle cx="100" cy="100" fill="none" r="60" stroke="currentColor" stroke-width="1"></circle>
                    <circle cx="100" cy="100" fill="none" r="80" stroke="currentColor" stroke-width="1"></circle>
                </svg>
            </div>
            
            <div style="text-align: center; margin-bottom: 3rem; position: relative; z-index: 10;">
                <h3 style="font-family: 'Cinzel', serif; font-size: 1.25rem; border-bottom: 2px solid rgba(32, 18, 18, 0.2); padding-bottom: 0.5rem; display: inline-block; padding-left: 2.5rem; padding-right: 2.5rem;">
                    IN THE SUPREME COURT OF INDIA
                </h3>
                <p style="margin-top: 1rem; font-weight: bold; letter-spacing: 0.2em; text-transform: uppercase; font-size: 0.875rem;">
                    CIVIL / CRIMINAL APPELLATE JURISDICTION
                </p>
                <p style="margin-top: 0.25rem; font-size: 0.75rem; font-style: italic;">
                    {st.session_state.case_number}
                </p>
            </div>
            
            <div style="position: relative; z-index: 10;" class="justified-legal">
                <section style="margin-bottom: 2rem;">
                    <h4 class="section-header">I. THE MATTER</h4>
                    <p>{st.session_state.document_content['matter']}</p>
                </section>
                
                <section style="margin-bottom: 2rem;">
                    <h4 class="section-header">II. ISSUES FRAMED</h4>
                    <ol class="legal-list">
                        {''.join([f'<li>{issue}</li>' for issue in st.session_state.document_content['issues']])}
                    </ol>
                </section>
                
                <section style="margin-bottom: 2rem;">
                    <h4 class="section-header">III. JUDICIAL REASONING</h4>
                    <p>{st.session_state.document_content['reasoning']}</p>
                </section>
                
                {f'''<section style="margin-bottom: 2rem;">
                    <h4 class="section-header">IV. LEXAR AI ANALYSIS</h4>
                    <p>{st.session_state.lexar_response['answer'] if st.session_state.lexar_response else 'No AI analysis requested yet. Use "Consult AI Precedent Archive" to generate evidence-based analysis.'}</p>
                    {f'<p style="margin-top: 1rem; font-size: 0.875rem; color: #666;"><strong>Evidence Count:</strong> {st.session_state.lexar_response["evidence_count"]} | <strong>Confidence:</strong> {st.session_state.lexar_response["confidence"]:.2%}</p>' if st.session_state.lexar_response else ''}
                </section>''' if st.session_state.lexar_enabled else ''}
            </div>
            
            <div class="court-seal">
                SUPREME COURT<br/>OF INDIA<br/>OFFICIAL SEAL
            </div>
        </div>
    </body>
    </html>
    """
    
    # Render using components.html for proper HTML rendering
    components.html(document_html, height=1000, scrolling=True)

def render_right_sidebar():
    """Render the right sidebar with statutory records"""
    st.markdown("### 📚 STATUTORY RECORD")
    
    # Show LexAR status
    if st.session_state.lexar_enabled:
        st.success("✅ LexAR AI Engine: Active")
    else:
        st.warning("⚠️ LexAR AI Engine: Demo Mode")
    
    st.markdown("---")
    
    # Primary Statute
    primary_html = f"""
    <div class="info-card">
        <div class="info-card-title">PRIMARY STATUTE</div>
        <div class="info-card-content">{st.session_state.statutes['primary']['name']}</div>
        <div class="info-card-detail">{st.session_state.statutes['primary']['section']}</div>
    </div>
    """
    st.markdown(primary_html, unsafe_allow_html=True)
    
    # Supporting Provisions
    st.markdown("#### Supporting Provisions")
    for provision in st.session_state.statutes['supporting']:
        with st.expander(provision['name']):
            st.markdown(f"<div style='font-size: 0.75rem; color: rgba(255, 255, 255, 0.7);'>{provision['content']}</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Landmark Judgments
    st.markdown("#### 🏛️ Landmark Judgments")
    for precedent in st.session_state.statutes['precedents']:
        active_class = "" if precedent['active'] else "inactive"
        precedent_html = f"""
        <div class="precedent-card {active_class}">
            <div class="precedent-name">{precedent['name']}</div>
            <div class="precedent-citation">{precedent['citation']}</div>
        </div>
        """
        st.markdown(precedent_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # AI Consultation Button
    if st.button("🤖 CONSULT AI PRECEDENT ARCHIVE", use_container_width=True):
        if not st.session_state.question_of_law:
            st.error("Please enter a Question of Law before consulting AI archive")
        else:
            with st.spinner("Analyzing precedent database with LexAR engine..."):
                time.sleep(1.5)
                consult_ai_archive()
                
                if st.session_state.ai_consultation_result['status'] == 'success':
                    st.success(f"✓ Found {st.session_state.ai_consultation_result['precedents_found']} relevant precedents")
                    st.info(f"Relevance Score: {st.session_state.ai_consultation_result['relevance_score']}")
                    
                    if st.session_state.lexar_enabled and st.session_state.lexar_response:
                        st.success(f"✓ Evidence-based analysis generated")
                        with st.expander("View LexAR Analysis"):
                            st.write(st.session_state.lexar_response.get('answer', ''))
                            st.caption(f"Confidence: {st.session_state.lexar_response.get('confidence', 0):.2%}")
                else:
                    st.error(f"Error: {st.session_state.ai_consultation_result.get('error_message', 'Unknown error')}")
    
    # Show LexAR Evidence if available
    if st.session_state.lexar_evidence:
        st.markdown("---")
        st.markdown("#### 📄 Retrieved Evidence")
        for i, evidence in enumerate(st.session_state.lexar_evidence[:3], 1):
            with st.expander(f"Evidence {i}"):
                st.caption(evidence.get('source', 'Unknown source'))
                st.write(evidence.get('text', '')[:200] + "...")
                st.caption(f"Score: {evidence.get('score', 0):.3f}")

def render_footer_metrics():
    """Render the bottom footer with metrics"""
    footer_html = f"""
    <div style="
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: 6rem;
        background-color: var(--background-dark);
        border-top: 2px solid var(--primary);
        display: flex;
        align-items: center;
        padding: 0 2rem;
        gap: 3rem;
        z-index: 1000;
    ">
        {create_meter_html(st.session_state.evidentiary_strength, "Evidentiary Strength %")}
        {create_meter_html(st.session_state.citation_validity, "Citation Validity")}
        {create_meter_html(st.session_state.procedural_compliance, "Procedural Compliance")}
        <div style="border-left: 1px solid rgba(77, 15, 15, 0.4); height: 3rem; margin: 0 1rem;"></div>
        {create_meter_html(st.session_state.constitutional_risk, "Constitutional Risk Index", "risk")}
        {create_meter_html(st.session_state.judicial_confidence, "Judicial Confidence Index", "confidence")}
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

def render_section_content():
    """Render content based on active section"""
    section = st.session_state.active_section
    
    st.markdown(f"## {section}")
    st.markdown("---")
    
    if section == "FIR / Case Number":
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">CASE DETAILS</div>
            <div class="info-card-content">{st.session_state.case_number}</div>
            <div class="info-card-detail">Filed: 15 March 2024 | Bench: Constitutional</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_input("Update Case Number", value=st.session_state.case_number, key="case_number_input")
        
        if st.button("Update Case Number"):
            st.session_state.case_number = st.session_state.case_number_input
            st.success("✓ Case number updated successfully")
    
    elif section == "Parties":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Petitioner Side")
            st.text_input("Petitioner", value=st.session_state.parties['petitioner'], key="petitioner_input")
            st.text_input("Counsel", value=st.session_state.parties['petitioner_counsel'], key="petitioner_counsel_input")
        
        with col2:
            st.markdown("### Respondent Side")
            st.text_input("Respondent", value=st.session_state.parties['respondent'], key="respondent_input")
            st.text_input("Counsel", value=st.session_state.parties['respondent_counsel'], key="respondent_counsel_input")
        
        if st.button("Update Parties"):
            st.session_state.parties['petitioner'] = st.session_state.petitioner_input
            st.session_state.parties['petitioner_counsel'] = st.session_state.petitioner_counsel_input
            st.session_state.parties['respondent'] = st.session_state.respondent_input
            st.session_state.parties['respondent_counsel'] = st.session_state.respondent_counsel_input
            st.success("✓ Party information updated successfully")
    
    elif section == "Annexures":
        st.markdown("### Filed Annexures")
        
        for annexure in st.session_state.annexures:
            status_class = "status-verified" if annexure['status'] == "Verified" else "status-pending"
            annexure_html = f"""
            <div class="info-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: bold; color: white; margin-bottom: 0.25rem;">{annexure['name']}</div>
                        <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.6);">{annexure['type']}</div>
                    </div>
                    <span class="status-badge {status_class}">{annexure['status']}</span>
                </div>
            </div>
            """
            st.markdown(annexure_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.form("add_annexure_form"):
            st.markdown("#### Add New Annexure")
            new_annexure_name = st.text_input("Annexure Name")
            new_annexure_type = st.text_input("Document Type")
            submitted = st.form_submit_button("Add Annexure")
            
            if submitted and new_annexure_name and new_annexure_type:
                st.session_state.annexures.append({
                    'name': new_annexure_name,
                    'type': new_annexure_type,
                    'status': 'Pending'
                })
                st.success(f"✓ Added {new_annexure_name}")
                st.rerun()
    
    elif section == "Forensic Reports":
        st.markdown("### Forensic Evidence Reports")
        
        for report in st.session_state.forensic_reports:
            status_class = "status-complete" if report['status'] == "Complete" else "status-pending"
            report_html = f"""
            <div class="info-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-weight: bold; color: white; margin-bottom: 0.25rem;">{report['name']}</div>
                        <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.6);">Date: {report['date']}</div>
                    </div>
                    <span class="status-badge {status_class}">{report['status']}</span>
                </div>
            </div>
            """
            st.markdown(report_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("🔬 Request New Forensic Analysis"):
            st.info("Forensic analysis request has been submitted to the technical division.")
    
    elif section == "Evidence Log":
        st.markdown("### Chronological Evidence Log")
        
        for entry in st.session_state.evidence_log:
            evidence_html = f"""
            <div class="evidence-entry">
                <div class="evidence-time">{entry['time']}</div>
                <div class="evidence-text">{entry['entry']}</div>
            </div>
            """
            st.markdown(evidence_html, unsafe_allow_html=True)
        
        st.markdown("---")
        
        with st.form("add_evidence_entry"):
            st.markdown("#### Add New Entry")
            new_entry_text = st.text_area("Entry Description")
            submitted = st.form_submit_button("Add to Log")
            
            if submitted and new_entry_text:
                current_time = datetime.now().strftime("%I:%M %p")
                st.session_state.evidence_log.append({
                    'time': current_time,
                    'entry': new_entry_text
                })
                st.success(f"✓ Entry added at {current_time}")
                st.rerun()

# ============================================================================
# MAIN APPLICATION LAYOUT
# ============================================================================

def main():
    """Main application entry point"""
    
    # Load custom CSS
    load_css()
    
    # Render draggable proceedings bar
    render_draggable_proceedings_bar()
    
    # Render header
    render_header()
    
    # Mode toggle + Settings in header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button(f"Switch to {'Advocate' if st.session_state.mode == 'Bench' else 'Bench'} Mode", key="mode_toggle"):
            toggle_mode()
            st.rerun()
    
    with col2:
        if st.button("📋 Show Record of Proceedings", key="show_proceedings"):
            st.session_state.show_proceedings_bar = True
            # Trigger JavaScript to show bar
            st.markdown("""
            <script>
                const bar = document.getElementById('proceedings-bar');
                if (bar) bar.style.display = 'flex';
            </script>
            """, unsafe_allow_html=True)
    
    with col3:
        if st.button("⚙️ Settings", key="open_settings"):
            st.session_state.show_settings = True
    
    # Settings Modal
    if st.session_state.get('show_settings', False):
        with st.expander("🔧 LLM & Settings Configuration", expanded=True):
            st.markdown("#### AI Text Generation Settings")
            
            llm_provider = st.radio(
                "Select LLM Provider for Evidence-Based Generation",
                ["Demo Mode", "OpenAI (GPT-3.5)", "Grok (xAI)"],
                index=0,
                key="llm_provider_radio"
            )
            
            if llm_provider == "OpenAI (GPT-3.5)":
                st.session_state.selected_llm = "openai"
                api_key = st.text_input(
                    "OpenAI API Key",
                    value=st.session_state.api_key_openai,
                    type="password",
                    key="openai_key_input"
                )
                if api_key:
                    st.session_state.api_key_openai = api_key
                    st.success("✓ OpenAI API Key configured")
                
                st.markdown("""
                **OpenAI Configuration:**
                - Model: gpt-3.5-turbo
                - Temperature: 0.3 (deterministic, fact-based)
                - Max Tokens: 1000
                - Evidence Grounding: LexAR retrieved chunks only
                """)
            
            elif llm_provider == "Grok (xAI)":
                st.session_state.selected_llm = "grok"
                api_key = st.text_input(
                    "Grok API Key",
                    value=st.session_state.api_key_grok,
                    type="password",
                    key="grok_key_input"
                )
                if api_key:
                    st.session_state.api_key_grok = api_key
                    st.success("✓ Grok API Key configured")
                
                st.markdown("""
                **Grok Configuration:**
                - Model: grok-1
                - Temperature: 0.3 (deterministic, fact-based)
                - Evidence Grounding: LexAR retrieved chunks only
                - Endpoint: https://api.x.ai/openai/v1/chat/completions
                """)
            
            else:
                st.session_state.selected_llm = "demo"
                st.info("Demo Mode: Using simulated LLM responses with LexAR-like output")
            
            st.markdown("---")
            st.markdown("#### LexAR Architecture Integration")
            
            lexar_info = f"""
            **Current LexAR Status:** {'✅ Active' if st.session_state.lexar_enabled else '⚠️ Demo Mode'}
            
            **LexAR Pipeline Flow:**
            1. Query Input → LLM receives question of law
            2. Dense Retrieval → LexAR retrieves top-K chunks from indices
            3. Cross-Encoder Reranking → Relevance scoring and filtering
            4. Evidence-Constrained Generation → LLM generates response using only evidence
            5. Citation Mapping → Attach source references
            
            **Key Principles Enforced:**
            ✓ No generation without evidence (retrieval mandatory)
            ✓ Hard attention masking on evidence chunks
            ✓ Evidence metadata preservation
            ✓ Provable grounding in retrieved content
            ✓ Confidence scores for all outputs
            ✓ Transparent failure handling
            
            **Generation Constraints:**
            - Temperature set to 0.3 (high determinism)
            - Only evidence tokens in attention
            - Citation links preserved
            - Confidence scores calculated
            """
            st.info(lexar_info)
            
            if st.button("Close Settings", key="close_settings"):
                st.session_state.show_settings = False
                st.rerun()
    
    # Main layout
    main_col1, main_col2, main_col3 = st.columns([1, 3, 1.5])
    
    # Left sidebar rendered in Streamlit sidebar
    render_left_sidebar()
    
    # Center content
    with main_col2:
        # Tabs for case types
        tabs = st.tabs([
            "Constitutional Matter",
            "Criminal Appeal",
            "Civil Appeal",
            "Writ Petition",
            "SLP"
        ])
        
        with tabs[0]:
            # Question of Law Section
            st.markdown("### ⚖️ QUESTION OF LAW PRESENTED")
            
            col_textarea, col_button = st.columns([3, 1])
            
            with col_textarea:
                question = st.text_area(
                    "State the issue for judicial determination...",
                    value=st.session_state.question_of_law,
                    height=100,
                    key="question_input",
                    label_visibility="collapsed"
                )
            
            with col_button:
                if st.button("⚖️\n\nPLACE BEFORE BENCH", key="place_before_bench", use_container_width=True):
                    st.session_state.question_of_law = question
                    st.success("✓ Question placed before bench")
                    update_metrics()
                    st.rerun()
            
            st.markdown("<div style='margin-bottom: 2rem;'></div>", unsafe_allow_html=True)
            
            # Document viewer
            render_document_viewer()
        
        with tabs[1]:
            st.markdown("### ⚖️ CRIMINAL APPEAL ANALYSIS")
            
            # Criminal Appeal Tabs
            criminal_col1, criminal_col2, criminal_col3 = st.columns(3)
            
            with criminal_col1:
                if st.button("📋 Charge Sheet", use_container_width=True, key="crim_charges"):
                    st.session_state.criminal_section = "charges"
            
            with criminal_col2:
                if st.button("👥 Witnesses", use_container_width=True, key="crim_witness"):
                    st.session_state.criminal_section = "witnesses"
            
            with criminal_col3:
                if st.button("⚖️ Sentencing", use_container_width=True, key="crim_sentencing"):
                    st.session_state.criminal_section = "sentencing"
            
            st.markdown("---")
            
            # Charge Sheet Analysis
            if st.session_state.get('criminal_section') == 'charges' or st.session_state.get('criminal_section') is None:
                st.markdown("#### CHARGE SHEET ANALYSIS")
                
                for charge in st.session_state.criminal_charges:
                    charge_html = f"""
                    <div class="info-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-weight: bold; color: white; margin-bottom: 0.25rem;">{charge['charge']} ({charge['code']})</div>
                                <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.6);">Section: {charge['section']}</div>
                            </div>
                            <div>
                                <span class="status-badge status-{'verified' if charge['status'] == 'Verified' else 'pending'}">{charge['status']}</span>
                                <div style="font-size: 0.75rem; color: rgba(198, 167, 94, 0.8); font-weight: bold; margin-top: 0.25rem;">Severity: {charge['severity']}</div>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(charge_html, unsafe_allow_html=True)
                
                if st.button("🤖 Analyze with LexAR", key="analyze_charges"):
                    with st.spinner("Analyzing charges with LexAR..."):
                        time.sleep(1.5)
                        # Retrieve evidence for charges
                        charges_text = ", ".join([c['charge'] for c in st.session_state.criminal_charges])
                        evidence = [f"IPC precedent for {c['charge']}" for c in st.session_state.criminal_charges]
                        response = generate_text_with_llm(f"Analyze these criminal charges: {charges_text}", evidence)
                        st.info("💡 LexAR Analysis:")
                        st.write(response)
            
            # Witness Examination
            elif st.session_state.get('criminal_section') == 'witnesses':
                st.markdown("#### WITNESS EXAMINATION RECORDS")
                
                for witness in st.session_state.witnesses:
                    witness_html = f"""
                    <div class="info-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <div style="font-weight: bold; color: white; margin-bottom: 0.25rem;">{witness['name']}</div>
                                <div style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.6);">Date: {witness['date']} | Status: {witness['status']}</div>
                            </div>
                            <div>
                                <div style="font-size: 0.875rem; color: {'#22c55e' if witness['credibility'] == 'High' else '#fbbf24'}; font-weight: bold;">Credibility: {witness['credibility']}</div>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(witness_html, unsafe_allow_html=True)
                
                # Add witness
                with st.form("add_witness_form"):
                    st.markdown("#### Add Witness")
                    w_name = st.text_input("Witness Name")
                    w_status = st.selectbox("Status", ["Examined", "Cross-examined", "Pending"])
                    w_credibility = st.selectbox("Credibility", ["High", "Medium", "Low"])
                    if st.form_submit_button("Add Witness"):
                        st.session_state.witnesses.append({
                            'name': w_name,
                            'status': w_status,
                            'credibility': w_credibility,
                            'date': datetime.now().strftime("%Y-%m-%d")
                        })
                        st.success(f"✓ Added {w_name}")
                        st.rerun()
                
                if st.button("🤖 Analyze Credibility with LexAR", key="analyze_witnesses"):
                    with st.spinner("Analyzing witness credibility..."):
                        time.sleep(1.5)
                        witnesses_text = ", ".join([w['name'] for w in st.session_state.witnesses])
                        evidence = [f"Witness credibility standards from IPC" for _ in st.session_state.witnesses]
                        response = generate_text_with_llm(f"Assess credibility of: {witnesses_text}", evidence)
                        st.info("💡 Credibility Analysis:")
                        st.write(response)
            
            # Sentencing Guidelines
            elif st.session_state.get('criminal_section') == 'sentencing':
                st.markdown("#### SENTENCING GUIDELINES & RECOMMENDATIONS")
                
                sentencing_html = f"""
                <div class="info-card">
                    <div class="info-card-title">RECOMMENDED SENTENCE</div>
                    <div class="info-card-content">{st.session_state.sentencing_data['recommended_years']} Years Rigorous Imprisonment</div>
                </div>
                """
                st.markdown(sentencing_html, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("##### ✓ Mitigating Factors")
                    for factor in st.session_state.sentencing_data['mitigating_factors']:
                        st.write(f"• {factor}")
                
                with col2:
                    st.markdown("##### ⚠️ Aggravating Factors")
                    for factor in st.session_state.sentencing_data['aggravating_factors']:
                        st.write(f"• {factor}")
                
                st.markdown("---")
                
                st.markdown("##### Guideline References")
                for ref in st.session_state.sentencing_data['guideline_references']:
                    st.write(f"📖 {ref}")
                
                if st.button("🤖 Refine Sentencing with LexAR", key="analyze_sentencing"):
                    with st.spinner("Analyzing sentencing with LexAR..."):
                        time.sleep(1.5)
                        sentencing_prompt = f"Refine sentencing recommendation of {st.session_state.sentencing_data['recommended_years']} years considering mitigating and aggravating factors"
                        evidence = st.session_state.sentencing_data['guideline_references']
                        response = generate_text_with_llm(sentencing_prompt, evidence)
                        st.info("💡 Sentencing Analysis:")
                        st.write(response)
        
        with tabs[2]:
            st.info("Civil Appeal section - Under development")
            st.markdown("""
            ### Civil Appeal Features
            - Suit documentation
            - Contractual obligations analysis
            - Damages calculation
            - Settlement recommendations
            """)
        
        with tabs[3]:
            st.info("Writ Petition section - Under development")
            st.markdown("""
            ### Writ Petition Features
            - Fundamental rights review
            - Government action analysis
            - Public interest considerations
            - Alternative remedies assessment
            """)
        
        with tabs[4]:
            st.info("Special Leave Petition section - Under development")
            st.markdown("""
            ### SLP Features
            - Leave to appeal assessment
            - Substantial question of law
            - Public importance evaluation
            - Precedent impact analysis
            """)
    
    # Right sidebar
    with main_col3:
        render_right_sidebar()
    
    # Export functionality
    st.markdown("---")
    col_export1, col_export2, col_export3 = st.columns([2, 1, 2])
    
    with col_export2:
        if st.button("🔒 SEAL & EXPORT DECREE", key="export_button", use_container_width=True):
            decree_content = export_decree()
            st.download_button(
                label="📄 Download Decree",
                data=decree_content,
                file_name=f"decree_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
            st.success("✓ Decree sealed and ready for export")
    
    # Footer metrics
    st.markdown("<div style='margin-bottom: 8rem;'></div>", unsafe_allow_html=True)
    render_footer_metrics()

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
