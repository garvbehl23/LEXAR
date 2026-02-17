import streamlit as st
from components.header import render_hero_header
from components.chat_interface import render_chat_interface
from components.citation_panel import render_citation_panel
from services.lexar_client import LexarClient

def inject_custom_css():
    """Inject premium Google DeepMind-style CSS with advanced animations."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Premium dark background with animated gradient */
        .stApp {
            background: #0a0a0f;
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(99, 102, 241, 0.1) 0px, transparent 50%),
                radial-gradient(at 0% 100%, rgba(236, 72, 153, 0.08) 0px, transparent 50%);
            color: #e5e7eb;
            animation: gradientShift 20s ease infinite;
        }
        
        @keyframes gradientShift {
            0%, 100% { filter: hue-rotate(0deg); }
            50% { filter: hue-rotate(10deg); }
        }
        
        /* Hide Streamlit elements */
        #MainMenu, footer, header {visibility: hidden;}
        .stDeployButton {display: none;}
        
        /* Remove default padding */
        .block-container {
            padding-top: 2rem !important;
            max-width: 1200px !important;
        }
        
        /* Premium glassmorphism */
        .glass-card {
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(40px) saturate(180%);
            -webkit-backdrop-filter: blur(40px) saturate(180%);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 24px;
            padding: 2rem;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
        }
        
        /* Chat bubbles - ultra premium */
        .user-bubble {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            padding: 1.25rem 1.75rem;
            border-radius: 28px 28px 4px 28px;
            margin: 1.5rem 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 
                0 10px 40px rgba(99, 102, 241, 0.3),
                0 2px 8px rgba(99, 102, 241, 0.2);
            animation: slideInRight 0.5s cubic-bezier(0.16, 1, 0.3, 1);
            font-size: 1.05rem;
            line-height: 1.7;
        }
        
        .ai-bubble {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 1.75rem 2rem;
            border-radius: 28px 28px 28px 4px;
            margin: 1.5rem 0;
            max-width: 85%;
            box-shadow: 
                0 8px 32px rgba(0, 0, 0, 0.3),
                0 1px 2px rgba(255, 255, 255, 0.05);
            animation: slideInLeft 0.6s cubic-bezier(0.16, 1, 0.3, 1);
            line-height: 1.9;
            font-size: 1.08rem;
            font-weight: 400;
            color: #f3f4f6;
        }
        
        /* Premium citations */
        .citation {
            color: #a5b4fc;
            background: rgba(165, 180, 252, 0.12);
            padding: 3px 10px;
            border-radius: 8px;
            border: 1px solid rgba(165, 180, 252, 0.25);
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            font-weight: 500;
            font-size: 0.95rem;
            text-decoration: none;
            display: inline-block;
        }
        
        .citation:hover {
            background: rgba(165, 180, 252, 0.2);
            border-color: #a5b4fc;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(165, 180, 252, 0.3);
        }
        
        /* Confidence indicator - DeepMind style */
        .confidence-wrapper {
            margin-top: 1.5rem;
            padding: 1.25rem;
            background: rgba(255, 255, 255, 0.02);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        
        .confidence-label {
            color: #9ca3af;
            font-size: 0.875rem;
            font-weight: 500;
            margin-bottom: 0.75rem;
            letter-spacing: 0.02em;
        }
        
        .confidence-track {
            height: 6px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 3px;
            overflow: hidden;
            position: relative;
        }
        
        .confidence-progress {
            height: 100%;
            background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7);
            border-radius: 3px;
            box-shadow: 
                0 0 20px rgba(16, 185, 129, 0.5),
                0 0 40px rgba(16, 185, 129, 0.3);
            animation: progressGlow 2s ease-in-out infinite, fillProgress 1.2s cubic-bezier(0.16, 1, 0.3, 1);
        }
        
        .confidence-progress.amber {
            background: linear-gradient(90deg, #f59e0b, #fbbf24, #fcd34d);
            box-shadow: 
                0 0 20px rgba(245, 158, 11, 0.5),
                0 0 40px rgba(245, 158, 11, 0.3);
        }
        
        @keyframes progressGlow {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.85; }
        }
        
        @keyframes fillProgress {
            from { width: 0; }
        }
        
        /* Text input - premium style */
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 20px !important;
            color: #f3f4f6 !important;
            font-size: 1.05rem !important;
            padding: 1.25rem 1.5rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-weight: 400 !important;
            line-height: 1.6 !important;
        }
        
        .stTextArea textarea:focus {
            border-color: rgba(99, 102, 241, 0.5) !important;
            box-shadow: 
                0 0 0 3px rgba(99, 102, 241, 0.1),
                0 8px 24px rgba(99, 102, 241, 0.2) !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }
        
        .stTextArea textarea::placeholder {
            color: #6b7280 !important;
        }
        
        /* Premium button */
        .stButton > button {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 16px !important;
            padding: 0.875rem 2.5rem !important;
            font-weight: 600 !important;
            font-size: 1.05rem !important;
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 
                0 4px 20px rgba(99, 102, 241, 0.4),
                0 1px 3px rgba(0, 0, 0, 0.2) !important;
            letter-spacing: 0.01em !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) scale(1.02) !important;
            box-shadow: 
                0 12px 40px rgba(99, 102, 241, 0.5),
                0 4px 12px rgba(99, 102, 241, 0.3) !important;
        }
        
        .stButton > button:active {
            transform: translateY(-1px) scale(0.98) !important;
        }
        
        /* Expander - DeepMind style */
        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            color: #e5e7eb !important;
            font-weight: 500 !important;
            padding: 1rem 1.25rem !important;
            transition: all 0.3s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(255, 255, 255, 0.05) !important;
            border-color: rgba(99, 102, 241, 0.3) !important;
        }
        
        .streamlit-expanderContent {
            background: rgba(255, 255, 255, 0.015) !important;
            border: 1px solid rgba(255, 255, 255, 0.04) !important;
            border-radius: 14px !important;
            padding: 1.5rem !important;
            margin-top: 0.5rem !important;
        }
        
        /* Evidence cards */
        .evidence-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 1.25rem;
            margin: 1rem 0;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .evidence-card:hover {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        }
        
        .statute-tag {
            display: inline-block;
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            padding: 6px 14px;
            border-radius: 10px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-right: 0.5rem;
            border: 1px solid rgba(99, 102, 241, 0.25);
        }
        
        .score-tag {
            background: rgba(16, 185, 129, 0.15);
            color: #6ee7b7;
            padding: 6px 12px;
            border-radius: 10px;
            font-size: 0.8rem;
            font-weight: 500;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }
        
        /* Token highlights */
        .token-pill {
            display: inline-block;
            padding: 4px 8px;
            margin: 2px;
            border-radius: 8px;
            background: rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(99, 102, 241, 0.2);
            cursor: pointer;
            transition: all 0.2s ease;
            font-size: 0.9rem;
        }
        
        .token-pill:hover {
            background: rgba(99, 102, 241, 0.2);
            border-color: rgba(99, 102, 241, 0.4);
            transform: translateY(-1px);
        }
        
        /* Animations */
        @keyframes slideInRight {
            from {
                opacity: 0;
                transform: translateX(30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes slideInLeft {
            from {
                opacity: 0;
                transform: translateX(-30px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
        }
        
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.15);
        }
        
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
            color: #f3f4f6 !important;
        }
        
        div[data-testid="stMetricLabel"] {
            color: #9ca3af !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="VakalaatGPT",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Inject custom CSS
    inject_custom_css()
    
    # Initialize LEXAR client
    if 'lexar_client' not in st.session_state:
        st.session_state.lexar_client = LexarClient()
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'selected_citation' not in st.session_state:
        st.session_state.selected_citation = None
    
    # Render hero header
    render_hero_header()
    
    # Render chat interface
    render_chat_interface()
    
    # Render citation panel if citation selected
    if st.session_state.selected_citation:
        render_citation_panel()

if __name__ == "__main__":
    main()