import streamlit as st

def render_citation_panel():
    """Render the slide-in citation panel."""
    
    citation = st.session_state.selected_citation
    
    if not citation:
        return
    
    st.markdown(
        f"""
        <div class="side-panel open">
            <h3 style='color: #e5e7eb; margin-bottom: 1.5rem;'>
                {citation['statute']} §{citation['section']}
            </h3>
            
            <div style='
                background: rgba(96, 165, 250, 0.1);
                border: 1px solid rgba(96, 165, 250, 0.3);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 1.5rem;
            '>
                <div style='color: #9ca3af; font-size: 0.85rem; margin-bottom: 0.5rem;'>
                    Confidence
                </div>
                <div style='color: #60a5fa; font-size: 1.5rem; font-weight: 600;'>
                    {citation.get('confidence', 0.85):.1%}
                </div>
            </div>
            
            <div style='margin-bottom: 1.5rem;'>
                <div style='color: #9ca3af; font-size: 0.85rem; margin-bottom: 0.5rem;'>
                    Full Text
                </div>
                <div style='
                    color: #d1d5db;
                    font-size: 0.95rem;
                    line-height: 1.7;
                    padding: 1rem;
                    background: rgba(255, 255, 255, 0.03);
                    border-radius: 8px;
                '>
                    {citation.get('text', 'Statutory text would appear here...')}
                </div>
            </div>
            
            <button 
                onclick="window.parent.postMessage({{type: 'close_panel'}}, '*')"
                style='
                    background: rgba(239, 68, 68, 0.2);
                    border: 1px solid rgba(239, 68, 68, 0.3);
                    color: #ef4444;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    cursor: pointer;
                    width: 100%;
                    font-weight: 500;
                '
            >
                Close
            </button>
        </div>
        """,
        unsafe_allow_html=True
    )
