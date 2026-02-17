import streamlit as st

def render_response_display(generation_result):
    """Render the generated legal answer section."""
    if not generation_result:
        return
    
    st.markdown("")
    st.markdown("### Generated Legal Answer")
    
    answer_text = generation_result.get('answer', 'No answer generated.')
    
    st.markdown(
        f"""
        <div style='background-color: #ffffff; padding: 25px; border-radius: 8px; 
                    border-left: 4px solid #3b82f6; font-size: 1.1rem; line-height: 1.8;'>
            {answer_text}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_evidence_gating(gating_result):
    """Render the evidence sufficiency gate section."""
    if not gating_result:
        return
    
    st.markdown("")
    st.markdown("### Evidence Sufficiency Gate")
    
    status = gating_result.get('status', 'UNKNOWN')
    threshold = gating_result.get('threshold', 0.0)
    max_attention = gating_result.get('max_attention', 0.0)
    dominant_section = gating_result.get('dominant_section', 'N/A')
    margin = gating_result.get('margin', 0.0)
    
    status_color = "#10b981" if status == "PASS" else "#ef4444"
    
    st.markdown(
        f"""
        <div style='background-color: #f9fafb; padding: 20px; border-radius: 8px; 
                    border: 2px solid {status_color};'>
            <p style='margin: 0; font-size: 1.2rem; font-weight: 600; color: {status_color};'>
                Status: {status}
            </p>
            <div style='margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                <p style='margin: 5px 0; color: #374151;'><strong>Threshold:</strong> {threshold:.3f}</p>
                <p style='margin: 5px 0; color: #374151;'><strong>Max Attention:</strong> {max_attention:.1f}%</p>
                <p style='margin: 5px 0; color: #374151;'><strong>Dominant Section:</strong> {dominant_section}</p>
                <p style='margin: 5px 0; color: #374151;'><strong>Margin:</strong> {margin:.3f}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if status == "FAIL":
        st.warning("⚠️ Evidence insufficiency detected. The system cannot provide a confident answer with the available statutory evidence.")