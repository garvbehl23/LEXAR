import streamlit as st
import pandas as pd

def render_provenance_graph(provenance_data):
    """Render the token-level provenance table."""
    if not provenance_data or not provenance_data.get('tokens'):
        return
    
    st.markdown("")
    st.markdown("### Token-Level Evidence Attribution")
    
    tokens = provenance_data['tokens']
    
    # Format data for display
    table_data = []
    for token_info in tokens:
        table_data.append({
            'Token': token_info.get('token', ''),
            'Statute': token_info.get('statute', 'N/A'),
            'Section': token_info.get('section', 'N/A'),
            'Confidence': f"{token_info.get('confidence', 0):.3f}"
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No token provenance data available.")