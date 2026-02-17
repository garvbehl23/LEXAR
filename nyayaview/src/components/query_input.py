import streamlit as st

def render_query_input():
    """Render the query input section."""
    st.markdown("")  # Spacing
    
    query = st.text_area(
        "Legal Query",
        placeholder="Enter your legal question here...",
        height=100,
        key="query_input"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_button = st.button("Run Legal Analysis", use_container_width=True, type="primary")
    
    return query if submit_button and query else None