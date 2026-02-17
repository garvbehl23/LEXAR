import streamlit as st

def render_evidence_viewer(evidence_data):
    """Render the retrieval analysis section with retrieved chunks."""
    if not evidence_data or not evidence_data.get('chunks'):
        return
    
    st.markdown("")
    st.markdown("### Retrieved Statutory Evidence")
    
    chunks = evidence_data['chunks']
    
    # Create 3-column grid
    num_chunks = len(chunks)
    cols_per_row = 3
    
    for i in range(0, num_chunks, cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            idx = i + j
            if idx < num_chunks:
                chunk = chunks[idx]
                with cols[j]:
                    st.markdown(
                        f"""
                        <div style='background-color: #f9fafb; padding: 15px; border-radius: 8px; 
                                    border: 1px solid #e5e7eb; margin-bottom: 15px;'>
                            <p style='margin: 0; color: #374151; font-weight: 600; font-size: 0.9rem;'>
                                {chunk.get('statute', 'N/A')}
                            </p>
                            <p style='margin: 5px 0; color: #6b7280; font-size: 0.85rem;'>
                                Section: {chunk.get('section', 'N/A')}
                            </p>
                            <p style='margin: 5px 0; color: #9ca3af; font-size: 0.75rem;'>
                                Chunk ID: {chunk.get('chunk_id', 'N/A')}
                            </p>
                            <p style='margin: 5px 0; color: #1f2937; font-size: 0.85rem; font-weight: 500;'>
                                Score: {chunk.get('score', 0):.4f}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    with st.expander("View full text"):
                        st.text(chunk.get('text', 'No text available'))