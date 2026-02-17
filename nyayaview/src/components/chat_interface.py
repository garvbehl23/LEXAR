import streamlit as st
import re

def format_answer_with_citations(answer_text):
    """Format answer text with clickable citations."""
    # Pattern to match citations like (IPC §302), (CrPC §41), etc.
    citation_pattern = r'\(([A-Z]+)\s*§(\d+)\)'
    
    def replace_citation(match):
        statute = match.group(1)
        section = match.group(2)
        citation_id = f"{statute}_{section}"
        return f'<span class="citation" onclick="window.parent.postMessage({{type: \'citation\', statute: \'{statute}\', section: \'{section}\'}}, \'*\')">{statute} §{section}</span>'
    
    formatted = re.sub(citation_pattern, replace_citation, answer_text)
    return formatted

def render_confidence_bar(confidence_percent, status):
    """Render premium confidence bar with glow effect."""
    bar_class = "amber" if status == "BORDERLINE" or confidence_percent < 75 else ""
    
    return f"""
    <div class="confidence-wrapper">
        <div class="confidence-label">EVIDENCE CONFIDENCE: {confidence_percent:.0f}%</div>
        <div class="confidence-track">
            <div class="confidence-progress {bar_class}" style="width: {confidence_percent}%;"></div>
        </div>
    </div>
    """

def render_chat_interface():
    """Render the chat-style interface."""
    
    # Create container for chat
    container = st.container()
    
    with container:
        # Input area
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            query = st.text_area(
                "Legal Query",
                placeholder="Ask a legal question about Indian statutes...",
                height=100,
                key="chat_input",
                label_visibility="collapsed"
            )
            
            col_a, col_b, col_c = st.columns([1, 1, 1])
            with col_b:
                submit = st.button("Analyze Query", use_container_width=True)
        
        # Process query
        if submit and query:
            with st.spinner(""):
                # Add user message to chat history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': query
                })
                
                # Process through LEXAR
                results = st.session_state.lexar_client.process_query(query)
                
                # Add AI response to chat history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': results['generation']['answer'],
                    'metadata': results
                })
    
    # Display chat history
    if st.session_state.chat_history:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(
                        f'<div class="user-bubble">{message["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    # AI message with formatted answer
                    formatted_answer = format_answer_with_citations(message['content'])
                    st.markdown(
                        f'<div class="ai-bubble">{formatted_answer}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Confidence bar
                    metadata = message.get('metadata', {})
                    gating = metadata.get('gating', {})
                    confidence = gating.get('max_attention', 85)
                    status = gating.get('status', 'PASS')
                    
                    st.markdown(
                        render_confidence_bar(confidence, status),
                        unsafe_allow_html=True
                    )
                    
                    # Expandable reasoning section
                    with st.expander("✨ Show Reasoning"):
                        render_reasoning_section(metadata)
                    
                    # Token attribution
                    with st.expander("🔍 Token Attribution"):
                        render_token_attribution(metadata.get('provenance', {}))

def render_reasoning_section(metadata):
    """Render the reasoning section with retrieved evidence."""
    retrieval = metadata.get('retrieval', {})
    gating = metadata.get('gating', {})
    
    st.markdown("##### Retrieved Statutory Evidence")
    
    chunks = retrieval.get('chunks', [])[:3]  # Show top 3
    for i, chunk in enumerate(chunks, 1):
        st.markdown(
            f"""
            <div class="evidence-card">
                <div>
                    <span class="statute-tag">{chunk.get('statute', 'N/A')}</span>
                    <span class="score-tag">{chunk.get('score', 0):.3f}</span>
                </div>
                <div style='margin-top: 0.75rem; color: #9ca3af; font-size: 0.9rem;'>
                    Section: {chunk.get('section', 'N/A')}
                </div>
                <div style='margin-top: 0.75rem; color: #d1d5db; font-size: 0.95rem; line-height: 1.6;'>
                    {chunk.get('text', '')[:200]}...
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("##### Evidence Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Dominant Section:** {gating.get('dominant_section', 'N/A')}")
        st.markdown(f"**Threshold:** {gating.get('threshold', 0):.3f}")
    with col2:
        st.markdown(f"**Max Attention:** {gating.get('max_attention', 0):.1f}%")
        st.markdown(f"**Margin:** {gating.get('margin', 0):.3f}")

def render_token_attribution(provenance_data):
    """Render token attribution with hover tooltips."""
    tokens = provenance_data.get('tokens', [])
    
    if not tokens:
        st.info("No token provenance data available.")
        return
    
    st.markdown("##### Attributed Tokens")
    
    # Group tokens by statute
    token_html = "<div style='line-height: 2.5;'>"
    for token_info in tokens[:20]:  # Show first 20 tokens
        token = token_info.get('token', '')
        statute = token_info.get('statute', 'N/A')
        section = token_info.get('section', 'N/A')
        confidence = token_info.get('confidence', 0)
        
        tooltip = f"{statute} {section} ({confidence:.3f})"
        token_html += f'<span class="token-pill" title="{tooltip}">{token}</span> '
    
    token_html += "</div>"
    st.markdown(token_html, unsafe_allow_html=True)
