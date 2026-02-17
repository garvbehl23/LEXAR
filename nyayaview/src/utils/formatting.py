def format_confidence_score(score):
    """Format the confidence score to three decimal places."""
    return f"{score:.3f}"

def format_evidence_chunk(chunk):
    """Format evidence chunk for display."""
    return f"Evidence: {chunk['text']} (Source: {chunk['source']})"

def format_provenance_data(provenance):
    """Format provenance data for display."""
    return {
        "token": provenance['token'],
        "top_source": provenance['top_source'],
        "attention_score": format_confidence_score(provenance['attention_score'])
    }