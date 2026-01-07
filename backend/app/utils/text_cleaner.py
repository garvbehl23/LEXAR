import re

def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing whitespace.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()
