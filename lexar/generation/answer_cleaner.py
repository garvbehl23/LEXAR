import re

def clean_answer(text: str) -> str:
    """
    Clean minor decoder artifacts without changing semantics.
    """
    # Remove leading enumeration artifacts like "1[" or "1."
    text = re.sub(r"^\s*\d+\s*[\[\.\]]\s*", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
