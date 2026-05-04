"""
Text cleaning utilities for the We vs Them NLP project.
"""

import re


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove special characters; keep letters, digits, spaces and basic punct
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
