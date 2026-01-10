"""Utility functions for text processing and validation."""

from lexiconweaver.utils.highlighting import highlight_terms, LongestMatchHighlighter
from lexiconweaver.utils.text_processor import extract_paragraphs, normalize_text
from lexiconweaver.utils.validators import validate_encoding, validate_text_file

__all__ = [
    "extract_paragraphs",
    "normalize_text",
    "highlight_terms",
    "LongestMatchHighlighter",
    "validate_encoding",
    "validate_text_file",
]
