"""Text processing utilities for paragraph extraction and normalization."""

import hashlib
import re
from typing import Iterator

from lexiconweaver.exceptions import ValidationError


def extract_paragraphs(text: str) -> list[str]:
    """Extract paragraphs from text, preserving structure."""
    # Split on double newlines (paragraph breaks)
    # Also handle single newlines if they separate paragraphs
    paragraphs = re.split(r"\n\s*\n", text.strip())

    # Filter out empty paragraphs
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    return paragraphs


def normalize_text(text: str) -> str:
    """Normalize text for processing."""
    # Remove excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    # Normalize line breaks
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\r", "\n", text)
    # Remove excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_hash(text: str) -> str:
    """Generate SHA-256 hash for cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (basic implementation)."""
    # Simple sentence splitting on period, exclamation, question mark
    # This is basic - could be improved with NLP libraries
    sentences = re.split(r"[.!?]+\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def extract_ngrams(text: str, n: int, min_length: int = 2) -> Iterator[tuple[str, int]]:
    """Extract N-grams from text with their positions."""
    words = re.findall(r"\b\w+\b", text.lower())
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        # Filter out short ngrams
        if len(ngram) >= min_length:
            yield (ngram, i)
