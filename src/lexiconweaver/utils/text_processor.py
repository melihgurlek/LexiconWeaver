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


def batch_paragraphs_smart(
    text: str, max_chars: int = 2000, context_sentences: int = 2
) -> list[str]:
    """Batch paragraphs intelligently without exceeding character limit.
    
    Batches paragraphs together until adding another would exceed max_chars.
    Context is handled separately at the translation/prompt layer (to avoid
    duplicating context inside the batch itself).
    
    Args:
        text: The text to batch
        max_chars: Maximum characters per batch (default: 2000)
        context_sentences: Deprecated (kept for compatibility; ignored)
        
    Returns:
        List of batched text chunks
    """
    paragraphs = extract_paragraphs(text)
    if not paragraphs:
        return []
    
    batches: list[str] = []
    current_batch: list[str] = []
    current_length = 0
    
    for paragraph in paragraphs:
        para_length = len(paragraph)
        
        # If a single paragraph exceeds max_chars, we still need to include it
        # but we'll add it as its own batch
        if para_length > max_chars:
            if current_batch:
                batches.append("\n\n".join(current_batch))
                current_batch = []
                current_length = 0
            
            # Add the oversized paragraph as its own batch
            batches.append(paragraph)
            continue
        
        # Check if adding this paragraph would exceed the limit
        separator_len = 2 if current_batch else 0  # "\n\n"
        potential_length = current_length + separator_len + para_length

        if potential_length > max_chars and current_batch:
            batches.append("\n\n".join(current_batch))
            current_batch = []
            current_length = 0
        
        current_batch.append(paragraph)
        current_length += para_length + (2 if len(current_batch) > 1 else 0)
    
    if current_batch:
        batches.append("\n\n".join(current_batch))
    
    return batches if batches else [text]
