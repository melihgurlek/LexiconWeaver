"""Unit tests for text processing utilities."""

from lexiconweaver.utils.text_processor import (
    extract_ngrams,
    extract_paragraphs,
    generate_hash,
    normalize_text,
)


def test_extract_paragraphs() -> None:
    """Test paragraph extraction."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paragraphs = extract_paragraphs(text)

    assert len(paragraphs) == 3
    assert "First paragraph." in paragraphs[0]
    assert "Second paragraph." in paragraphs[1]


def test_normalize_text() -> None:
    """Test text normalization."""
    text = "  Multiple   spaces\n\n\nMultiple  newlines  "
    normalized = normalize_text(text)

    assert "  " not in normalized  # Multiple spaces removed
    assert "\n\n\n" not in normalized  # Excessive newlines removed


def test_generate_hash() -> None:
    """Test hash generation."""
    text1 = "Test text"
    text2 = "Test text"
    text3 = "Different text"

    hash1 = generate_hash(text1)
    hash2 = generate_hash(text2)
    hash3 = generate_hash(text3)

    assert hash1 == hash2  # Same text = same hash
    assert hash1 != hash3  # Different text = different hash
    assert len(hash1) == 64  # SHA-256 produces 64-char hex string


def test_extract_ngrams() -> None:
    """Test N-gram extraction."""
    text = "The Golden Core technique"
    ngrams = list(extract_ngrams(text, n=2, min_length=3))

    assert len(ngrams) > 0
    assert any("golden core" in ngram[0].lower() for ngram in ngrams)
