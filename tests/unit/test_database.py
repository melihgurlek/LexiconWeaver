"""Unit tests for database models and manager."""

import pytest

from lexiconweaver.database.models import GlossaryTerm, IgnoredTerm, Project, TranslationCache
from tests.conftest import initialized_db


def test_project_creation(initialized_db: Project) -> None:
    """Test project creation."""
    assert initialized_db.title == "test_project"
    assert initialized_db.source_lang == "en"
    assert initialized_db.target_lang == "tr"
    assert initialized_db.id is not None


def test_glossary_term_creation(initialized_db: Project) -> None:
    """Test glossary term creation."""
    term = GlossaryTerm.create(
        project=initialized_db,
        source_term="Test Term",
        target_term="Test Çeviri",
        category="Skill",
        confidence=0.9,
    )

    assert term.source_term == "Test Term"
    assert term.target_term == "Test Çeviri"
    assert term.category == "Skill"
    assert term.confidence == 0.9


def test_ignored_term_creation(initialized_db: Project) -> None:
    """Test ignored term creation."""
    ignored = IgnoredTerm.create(project=initialized_db, term="common word")

    assert ignored.term == "common word"
    assert ignored.project == initialized_db


def test_translation_cache(initialized_db: Project) -> None:
    """Test translation cache."""
    cache = TranslationCache.create(
        hash="abc123",
        project=initialized_db,
        translation="Test translation",
    )

    assert cache.hash == "abc123"
    assert cache.translation == "Test translation"
