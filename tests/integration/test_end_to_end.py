"""End-to-end integration tests."""

import pytest

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project
from lexiconweaver.engines.scout import Scout
from tests.conftest import initialized_db, sample_text


def test_scout_to_database_integration(config: Config, initialized_db: Project, sample_text: str) -> None:
    """Test integration between Scout and database."""
    scout = Scout(config, initialized_db)
    scout.min_confidence = 0.1  # Lower for testing

    candidates = scout.process(sample_text)

    # Should find candidates
    assert len(candidates) > 0

    # Save a term to database
    term = candidates[0]
    GlossaryTerm.create(
        project=initialized_db,
        source_term=term.term,
        target_term="Test Translation",
        category="Skill",
        confidence=term.confidence,
    )

    # Verify it was saved
    saved = GlossaryTerm.get(GlossaryTerm.source_term == term.term)
    assert saved.target_term == "Test Translation"
