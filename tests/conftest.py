"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest

from lexiconweaver.config import Config
from lexiconweaver.database import (
    GlossaryTerm,
    IgnoredTerm,
    Project,
    TranslationCache,
    close_database,
    initialize_database,
)


@pytest.fixture
def temp_db() -> Generator[Path, None, None]:
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    yield db_path

    # Cleanup
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def config(temp_db: Path) -> Config:
    """Create a test configuration."""
    config = Config.load()
    config.database.path = str(temp_db)
    return config


@pytest.fixture
def initialized_db(config: Config) -> Generator[Project, None, None]:
    """Initialize database and return a test project."""
    initialize_database(config)
    project = Project.create(
        title="test_project", source_lang="en", target_lang="tr"
    )
    yield project
    close_database()


@pytest.fixture
def sample_text() -> str:
    """Sample text for testing."""
    return """
    The Golden Core technique was known as the most powerful cultivation method.
    Rank 3 Gu Masters could use this technique to advance their cultivation.
    The technique called Void Step allowed practitioners to move through space.
    """


@pytest.fixture
def sample_glossary_terms(initialized_db: Project) -> list[GlossaryTerm]:
    """Create sample glossary terms."""
    terms = [
        GlossaryTerm.create(
            project=initialized_db,
            source_term="Golden Core",
            target_term="Altın Çekirdek",
            category="Skill",
            confidence=0.9,
        ),
        GlossaryTerm.create(
            project=initialized_db,
            source_term="Gu Master",
            target_term="Gu Ustası",
            category="Person",
            confidence=0.85,
        ),
    ]
    return terms
