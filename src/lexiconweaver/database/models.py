"""Database models for LexiconWeaver using Peewee ORM."""

from datetime import datetime
from typing import Optional

from peewee import (
    BooleanField,
    CharField,
    DatabaseProxy,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    TextField,
)

from lexiconweaver.exceptions import DatabaseError

# Database proxy that will be initialized by manager
db_proxy = DatabaseProxy()


class BaseModel(Model):
    """Base model with common fields."""

    class Meta:
        database = db_proxy


class Project(BaseModel):
    """Project model representing a translation project."""

    title = CharField(max_length=255, unique=True)
    source_lang = CharField(max_length=10, default="en")
    target_lang = CharField(max_length=10, default="tr")
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "projects"


class GlossaryTerm(BaseModel):
    """Glossary term model storing source-target term mappings."""

    project = ForeignKeyField(Project, backref="glossary_terms", on_delete="CASCADE")
    source_term = CharField(max_length=255, index=True)
    target_term = CharField(max_length=255)
    category = CharField(max_length=50, null=True)  # Skill, Item, Person, Location, etc.
    is_regex = BooleanField(default=False)
    confidence = FloatField(null=True)  # Stored for historical analytics
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "glossary_terms"
        indexes = (
            (("project", "source_term"), True),  # Unique constraint
        )


class IgnoredTerm(BaseModel):
    """Terms that should be ignored by the Scout."""

    project = ForeignKeyField(Project, backref="ignored_terms", on_delete="CASCADE")
    term = CharField(max_length=255, index=True)
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "ignored_terms"
        indexes = (
            (("project", "term"), True),  # Unique constraint
        )


class TranslationCache(BaseModel):
    """Cache for translated paragraphs."""

    hash = CharField(max_length=64, primary_key=True)  # SHA-256 hash
    project = ForeignKeyField(Project, backref="cache_entries", on_delete="CASCADE")
    translation = TextField()
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "translation_cache"


class ProposedTerm(BaseModel):
    """LLM-proposed terms awaiting user approval.
    
    This table stores terms proposed by the ScoutRefiner (LLM-based)
    with their suggested translations and categories.
    """

    project = ForeignKeyField(Project, backref="proposed_terms", on_delete="CASCADE")
    source_term = CharField(max_length=255, index=True)
    proposed_translation = CharField(max_length=255)
    proposed_category = CharField(max_length=50, null=True)
    llm_reasoning = TextField(null=True)  # LLM's explanation for the proposal
    context_snippet = TextField(null=True)  # Sentence where term appeared
    status = CharField(max_length=20, default="pending")  # pending/approved/rejected/modified
    user_translation = CharField(max_length=255, null=True)  # User's override
    user_category = CharField(max_length=50, null=True)  # User's category choice
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "proposed_terms"
        indexes = (
            (("project", "source_term"), True),  # Unique per project
        )


class ChapterMetadata(BaseModel):
    """Metadata for chapters in batch translation workflow.
    
    Stores information about chapters for multi-chapter projects,
    including chapter numbers, titles, and statistics.
    """

    project = ForeignKeyField(Project, backref="chapters", on_delete="CASCADE")
    chapter_num = IntegerField(index=True)
    filename = CharField(max_length=255)
    title = CharField(max_length=255, null=True)
    word_count = IntegerField(null=True)
    char_count = IntegerField(null=True)
    translated = BooleanField(default=False)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "chapter_metadata"
        indexes = (
            (("project", "chapter_num"), True),
        )


class BurstTerm(BaseModel):
    """Terms detected by burst detection algorithm.
    
    Stores terms that show high local density (sudden frequency spikes)
    across chapters, indicating important story elements introduced mid-narrative.
    """

    project = ForeignKeyField(Project, backref="burst_terms", on_delete="CASCADE")
    term = CharField(max_length=255, index=True)
    first_chapter = IntegerField()
    max_window_frequency = IntegerField()
    global_frequency = IntegerField()
    density_score = FloatField()
    status = CharField(max_length=20, default="pending")
    created_at = DateTimeField(default=datetime.now)

    class Meta:
        table_name = "burst_terms"
        indexes = (
            (("project", "term"), True),
        )


def create_tables() -> None:
    """Create all database tables."""
    if db_proxy.obj is None:
        raise DatabaseError("Database not initialized")
    db_proxy.create_tables(
        [
            Project,
            GlossaryTerm,
            IgnoredTerm,
            TranslationCache,
            ProposedTerm,
            ChapterMetadata,
            BurstTerm,
        ],
        safe=True,
    )


def drop_tables() -> None:
    """Drop all database tables (use with caution!)."""
    if db_proxy.obj is None:
        raise DatabaseError("Database not initialized")
    db_proxy.drop_tables(
        [
            Project,
            GlossaryTerm,
            IgnoredTerm,
            TranslationCache,
            ProposedTerm,
            ChapterMetadata,
            BurstTerm,
        ]
    )
