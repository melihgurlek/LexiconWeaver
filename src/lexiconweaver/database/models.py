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


def create_tables() -> None:
    """Create all database tables."""
    if db_proxy.obj is None:
        raise DatabaseError("Database not initialized")
    db_proxy.create_tables([Project, GlossaryTerm, IgnoredTerm, TranslationCache])


def drop_tables() -> None:
    """Drop all database tables (use with caution!)."""
    if db_proxy.obj is None:
        raise DatabaseError("Database not initialized")
    db_proxy.drop_tables([Project, GlossaryTerm, IgnoredTerm, TranslationCache])
