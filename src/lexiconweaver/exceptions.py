"""Custom exception hierarchy for LexiconWeaver."""

from typing import Optional


class LexiconWeaverError(Exception):
    """Base exception for all LexiconWeaver errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ScoutError(LexiconWeaverError):
    """Error in the Scout (Discovery Engine)."""

    pass


class WeaverError(LexiconWeaverError):
    """Error in the Weaver (Generation Engine)."""

    pass


class DatabaseError(LexiconWeaverError):
    """Error in database operations."""

    pass


class ConfigurationError(LexiconWeaverError):
    """Error in configuration loading or validation."""

    pass


class TranslationError(WeaverError):
    """Error during translation process."""

    pass


class ValidationError(LexiconWeaverError):
    """Error in input validation."""

    pass
