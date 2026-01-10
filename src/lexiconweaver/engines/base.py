"""Base engine class for LexiconWeaver engines."""

from abc import ABC, abstractmethod
from typing import Any


class BaseEngine(ABC):
    """Base class for all engines."""

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Process input and return results."""
        pass
