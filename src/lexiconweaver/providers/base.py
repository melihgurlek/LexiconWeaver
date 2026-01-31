"""Base LLM provider abstract class."""

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.
    
    All LLM providers (Ollama, DeepSeek, etc.) must implement this interface.
    """

    @abstractmethod
    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles are typically 'system', 'user', 'assistant'.
        
        Returns:
            The generated text response.
        
        Raises:
            ProviderError: If the API call fails.
        """
        pass

    @abstractmethod
    async def generate_streaming(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Yields:
            Text chunks as they arrive from the LLM.
        
        Raises:
            ProviderError: If the API call fails.
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if the provider is available and configured.
        
        Returns:
            True if the provider can be used, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name for logging/display."""
        pass
