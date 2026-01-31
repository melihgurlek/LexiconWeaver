"""LLM Provider Manager with fallback support."""

from typing import AsyncIterator, Optional

from lexiconweaver.config import Config
from lexiconweaver.exceptions import ProviderError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.providers.base import BaseLLMProvider
from lexiconweaver.providers.deepseek import DeepSeekProvider
from lexiconweaver.providers.ollama import OllamaProvider

logger = get_logger(__name__)


class LLMProviderManager:
    """Manages LLM providers with fallback support.
    
    This class handles provider selection, availability checking, and
    automatic fallback when the primary provider fails.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the provider manager.
        
        Args:
            config: Application configuration with provider settings.
        """
        self.config = config
        self._primary: Optional[BaseLLMProvider] = None
        self._fallback: Optional[BaseLLMProvider] = None
        self._current: Optional[BaseLLMProvider] = None

        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Initialize primary and fallback providers based on config."""
        provider_config = self.config.provider

        # Initialize primary provider
        self._primary = self._create_provider(provider_config.primary)
        if self._primary:
            logger.info("Primary provider initialized", provider=self._primary.name)

        # Initialize fallback provider
        if provider_config.fallback != "none":
            self._fallback = self._create_provider(provider_config.fallback)
            if self._fallback:
                logger.info(
                    "Fallback provider initialized", provider=self._fallback.name
                )

        # Set current to primary
        self._current = self._primary

    def _create_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Create a provider instance by name.
        
        Args:
            provider_name: Name of the provider ('ollama' or 'deepseek').
        
        Returns:
            Provider instance or None if creation failed.
        """
        if provider_name == "ollama":
            return OllamaProvider(self.config.ollama)
        elif provider_name == "deepseek":
            return DeepSeekProvider(self.config.deepseek)
        else:
            logger.warning("Unknown provider", provider=provider_name)
            return None

    @property
    def primary(self) -> Optional[BaseLLMProvider]:
        """Get the primary provider."""
        return self._primary

    @property
    def fallback(self) -> Optional[BaseLLMProvider]:
        """Get the fallback provider."""
        return self._fallback

    @property
    def current(self) -> Optional[BaseLLMProvider]:
        """Get the currently active provider."""
        return self._current

    @property
    def current_name(self) -> str:
        """Get the name of the current provider."""
        return self._current.name if self._current else "none"

    async def get_available_provider(self) -> BaseLLMProvider:
        """Get an available provider, checking primary then fallback.
        
        Returns:
            An available provider.
        
        Raises:
            ProviderError: If no provider is available.
        """
        # Check primary
        if self._primary:
            if await self._primary.is_available():
                self._current = self._primary
                return self._primary
            else:
                logger.warning(
                    "Primary provider not available", provider=self._primary.name
                )

        # Check fallback
        if self._fallback and self.config.provider.fallback_on_error:
            if await self._fallback.is_available():
                logger.info(
                    "Using fallback provider", provider=self._fallback.name
                )
                self._current = self._fallback
                return self._fallback
            else:
                logger.warning(
                    "Fallback provider not available", provider=self._fallback.name
                )

        raise ProviderError(
            "No LLM provider available. Check your configuration and ensure "
            "Ollama is running or DeepSeek API key is configured.",
            provider="manager",
        )

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response using available provider with fallback.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Returns:
            The generated text response.
        
        Raises:
            ProviderError: If all providers fail.
        """
        # Try primary provider first
        if self._primary:
            try:
                return await self._primary.generate(messages)
            except ProviderError as e:
                logger.warning(
                    "Primary provider failed",
                    provider=self._primary.name,
                    error=str(e),
                )
                if not self.config.provider.fallback_on_error or not self._fallback:
                    raise

        # Try fallback
        if self._fallback and self.config.provider.fallback_on_error:
            try:
                logger.info("Attempting fallback provider", provider=self._fallback.name)
                self._current = self._fallback
                return await self._fallback.generate(messages)
            except ProviderError as e:
                logger.error(
                    "Fallback provider also failed",
                    provider=self._fallback.name,
                    error=str(e),
                )
                raise

        raise ProviderError(
            "All LLM providers failed",
            provider="manager",
        )

    async def generate_streaming(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Generate a streaming response using available provider with fallback.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Yields:
            Text chunks as they arrive from the LLM.
        
        Raises:
            ProviderError: If all providers fail.
        """
        # Try primary provider first
        if self._primary:
            try:
                async for chunk in self._primary.generate_streaming(messages):
                    yield chunk
                return
            except ProviderError as e:
                logger.warning(
                    "Primary provider streaming failed",
                    provider=self._primary.name,
                    error=str(e),
                )
                if not self.config.provider.fallback_on_error or not self._fallback:
                    raise

        # Try fallback
        if self._fallback and self.config.provider.fallback_on_error:
            try:
                logger.info(
                    "Attempting fallback provider for streaming",
                    provider=self._fallback.name,
                )
                self._current = self._fallback
                async for chunk in self._fallback.generate_streaming(messages):
                    yield chunk
                return
            except ProviderError as e:
                logger.error(
                    "Fallback provider streaming also failed",
                    provider=self._fallback.name,
                    error=str(e),
                )
                raise

        raise ProviderError(
            "All LLM providers failed for streaming",
            provider="manager",
        )

    async def check_availability(self) -> dict[str, bool]:
        """Check availability of all configured providers.
        
        Returns:
            Dict mapping provider names to their availability status.
        """
        status = {}

        if self._primary:
            status[self._primary.name] = await self._primary.is_available()

        if self._fallback:
            status[self._fallback.name] = await self._fallback.is_available()

        return status
