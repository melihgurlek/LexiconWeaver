"""Ollama LLM provider implementation."""

import asyncio
import json
from typing import AsyncIterator

import httpx

from lexiconweaver.config import OllamaConfig
from lexiconweaver.exceptions import ProviderError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.providers.base import BaseLLMProvider

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """LLM provider for local Ollama server."""

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize the Ollama provider.
        
        Args:
            config: Ollama configuration with url, model, timeout, max_retries.
        """
        self.url = config.url
        self.model = config.model
        self.timeout = config.timeout
        self.max_retries = config.max_retries

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "ollama"

    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Returns:
            The generated text response.
        
        Raises:
            ProviderError: If the API call fails after retries.
        """
        url = f"{self.url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.6},
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)

                    if response.status_code != 200:
                        raise ProviderError(
                            f"Ollama API returned status {response.status_code}: {response.text}",
                            provider="ollama",
                        )

                    result = response.json()
                    return result.get("message", {}).get("content", "").strip()

            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    logger.warning(
                        "Ollama timeout, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise ProviderError(
                        f"Ollama request timed out after {self.max_retries + 1} attempts",
                        provider="ollama",
                    ) from e

            except httpx.RequestError as e:
                raise ProviderError(
                    f"Failed to connect to Ollama at {self.url}: {e}",
                    provider="ollama",
                ) from e

            except ProviderError:
                raise

            except Exception as e:
                raise ProviderError(
                    f"Unexpected error calling Ollama: {e}",
                    provider="ollama",
                ) from e

        raise ProviderError(
            "Failed to get response from Ollama after retries",
            provider="ollama",
        )

    async def generate_streaming(
        self, messages: list[dict[str, str]]
    ) -> AsyncIterator[str]:
        """Generate a streaming response from Ollama.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
        
        Yields:
            Text chunks as they arrive from Ollama.
        
        Raises:
            ProviderError: If the API call fails.
        """
        url = f"{self.url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.6},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        text = await response.aread()
                        raise ProviderError(
                            f"Ollama API returned status {response.status_code}: {text.decode()}",
                            provider="ollama",
                        )

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.TimeoutException as e:
            raise ProviderError(
                f"Ollama streaming request timed out: {e}",
                provider="ollama",
            ) from e
        except httpx.RequestError as e:
            raise ProviderError(
                f"Failed to connect to Ollama at {self.url}: {e}",
                provider="ollama",
            ) from e
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(
                f"Unexpected error streaming from Ollama: {e}",
                provider="ollama",
            ) from e
