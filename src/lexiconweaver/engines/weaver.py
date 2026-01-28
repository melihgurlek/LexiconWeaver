"""Weaver engine for LLM-based translation with glossary enforcement."""

import asyncio
import json
from typing import AsyncIterator

import httpx
from flashtext import KeywordProcessor
from peewee import IntegrityError

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project, TranslationCache
from lexiconweaver.engines.base import BaseEngine
from lexiconweaver.exceptions import TranslationError, WeaverError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.utils.text_processor import extract_paragraphs, generate_hash

logger = get_logger(__name__)


class Weaver(BaseEngine):
    """Generation engine that interfaces with Ollama for translation."""

    def __init__(self, config: Config, project: Project) -> None:
        """Initialize the Weaver engine."""
        self.config = config
        self.project = project
        self.ollama_url = config.ollama.url
        self.model = config.ollama.model
        self.timeout = config.ollama.timeout
        self.max_retries = config.ollama.max_retries
        self.cache_enabled = config.cache.enabled

    def _prepare_messages(
        self, text: str, mini_glossary: dict[str, str], context: str = ""
    ) -> list[dict[str, str]]:
        """Construct the chat messages for Ollama."""

        # 1. Prepare glossary block (strict mapping format).
        glossary_block = "No specific terms."
        if mini_glossary:
            glossary_lines = [
                f"- {src} -> {tgt}" for src, tgt in mini_glossary.items()
            ]
            glossary_block = "\n".join(glossary_lines)

        # 2. System prompt (strict rules).
        system_content = (
           "You are a specialized translation engine for Wuxia/Xianxia fantasy novels.\n"
            "Target Language: Turkish.\n\n"
            "CRITICAL RULES:\n"
            "1. **Glossary as Root:** The glossary provides the ROOT form (Kök) of the word.\n"
            "   - You MUST use this root for the specific term.\n"
            "   - You MUST apply correct Turkish suffixes (sondan ekleme) to this root to fit the sentence grammar.\n"
            "   - You MUST handle consonant mutation (ünsüz yumuşaması) correctly (e.g., Böcek -> Böceği).\n"
            "2. **Protect Proper Nouns:** Do NOT translate or modify Names of People or Places unless they are explicitly in the glossary.\n"
            "   - 'Fang Yuan' -> 'Fang Yuan' (Do not change to 'Fang Kaynak').\n"
            "   - 'Qing Mao Mountain' -> 'Qing Mao Dağı' (Only translate the generic part 'Mountain').\n"
            "3. **Output ONLY the translation.** No notes, headers, or explanations.\n"
            "4. **Genre Tone:** Use 'Klan' for Clan, 'Tarikat' for Sect.\n"
            "5. **No Repetition:** Do not translate the 'CONTEXT' section.\n"

        )

        user_content = (
            "### GLOSSARY (Use as Roots):\n"
            f"{glossary_block}\n\n"
            "### CONTEXT (PREVIOUSLY TRANSLATED - DO NOT TRANSLATE):\n"
            f'{context if context else "No previous context."}\n\n'
            "### CURRENT CHUNK (TRANSLATE THIS):\n"
            f"{text}\n"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    async def translate_paragraph(
        self, paragraph: str, use_cache: bool = True
    ) -> str:
        """Translate a single paragraph."""
        if use_cache and self.cache_enabled:
            cached = self._get_from_cache(paragraph)
            if cached is not None:
                logger.debug("Cache hit for paragraph", hash=generate_hash(paragraph)[:8])
                return cached

        mini_glossary = self._build_mini_glossary(paragraph)

        messages = self._prepare_messages(paragraph, mini_glossary)

        translation = await self._call_ollama(messages)

        self._verify_translation(paragraph, translation, mini_glossary)

        if use_cache and self.cache_enabled:
            self._store_in_cache(paragraph, translation)

        return translation

    async def translate_streaming(
        self, paragraph: str
    ) -> AsyncIterator[str]:
        """Translate a paragraph with streaming response."""
        mini_glossary = self._build_mini_glossary(paragraph)

        messages = self._prepare_messages(paragraph, mini_glossary)

        full_translation = ""
        async for chunk in self._call_ollama_streaming(messages):
            full_translation += chunk
            yield chunk

        self._verify_translation(paragraph, full_translation, mini_glossary)

        if self.cache_enabled:
            self._store_in_cache(paragraph, full_translation)

    async def translate_batch_streaming(
        self, batch: str, context: str = ""
    ) -> AsyncIterator[str]:
        """Translate a batch with streaming response, optionally including context.
        
        Args:
            batch: The text batch to translate (ONLY new text, not merged with context)
            context: Optional context from previous batch (e.g., last 2 sentences)
            
        Yields:
            Translation chunks as they arrive from the LLM
        """
        text_to_translate = batch
        
        mini_glossary = self._build_mini_glossary(text_to_translate)
        
        messages = self._prepare_messages(text_to_translate, mini_glossary, context)
        
        full_translation = ""
        async for chunk in self._call_ollama_streaming(messages):
            full_translation += chunk
            yield chunk
        
        self._verify_translation(text_to_translate, full_translation, mini_glossary)
        
        if self.cache_enabled:
            self._store_in_cache(batch, full_translation)

    async def translate_text(
        self, text: str, use_cache: bool = True
    ) -> str:
        """Translate full text by processing paragraphs."""
        paragraphs = extract_paragraphs(text)
        translated_paragraphs: list[str] = []

        for i, paragraph in enumerate(paragraphs):
            logger.info("Translating paragraph", paragraph=i + 1, total=len(paragraphs))
            try:
                translation = await self.translate_paragraph(paragraph, use_cache)
                translated_paragraphs.append(translation)
            except TranslationError as e:
                logger.error(
                    "Translation failed for paragraph",
                    paragraph=i + 1,
                    error=str(e),
                )
                # Mark as unstable by adding marker
                translated_paragraphs.append(f"[UNSTABLE] {paragraph}")
            except Exception as e:
                logger.error(
                    "Unexpected error translating paragraph",
                    paragraph=i + 1,
                    error=str(e),
                )
                translated_paragraphs.append(f"[ERROR] {paragraph}")

        return "\n\n".join(translated_paragraphs)

    def _build_mini_glossary(self, text: str) -> dict[str, str]:
        """Build a mini-glossary containing only terms that appear in the text."""
        glossary_terms = GlossaryTerm.select().where(
            GlossaryTerm.project == self.project
        )


        keyword_processor = KeywordProcessor(case_sensitive=False)
        term_map: dict[str, str] = {}
        
        for term in glossary_terms:
            if term.is_regex:
                continue

            source_term = term.source_term
            target_term = term.target_term

            keyword_processor.add_keyword(source_term)
            term_map[source_term.lower()] = target_term

        match_text = text.replace("\u00a0", " ")
        found_terms = keyword_processor.extract_keywords(match_text)
        
        mini_glossary: dict[str, str] = {}
        for found_term in dict.fromkeys(found_terms):
            target = term_map.get(found_term.lower())
            if target:
                mini_glossary[found_term] = target

        return mini_glossary

    async def _call_ollama(self, messages: list[dict]) -> str:
        """Call Ollama API for translation with retry logic."""
        url = f"{self.ollama_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.6
            }
        }

        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)

                    if response.status_code != 200:
                        raise WeaverError(
                            f"Ollama API returned status {response.status_code}: {response.text}"
                        )

                    result = response.json()
                    return result.get("message", {}).get("content", "").strip()

            except httpx.TimeoutException as e:
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt
                    logger.warning(
                        "Ollama timeout, retrying",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise TranslationError(
                        f"Ollama request timed out after {self.max_retries + 1} attempts"
                    ) from e

            except httpx.RequestError as e:
                raise TranslationError(
                    f"Failed to connect to Ollama at {self.ollama_url}: {e}"
                ) from e

            except Exception as e:
                raise TranslationError(f"Unexpected error calling Ollama: {e}") from e

        raise TranslationError("Failed to get translation after retries")

    async def _call_ollama_streaming(
        self, messages: list[dict]
    ) -> AsyncIterator[str]:
        """Call Ollama API with streaming response."""
        url = f"{self.ollama_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": 0.6}
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        text = await response.aread()
                        raise WeaverError(
                            f"Ollama API returned status {response.status_code}: {text.decode()}"
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
            raise TranslationError(f"Ollama streaming request timed out: {e}") from e
        except httpx.RequestError as e:
            raise TranslationError(
                f"Failed to connect to Ollama at {self.ollama_url}: {e}"
            ) from e
        except Exception as e:
            raise TranslationError(f"Unexpected error streaming from Ollama: {e}") from e

    def _verify_translation(
        self, source: str, translation: str, mini_glossary: dict[str, str]
    ) -> None:
        """Verify that all glossary terms in source appear in translation."""
        missing_terms: list[str] = []

        source_lower = source.lower()
        translation_lower = translation.lower()

        for source_term, target_term in mini_glossary.items():
            source_term_lower = source_term.lower()
            target_term_lower = target_term.lower()

            if source_term_lower not in source_lower:
                continue

            if target_term_lower in translation_lower:
                continue

            if len(target_term_lower) >= 4:
                root_len = int(len(target_term_lower) * 0.7)
                root_guess = target_term_lower[:root_len]
                
                if root_guess in translation_lower:
                    continue

            missing_terms.append(source_term)

        if missing_terms:
            logger.warning(
                "Translation verification failed",
                missing_terms=missing_terms,
                source_hash=generate_hash(source)[:8],
            )

    def _get_from_cache(self, paragraph: str) -> str | None:
        """Get translation from cache if available."""
        try:
            project_id = getattr(self.project, "id", None)
            cache_hash = (
                generate_hash(f"{project_id}:{paragraph}") if project_id is not None else generate_hash(paragraph)
            )

            cached = TranslationCache.get_or_none(
                TranslationCache.hash == cache_hash,
                TranslationCache.project == self.project,
            )

            if cached is None:
                legacy_hash = generate_hash(paragraph)
                cached = TranslationCache.get_or_none(
                    TranslationCache.hash == legacy_hash,
                    TranslationCache.project == self.project,
                )
            return cached.translation if cached else None
        except Exception as e:
            logger.warning("Failed to get from cache", error=str(e))
            return None

    def _store_in_cache(self, paragraph: str, translation: str) -> None:
        """Store translation in cache."""
        try:
            project_id = getattr(self.project, "id", None)
            cache_hash = (
                generate_hash(f"{project_id}:{paragraph}") if project_id is not None else generate_hash(paragraph)
            )

            try:
                TranslationCache.create(
                    hash=cache_hash,
                    project=self.project,
                    translation=translation,
                )
            except IntegrityError:
                (
                    TranslationCache.update(translation=translation, project=self.project)
                    .where(TranslationCache.hash == cache_hash)
                    .execute()
                )
        except Exception as e:
            logger.warning("Failed to store in cache", error=str(e))

    def process(self, text: str) -> str:
        """Process text synchronously (wrapper for async method)."""
        return asyncio.run(self.translate_text(text))
