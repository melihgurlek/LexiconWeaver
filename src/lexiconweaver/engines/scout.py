"""Scout engine for discovering potential terms in text."""

import re
from collections import Counter
from typing import NamedTuple

from lexiconweaver.config import Config
from lexiconweaver.database.models import IgnoredTerm, Project
from lexiconweaver.engines.base import BaseEngine
from lexiconweaver.exceptions import ScoutError
from lexiconweaver.logging_config import get_logger
from lexiconweaver.utils.cache import get_cache
from lexiconweaver.utils.text_processor import extract_ngrams

logger = get_logger(__name__)


class CandidateTerm(NamedTuple):
    """Represents a candidate term with confidence score."""

    term: str
    confidence: float
    frequency: int
    context_pattern: str | None
    context_snippet: str | None


class Scout(BaseEngine):
    """Discovery engine that identifies potential terms using heuristics."""

    # Cache for spaCy stopwords (loaded lazily)
    _spacy_stopwords: set[str] | None = None

    # Definition patterns that indicate a term
    DEFINITION_PATTERNS = [
        (r"called\s+([A-Z][\w\s]+)", "called"),
        (r"known\s+as\s+([A-Z][\w\s]+)", "known as"),
        (r"rank\s+(\d+|\w+)", "rank"),
        (r'["\'`]([A-Z][\w\s]+)["\'`]', "quoted"),
        (r"the\s+([A-Z][\w]+(?:\s+[A-Z][\w]+)*)", "the capitalized"),
        (r"named\s+([A-Z][\w\s]+)", "named"),
        (r"([A-Z][\w\s]+)\s+technique", "technique"),
        (r"([A-Z][\w\s]+)\s+realm", "realm"),
        (r"([A-Z][\w\s]+)\s+stage", "stage"),
        (r"([A-Z][\w\s]+)\s+clan", "clan"),
        (r"([A-Z][\w\s]+)\s+sect", "sect"),
        (r"([A-Z][\w\s]+)\s+mountain", "mountain"),
        (r"([A-Z][\w\s]+)\s+valley", "valley"),
        (r"([A-Z][\w\s]+)\s+village", "village"),
        (r"([A-Z][\w\s]+)\s+city", "city"),
        (r"([A-Z][\w\s]+)\s+pill", "pill"),
        (r"([A-Z][\w\s]+)\s+Gu", "gu"),
        (r"([A-Z][\w\s]+)\s+Scripture", "scripture"),
        (r"([A-Z][\w\s]+)\s+Manual", "manual"),
        (r"([A-Z][\w\s]+)\s+Art", "art"),
        (r"([A-Z][\w\s]+)\s+Master", "master"),
        (r"([A-Z][\w\s]+)\s+Elder", "elder"),
        (r"([A-Z][\w\s]+)\s+Lord", "lord"),
        (r"Elder\s+([A-Z][\w\s]+)", "elder prefix"),
        (r"Master\s+([A-Z][\w\s]+)", "master prefix"),
        (r"Lord\s+([A-Z][\w\s]+)", "lord prefix"),
        (r"Young\s+Master\s+([A-Z][\w\s]+)", "young master"),
        (r"(\w+)\s+Immortal", "immortal"),
        (r"(\w+)\s+Demon", "demon"),
        (r"(\w+)\s+Devil", "devil"),
    ]

    # Words that commonly follow proper nouns but shouldn't be included
    COMMON_SUFFIXES = {
        "technique", "realm", "stage", "clan", "sect", "mountain", "valley",
        "village", "city", "pill", "gu", "scripture", "manual", "art",
        "master", "elder", "lord", "immortal", "demon", "devil",
    }

    # Simple POS-like filtering - words that are likely not terms
    UNLIKELY_TERMS = {
        "however", "therefore", "moreover", "furthermore", "meanwhile",
        "suddenly", "immediately", "quickly", "slowly", "finally",
        "perhaps", "probably", "certainly", "definitely", "clearly",
        "actually", "really", "simply", "merely", "just", "only",
        "before", "after", "during", "while", "when", "where", "why",
        "how", "what", "which", "who", "whose", "whom", "then", "now",
    }

    def __init__(self, config: Config, project: Project | None = None) -> None:
        """Initialize the Scout engine."""
        self.config = config
        self.project = project
        self.min_confidence = config.scout.min_confidence
        self.max_ngram_size = config.scout.max_ngram_size
        self._cache = get_cache()
        self._sentences: list[str] = []

    def process(self, text: str) -> list[CandidateTerm]:
        """Process text and return candidate terms with confidence scores."""
        try:
            self._sentences = self._split_into_sentences(text)
            ignored_terms = self._get_ignored_terms()
            candidates = self._extract_candidates(text)
            filtered = self._filter_candidates(candidates, ignored_terms)

            # Calculate confidence scores
            scored = self._score_candidates(filtered, text)

            # Filter by minimum confidence
            final = [
                c for c in scored if c.confidence >= self.min_confidence
            ]

            # Sort by confidence (descending)
            final.sort(key=lambda x: x.confidence, reverse=True)

            # Deduplicate overlapping terms (prefer longer matches) 
            # TODO: Not sure about longer matches, maybe we should use a different algorithm?
            final = self._deduplicate_overlapping(final)

            logger.info(
                "Scout processed text",
                total_candidates=len(candidates),
                filtered=len(filtered),
                final=len(final),
            )

            return final

        except Exception as e:
            raise ScoutError(f"Failed to process text: {e}") from e

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences for context extraction."""
        pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_context_sentence(self, term: str, text: str) -> str | None:
        """Find the sentence containing the term for context."""
        term_lower = term.lower()
        for sentence in self._sentences:
            if term_lower in sentence.lower():
                # Truncate if too long (max ~200 chars)
                if len(sentence) > 200:
                    # Try to find the term position and extract around it
                    idx = sentence.lower().find(term_lower)
                    start = max(0, idx - 80)
                    end = min(len(sentence), idx + len(term) + 80)
                    snippet = sentence[start:end]
                    if start > 0:
                        snippet = "..." + snippet
                    if end < len(sentence):
                        snippet = snippet + "..."
                    return snippet
                return sentence
        return None

    def _extract_candidates(self, text: str) -> list[str]:
        """Extract candidate terms from text using multiple strategies."""
        candidates: set[str] = set()

        # Strategy 1: Extract from definition patterns
        for pattern, _ in self.DEFINITION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                if term:
                    term = self._clean_term(term)
                    if term:
                        candidates.add(term)

        # Strategy 2: Extract capitalized phrases (Proper Nouns)
        capitalized_pattern = r"(?<![.!?]\s)(?<!\A)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"
        matches = re.finditer(capitalized_pattern, text)
        for match in matches:
            term = match.group(1).strip()
            if len(term.split()) <= self.max_ngram_size:
                candidates.add(term)

        # Also get capitalized phrases at start of sentences
        for sentence in self._sentences:
            if sentence:
                # Get first capitalized word(s) that look like proper nouns
                # (not common sentence starters)
                first_words = sentence.split()[:self.max_ngram_size]
                proper_phrase = []
                for word in first_words:
                    if word[0].isupper() and word.lower() not in self.UNLIKELY_TERMS:
                        if word != word.upper() and len(word) > 1:
                            proper_phrase.append(word)
                    else:
                        break
                if len(proper_phrase) >= 2:
                    candidates.add(" ".join(proper_phrase))

        # Strategy 3: Extract N-grams (up to max_ngram_size)
        for n in range(2, self.max_ngram_size + 1):  # Start from 2 to avoid single common words
            for ngram, _ in extract_ngrams(text, n, min_length=3):
                # Only add if it contains at least one capitalized word
                words = ngram.split()
                if any(w[0].isupper() for w in words if w and len(w) > 0):
                    candidates.add(ngram)

        return list(candidates)

    def _clean_term(self, term: str) -> str:
        """Clean up a term by removing trailing common suffixes."""
        words = term.split()
        while words and words[-1].lower() in self.COMMON_SUFFIXES:
            words.pop()
        return " ".join(words)

    def _filter_candidates(
        self, candidates: list[str], ignored_terms: set[str]
    ) -> list[str]:
        """Filter out stopwords and ignored terms."""
        filtered = []
        stopwords = self._get_stopwords()

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Skip if in ignored terms
            if candidate_lower in ignored_terms:
                continue

            words = candidate_lower.split()
            if all(word in stopwords for word in words):
                continue

            if candidate_lower in self.UNLIKELY_TERMS:
                continue

            # Skip single-letter or very short terms
            if len(candidate.strip()) < 2:
                continue

            if candidate.isdigit():
                continue

            filtered.append(candidate)

        return filtered

    def _score_candidates(
        self, candidates: list[str], text: str
    ) -> list[CandidateTerm]:
        """Calculate confidence scores for candidates."""
        scored: list[CandidateTerm] = []

        # Count frequencies
        text_lower = text.lower()
        frequencies = Counter()
        capitalization_counts = Counter()
        pattern_matches: dict[str, str | None] = {}

        for candidate in candidates:
            candidate_lower = candidate.lower()

            # Count frequency
            count = text_lower.count(candidate_lower)
            frequencies[candidate] = count

            # Count capitalization occurrences
            capitalized_pattern = re.escape(candidate)
            caps_matches = len(re.findall(rf"\b{capitalized_pattern}\b", text))
            capitalization_counts[candidate] = caps_matches

            # Check definition patterns
            matched_pattern = None
            for pattern, pattern_name in self.DEFINITION_PATTERNS:
                try:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match and match.group(1).strip().lower() == candidate_lower:
                        matched_pattern = pattern_name
                        break
                except (IndexError, AttributeError):
                    continue
            pattern_matches[candidate] = matched_pattern

        # Calculate scores
        max_freq = max(frequencies.values()) if frequencies else 1
        max_caps = max(capitalization_counts.values()) if capitalization_counts else 1

        for candidate in candidates:
            freq = frequencies[candidate]
            caps = capitalization_counts[candidate]
            has_pattern = pattern_matches[candidate] is not None

            # Frequency weight (25%)
            freq_score = min(freq / max_freq, 1.0) * 0.25

            # Capitalization weight (25%)
            caps_score = min(caps / max_caps, 1.0) * 0.25

            # Structural context weight (35%)
            pattern_score = 1.0 * 0.35 if has_pattern else 0.0

            # Length bonus (15%) - prefer multi-word terms
            word_count = len(candidate.split())
            length_score = min(word_count / 3, 1.0) * 0.15

            confidence = freq_score + caps_score + pattern_score + length_score

            # Get context sentence for the term
            context = self._find_context_sentence(candidate, text)

            scored.append(
                CandidateTerm(
                    term=candidate,
                    confidence=confidence,
                    frequency=freq,
                    context_pattern=pattern_matches.get(candidate),
                    context_snippet=context,
                )
            )

        return scored

    def _deduplicate_overlapping(
        self, candidates: list[CandidateTerm]
    ) -> list[CandidateTerm]:
        """Remove overlapping terms, preferring longer ones."""
        if not candidates:
            return candidates

        # Sort by term length (descending) then by confidence
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (-len(x.term), -x.confidence)
        )

        result: list[CandidateTerm] = []
        used_terms: set[str] = set()

        for candidate in sorted_candidates:
            candidate_lower = candidate.term.lower()
            
            is_substring = any(
                candidate_lower in used_term and candidate_lower != used_term
                for used_term in used_terms
            )
            
            if not is_substring:
                result.append(candidate)
                used_terms.add(candidate_lower)

        # Re-sort by confidence
        result.sort(key=lambda x: x.confidence, reverse=True)
        return result

    @classmethod
    def _get_stopwords(cls) -> set[str]:
        """Get spaCy stopwords, loading them lazily if needed.
        
        Returns:
            Set of stopwords (lowercased)
        """
        if cls._spacy_stopwords is None:
            try:
                from spacy.lang.en.stop_words import STOP_WORDS
                cls._spacy_stopwords = set(STOP_WORDS)
                logger.debug("Loaded spaCy stopwords", count=len(cls._spacy_stopwords))
            except ImportError:
                logger.warning("spaCy not available, using minimal stopword set")
                # Fallback to minimal set if spaCy is not installed
                cls._spacy_stopwords = {
                    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
                    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
                    "be", "have", "has", "had", "do", "does", "did", "will", "would",
                    "should", "could", "may", "might", "must", "can", "this", "that",
                    "these", "those", "i", "you", "he", "she", "it", "we", "they",
                }
        
        return cls._spacy_stopwords
    
    def _get_ignored_terms(self) -> set[str]:
        """Get ignored terms for the current project, using cache if available."""
        def _fetch_ignored_terms(project: Project | None) -> set[str]:
            """Fetch ignored terms from database."""
            if project is None:
                return set()
            try:
                ignored = IgnoredTerm.select().where(
                    IgnoredTerm.project == project
                )
                return {term.term.lower() for term in ignored}
            except Exception as e:
                logger.warning("Failed to load ignored terms", error=str(e))
                return set()
        
        return self._cache.get_ignored_terms(self.project, _fetch_ignored_terms)
