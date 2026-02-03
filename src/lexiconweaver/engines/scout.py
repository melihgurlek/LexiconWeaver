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
        self._word_to_sentence_ids: dict[str, list[int]] = {}
        self._last_freq_map: Counter[str] | None = None
        self._last_caps_map: Counter[str] | None = None

    def _build_ngram_maps(
        self, text: str
    ) -> tuple[Counter[str], Counter[str]]:
        """Build frequency and capitalization maps in one pass over tokenized text.

        Returns:
            (freq_map, caps_map): lowercased n-gram -> count. caps_map counts
            occurrences where the span in the original text contains uppercase.
        """
        tokens: list[tuple[str, int, int]] = [
            (m.group(), m.start(), m.end())
            for m in re.finditer(r"\b\w+\b", text)
        ]
        freq_map: Counter[str] = Counter()
        caps_map: Counter[str] = Counter()
        for i in range(len(tokens)):
            for n in range(1, min(self.max_ngram_size + 1, len(tokens) - i + 1)):
                span_tokens = tokens[i : i + n]
                ngram_lower = " ".join(t[0].lower() for t in span_tokens)
                start = span_tokens[0][1]
                end = span_tokens[-1][2]
                freq_map[ngram_lower] += 1
                if any(c.isupper() for c in text[start:end]):
                    caps_map[ngram_lower] += 1
        return freq_map, caps_map

    def get_last_ngram_maps(
        self,
    ) -> tuple[Counter[str], Counter[str]] | None:
        """Return the frequency and caps maps from the last process() run for reuse.

        Returns (freq_map, caps_map) if process() has been called, else None.
        Refiner can use these for missed-term frequency without rescanning text.
        """
        if self._last_freq_map is None or self._last_caps_map is None:
            return None
        return self._last_freq_map, self._last_caps_map

    def _build_sentence_index(self) -> None:
        """Build inverted index: word_lower -> list of sentence indices."""
        self._word_to_sentence_ids = {}
        for i, sentence in enumerate(self._sentences):
            words = re.findall(r"\b\w+\b", sentence.lower())
            for w in words:
                if w not in self._word_to_sentence_ids:
                    self._word_to_sentence_ids[w] = []
                self._word_to_sentence_ids[w].append(i)

    def process(self, text: str) -> list[CandidateTerm]:
        """Process text and return candidate terms with confidence scores."""
        try:
            self._sentences = self._split_into_sentences(text)
            self._build_sentence_index()
            ignored_terms = self._get_ignored_terms()
            candidates, pattern_matches = self._extract_candidates(text)
            filtered = self._filter_candidates(candidates, ignored_terms)
            freq_map, caps_map = self._build_ngram_maps(text)
            self._last_freq_map = freq_map
            self._last_caps_map = caps_map

            # Calculate confidence scores
            scored = self._score_candidates(
                filtered, text, freq_map, caps_map, pattern_matches
            )

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

    def _find_context_sentence(self, term: str) -> str | None:
        """Find the sentence containing the term for context using inverted index."""
        term_words = term.lower().split()
        if not term_words:
            return None
        # Use first word to get candidate sentence IDs; intersect with others if multiple words
        sentence_ids = set(self._word_to_sentence_ids.get(term_words[0], []))
        for w in term_words[1:]:
            sentence_ids &= set(self._word_to_sentence_ids.get(w, []))
        term_lower = term.lower()
        for sid in sentence_ids:
            sentence = self._sentences[sid]
            if term_lower not in sentence.lower():
                continue
            if len(sentence) > 200:
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

    def _extract_candidates(self, text: str) -> tuple[list[str], dict[str, str]]:
        """Extract candidate terms from text using multiple strategies.

        Returns:
            (candidates, pattern_matches): list of candidate strings and
            dict mapping candidate_lower -> pattern name for definition matches.
        """
        candidates: set[str] = set()
        pattern_matches: dict[str, str] = {}

        # Strategy 1: Extract from definition patterns
        for pattern, pattern_name in self.DEFINITION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).strip()
                if term:
                    term = self._clean_term(term)
                    if term:
                        candidates.add(term)
                        pattern_matches[term.strip().lower()] = pattern_name

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

        return list(candidates), pattern_matches

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
        self,
        candidates: list[str],
        text: str,
        freq_map: Counter[str],
        caps_map: Counter[str],
        pattern_matches: dict[str, str],
    ) -> list[CandidateTerm]:
        """Calculate confidence scores for candidates using prebuilt maps."""
        scored: list[CandidateTerm] = []

        def normalize(s: str) -> str:
            return " ".join(s.lower().split())

        # Normalize candidate for lookup; fallback count only when missing
        frequencies: dict[str, int] = {}
        caps_counts: dict[str, int] = {}
        for candidate in candidates:
            key = normalize(candidate)
            freq = freq_map.get(key, 0)
            if freq == 0 and key != candidate.lower():
                freq = text.lower().count(key)
            frequencies[candidate] = freq
            caps_counts[candidate] = caps_map.get(key, 0)

        max_freq = max(frequencies.values()) if frequencies else 1
        max_caps = max(caps_counts.values()) if caps_counts else 1

        for candidate in candidates:
            freq = frequencies[candidate]
            caps = caps_counts[candidate]
            has_pattern = pattern_matches.get(candidate.lower()) is not None

            freq_score = min(freq / max_freq, 1.0) * 0.25
            caps_score = min(caps / max_caps, 1.0) * 0.25
            pattern_score = 0.35 if has_pattern else 0.0
            word_count = len(candidate.split())
            length_score = min(word_count / 3, 1.0) * 0.15
            confidence = freq_score + caps_score + pattern_score + length_score

            context = self._find_context_sentence(candidate)

            scored.append(
                CandidateTerm(
                    term=candidate,
                    confidence=confidence,
                    frequency=freq,
                    context_pattern=pattern_matches.get(candidate.lower()),
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
