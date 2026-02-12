"""Global Scout engine for batch chapter analysis with burst detection."""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from lexiconweaver.config import Config
from lexiconweaver.database.models import BurstTerm, GlossaryTerm, IgnoredTerm, Project
from lexiconweaver.engines.base import BaseEngine
from lexiconweaver.logging_config import get_logger
from lexiconweaver.providers import LLMProviderManager
from lexiconweaver.utils.text_processor import extract_ngrams

logger = get_logger(__name__)


@dataclass
class Chapter:
    """Represents a chapter for global scouting."""
    
    number: int
    text: str
    filename: str


@dataclass
class BurstTermCandidate:
    """Candidate term detected by burst analysis."""
    
    term: str
    max_window_frequency: int
    global_frequency: int
    first_chapter: int
    density_score: float
    chapters_appeared: list[int]


class GlobalScout(BaseEngine):
    """
    Global scout engine for multi-chapter term discovery.
    
    Uses frequency analysis, burst detection, and capitalization patterns
    to discover important terms across multiple chapters.
    """
    
    def __init__(self, config: Config, project: Project):
        """
        Initialize Global Scout.
        
        Args:
            config: Application configuration
            project: Project context for database operations
        """
        self.config = config
        self.project = project
        self.provider_manager = LLMProviderManager(config)
    
    def analyze_chapters(
        self,
        chapters: list[Chapter],
        min_frequency: int = 3,
        window_size: int = 3,
        burst_threshold: int = 5,
        progress_callback: Callable[[str], None] | None = None
    ) -> list[BurstTermCandidate]:
        """
        Analyze chapters to find important terms using burst detection.
        
        Args:
            chapters: List of Chapter objects to analyze
            min_frequency: Minimum global frequency to consider
            window_size: Sliding window size for burst detection (default: 3 chapters)
            burst_threshold: Minimum frequency in window to flag as burst
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of BurstTermCandidate objects sorted by density score
        """
        if progress_callback:
            progress_callback(f"Analyzing {len(chapters)} chapters...")
        
        existing_terms = self._get_existing_terms()
        ignored_terms = self._get_ignored_terms()
        
        if progress_callback:
            progress_callback("Extracting n-grams...")
        
        term_frequencies = self._extract_ngrams_from_chapters(chapters)
        
        if progress_callback:
            progress_callback("Analyzing capitalization patterns...")
        
        capitalized_terms = self._track_capitalization(chapters, term_frequencies)
        
        if progress_callback:
            progress_callback("Detecting burst terms...")
        
        burst_terms = self._detect_burst_terms(
            term_frequencies,
            len(chapters),
            window_size,
            burst_threshold,
            min_frequency
        )
        
        if progress_callback:
            progress_callback("Filtering candidates...")
        
        filtered_burst_terms = []
        for burst_term in burst_terms:
            term_lower = burst_term.term.lower()
            
            if term_lower in existing_terms or term_lower in ignored_terms:
                continue
            
            if term_lower in capitalized_terms:
                burst_term.density_score *= 1.2
            
            filtered_burst_terms.append(burst_term)
        
        filtered_burst_terms.sort(key=lambda t: t.density_score, reverse=True)
        
        if progress_callback:
            progress_callback(f"Found {len(filtered_burst_terms)} candidate terms")
        
        return filtered_burst_terms
    
    def _extract_ngrams_from_chapters(
        self, chapters: list[Chapter]
    ) -> dict[str, list[int]]:
        """
        Extract n-grams (1-4 words) from all chapters and track per-chapter frequency.
        
        Args:
            chapters: List of chapters to analyze
            
        Returns:
            Dictionary mapping term -> list of frequencies per chapter
        """
        term_frequencies = defaultdict(lambda: [0] * len(chapters))
        
        for ch_idx, chapter in enumerate(chapters):
            for n in range(1, 5):
                ngrams = extract_ngrams(chapter.text, n)
                
                for ngram_tuple in ngrams:
                    ngram_text = ngram_tuple[0] if isinstance(ngram_tuple, tuple) else ngram_tuple
                    
                    ngram_clean = " ".join(ngram_text.split())
                    
                    if len(ngram_clean) < 3:
                        continue
                    
                    if self._is_common_prefix(ngram_clean):
                        continue
                    
                    term_frequencies[ngram_clean][ch_idx] += 1
        
        return term_frequencies
    
    def _detect_burst_terms(
        self,
        term_frequencies: dict[str, list[int]],
        num_chapters: int,
        window_size: int,
        burst_threshold: int,
        min_global_frequency: int
    ) -> list[BurstTermCandidate]:
        """
        Detect terms with high local density using sliding window.
        
        This catches:
        - Late-appearing terms (new skills in ch49+)
        - Arc-specific terms (tournament terms in ch20-25)
        - Foreshadowed terms (mentioned once in ch1, important in ch50+)
        
        Args:
            term_frequencies: Dictionary of term -> frequency list
            num_chapters: Total number of chapters
            window_size: Sliding window size (default: 3)
            burst_threshold: Minimum frequency in window
            min_global_frequency: Minimum total frequency
            
        Returns:
            List of BurstTermCandidate objects
        """
        burst_terms = []
        
        for term, freq_list in term_frequencies.items():
            global_freq = sum(freq_list)
            
            if global_freq < min_global_frequency:
                continue
            
            max_window_freq = 0
            best_window_start = 0
            
            for i in range(num_chapters - window_size + 1):
                window_sum = sum(freq_list[i:i+window_size])
                if window_sum > max_window_freq:
                    max_window_freq = window_sum
                    best_window_start = i
            
            if max_window_freq >= burst_threshold:
                first_chapter = next((i for i, count in enumerate(freq_list) if count > 0), 0)
                
                chapters_appeared = [i for i, count in enumerate(freq_list) if count > 0]
                
                density_score = max_window_freq / global_freq
                
                burst_terms.append(BurstTermCandidate(
                    term=term,
                    max_window_frequency=max_window_freq,
                    global_frequency=global_freq,
                    first_chapter=first_chapter,
                    density_score=density_score,
                    chapters_appeared=chapters_appeared
                ))
        
        return burst_terms
    
    def _track_capitalization(
        self,
        chapters: list[Chapter],
        term_frequencies: dict[str, list[int]]
    ) -> set[str]:
        """
        Track consistently capitalized terms.
        
        Args:
            chapters: List of chapters
            term_frequencies: Term frequency data
            
        Returns:
            Set of term (lowercase) that are consistently capitalized
        """
        consistently_capitalized = set()
        
        for term in term_frequencies.keys():
            if term and term[0].isupper():
                term_lower = term.lower()
                global_freq = sum(term_frequencies[term])
                
                if " " in term or global_freq >= 3:
                    consistently_capitalized.add(term_lower)
        
        return consistently_capitalized
    
    def _is_common_prefix(self, term: str) -> bool:
        """
        Check if term starts with common words that shouldn't be terms.
        
        Args:
            term: Term to check
            
        Returns:
            True if starts with common prefix
        """
        common_prefixes = [
            "the ", "a ", "an ", "this ", "that ", "these ", "those ",
            "his ", "her ", "my ", "your ", "our ", "their ",
            "was ", "were ", "is ", "are ", "will ", "would ", "could ", "should ",
            "and ", "or ", "but ", "if ", "when ", "where ", "how ", "why ",
            "said ", "says ", "told ", "asked ",
        ]
        
        term_lower = term.lower()
        for prefix in common_prefixes:
            if term_lower.startswith(prefix):
                return True
        
        return False
    
    def _get_existing_terms(self) -> set[str]:
        """Get set of existing glossary terms (lowercase) for this project."""
        terms = GlossaryTerm.select().where(GlossaryTerm.project == self.project)
        return {term.source_term.lower() for term in terms}
    
    def _get_ignored_terms(self) -> set[str]:
        """Get set of ignored terms (lowercase) for this project."""
        terms = IgnoredTerm.select().where(IgnoredTerm.project == self.project)
        return {term.term.lower() for term in terms}
    
    async def refine_with_llm(
        self,
        candidates: list[BurstTermCandidate],
        top_percent: int = 20,
        progress_callback: Callable[[str], None] | None = None
    ) -> list[dict]:
        """
        Refine top N% of candidates using LLM.
        
        Sends candidates to LLM for validation and translation proposals.
        
        Args:
            candidates: List of burst term candidates
            top_percent: Percentage of top candidates to refine (default: 20%)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of dictionaries with term, translation, category, reasoning
        """
        if not candidates:
            return []
        
        num_to_refine = max(1, int(len(candidates) * top_percent / 100))
        top_candidates = candidates[:num_to_refine]
        
        if progress_callback:
            progress_callback(f"Refining top {num_to_refine} candidates with LLM...")
        
        refined_terms = []
        
        batch_size = 10
        for i in range(0, len(top_candidates), batch_size):
            batch = top_candidates[i:i+batch_size]
            
            if progress_callback:
                progress_callback(f"Processing batch {i//batch_size + 1}...")
            
            terms_list = "\n".join([
                f"{idx+1}. {cand.term} (frequency: {cand.global_frequency}, density: {cand.density_score:.2f})"
                for idx, cand in enumerate(batch)
            ])
            
            prompt = f"""Analyze these terms from a fantasy/xianxia novel and determine which are important terms that should be added to a translation glossary.

                        For each term that IS important (names, techniques, ranks, locations, items, etc.), provide:
                        - The term
                        - Suggested Turkish translation
                        - Category (Person, Location, Technique, Item, Rank, etc.)
                        - Brief reasoning

                        For terms that are NOT important (common phrases, generic words), skip them.

                        Terms to analyze:
                        {terms_list}

                        Respond in this format for each important term:
                        TERM: [term]
                        TRANSLATION: [suggested translation]
                        CATEGORY: [category]
                        REASONING: [why this is important]
                        ---

                        Only include terms worth adding to glossary."""
            
            messages = [
                {"role": "system", "content": "You are an expert at identifying important terms in fantasy novels for translation glossaries."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response = await self.provider_manager.generate(messages)
                
                parsed = self._parse_llm_refinement(response)
                refined_terms.extend(parsed)
                
            except Exception as e:
                logger.error(f"LLM refinement failed for batch: {e}")
                continue
        
        if progress_callback:
            progress_callback(f"LLM refined {len(refined_terms)} terms")
        
        return refined_terms
    
    def _parse_llm_refinement(self, response: str) -> list[dict]:
        """
        Parse LLM refinement response.
        
        Args:
            response: LLM response text
            
        Returns:
            List of dictionaries with term info
        """
        refined = []
        
        entries = response.split("---")
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            
            term = None
            translation = None
            category = None
            reasoning = None
            
            for line in entry.split("\n"):
                line = line.strip()
                if line.startswith("TERM:"):
                    term = line[5:].strip()
                elif line.startswith("TRANSLATION:"):
                    translation = line[12:].strip()
                elif line.startswith("CATEGORY:"):
                    category = line[9:].strip()
                elif line.startswith("REASONING:"):
                    reasoning = line[10:].strip()
            
            if term and translation:
                refined.append({
                    "term": term,
                    "translation": translation,
                    "category": category or "Unknown",
                    "reasoning": reasoning or ""
                })
        
        return refined
    
    def save_burst_terms_to_db(self, candidates: list[BurstTermCandidate]) -> int:
        """
        Save burst term candidates to database.
        
        Args:
            candidates: List of burst term candidates
            
        Returns:
            Number of terms saved
        """
        saved_count = 0
        
        for candidate in candidates:
            try:
                BurstTerm.create(
                    project=self.project,
                    term=candidate.term,
                    first_chapter=candidate.first_chapter,
                    max_window_frequency=candidate.max_window_frequency,
                    global_frequency=candidate.global_frequency,
                    density_score=candidate.density_score,
                    status="pending"
                )
                saved_count += 1
            except Exception as e:
                logger.warning(f"Failed to save burst term '{candidate.term}': {e}")
                continue
        
        logger.info(f"Saved {saved_count} burst terms to database")
        return saved_count
    
    def process(self, text: str) -> list[BurstTermCandidate]:
        """
        Process text to find burst terms (BaseEngine interface implementation).
        
        This is a simplified interface for single-text processing.
        For multi-chapter analysis, use analyze_chapters() directly.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of BurstTermCandidate objects
        """
        chapter = Chapter(number=1, text=text, filename="text")
        
        candidates = self.analyze_chapters(
            [chapter],
            min_frequency=self.config.batch.scout_min_frequency,
            window_size=self.config.batch.scout_burst_window_size,
            burst_threshold=self.config.batch.scout_burst_threshold
        )
        
        return candidates
