"""Term highlighting utilities using Longest Match First algorithm."""

from typing import NamedTuple


class HighlightSpan(NamedTuple):
    """Represents a highlighted span in text."""

    start: int
    end: int
    term: str
    is_confirmed: bool  # True for confirmed terms (green), False for candidates (yellow)


class LongestMatchHighlighter:
    """Highlighter that uses Longest Match First algorithm for overlapping terms."""

    def __init__(self) -> None:
        """Initialize the highlighter."""
        self.confirmed_terms: set[str] = set()
        self.candidate_terms: set[str] = set()

    def add_confirmed_term(self, term: str) -> None:
        """Add a confirmed term (will be highlighted in green)."""
        self.confirmed_terms.add(term.lower())

    def add_candidate_term(self, term: str) -> None:
        """Add a candidate term (will be highlighted in yellow)."""
        self.candidate_terms.add(term.lower())

    def find_spans(self, text: str) -> list[HighlightSpan]:
        """Find all term spans in text using Longest Match First algorithm."""
        spans: list[HighlightSpan] = []
        text_lower = text.lower()

        # Combine all terms and sort by length (longest first)
        all_terms: list[tuple[str, bool]] = [
            (term, True) for term in self.confirmed_terms
        ] + [(term, False) for term in self.candidate_terms]

        # Sort by length descending, then alphabetically
        all_terms.sort(key=lambda x: (-len(x[0]), x[0]))

        # Track which characters are already covered
        covered = [False] * len(text)

        for term, is_confirmed in all_terms:
            if not term:
                continue

            # Find all occurrences of this term
            start = 0
            while True:
                pos = text_lower.find(term, start)
                if pos == -1:
                    break

                end = pos + len(term)

                # Check if this span overlaps with already covered areas
                # We allow overlaps if the new term is longer (longest match wins)
                overlaps = any(covered[i] for i in range(pos, min(end, len(covered))))

                if not overlaps:
                    spans.append(HighlightSpan(pos, end, text[pos:end], is_confirmed))
                    # Mark as covered
                    for i in range(pos, min(end, len(covered))):
                        covered[i] = True

                start = pos + 1

        # Sort spans by position
        spans.sort(key=lambda x: x.start)

        return spans


def highlight_terms(
    text: str, confirmed_terms: set[str], candidate_terms: set[str]
) -> list[HighlightSpan]:
    """Convenience function to highlight terms in text."""
    highlighter = LongestMatchHighlighter()
    for term in confirmed_terms:
        highlighter.add_confirmed_term(term)
    for term in candidate_terms:
        highlighter.add_candidate_term(term)
    return highlighter.find_spans(text)
