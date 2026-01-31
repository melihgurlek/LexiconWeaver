"""Candidate list widget for displaying potential terms."""

from textual.binding import Binding
from textual.events import Key
from textual.message import Message
from textual.widgets import OptionList
from rich.text import Text

from lexiconweaver.engines.scout import CandidateTerm


class CandidateList(OptionList):
    """List widget for displaying candidate terms sorted by confidence."""

    BINDINGS = [
        Binding("enter", "select", "Edit/Confirm", show=True),
        Binding("delete", "ignore", "Ignore", show=True),
        Binding("s", "skip", "Skip", show=True),
    ]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the candidate list."""
        super().__init__(*args, **kwargs)
        self.candidates: list[CandidateTerm] = []
        self._all_candidates: list[CandidateTerm] = []
        self._terms_with_translations: set[str] = set()
        self._pending_keypress = False
        self._processing_selection = False
        self._filter_query: str = ""
        self._has_proposals = False  # Whether these are LLM proposals
        self._proposals_map: dict[str, dict] = {}  # term -> proposal info

    def set_candidates(
        self,
        candidates: list[CandidateTerm],
        terms_with_translations: set[str] | None = None,
        has_proposals: bool = False,
        proposals_map: dict[str, dict] | None = None,
    ) -> None:
        """Set the candidate terms to display.
        
        Args:
            candidates: List of candidate terms to display
            terms_with_translations: Set of terms that already have translations
            has_proposals: Whether these candidates have LLM proposals
            proposals_map: Optional map of term -> proposal info for display
        """
        self._all_candidates = candidates
        self._terms_with_translations = terms_with_translations or set()
        self._has_proposals = has_proposals
        self._proposals_map = proposals_map or {}
        
        self._apply_filter()
    
    def _apply_filter(self) -> None:
        """Apply the current filter query to candidates."""
        if not self._filter_query:
            filtered_candidates = self._all_candidates
        else:
            query_lower = self._filter_query.lower()
            filtered_candidates = [
                c for c in self._all_candidates
                if query_lower in c.term.lower()
            ]
        
        self.candidates = filtered_candidates
        self.clear_options()

        for candidate in filtered_candidates:
            label = self._format_candidate_label(candidate)
            self.add_option(label)
    
    def _format_candidate_label(self, candidate: CandidateTerm) -> Text | str:
        """Format a candidate's display label with appropriate styling."""
        term = candidate.term
        has_translation = term in self._terms_with_translations
        is_llm_refined = candidate.context_pattern == "llm_refined"
        
        label = Text()
        
        if has_translation:
            # Green for terms with translations
            label.append(term, style="bold green")
        elif is_llm_refined or self._has_proposals:
            # Yellow for LLM proposals pending review
            label.append(term, style="bold yellow")
        else:
            # Default for regular candidates
            label.append(term, style="bold")
        
        # Add confidence and frequency info
        if is_llm_refined:
            label.append(" [LLM]", style="dim cyan")
        else:
            label.append(f" (conf: {candidate.confidence:.2f}, freq: {candidate.frequency})", style="dim")
        
        return label
    
    def filter_candidates(self, query: str) -> None:
        """Filter candidates by search query.
        
        Args:
            query: Search query string (case-insensitive)
        """
        self._filter_query = query
        self._apply_filter() 

    def get_selected_candidate(self) -> CandidateTerm | None:
        """Get the currently selected candidate."""
        if self.highlighted is None:
            return None

        if 0 <= self.highlighted < len(self.candidates):
            return self.candidates[self.highlighted]

        return None

    def on_key(self, event: Key) -> None:
        """Track Enter key press."""
        if event.key == "enter":
            self._pending_keypress = True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle option selection (click or keyboard)."""
        event.stop()
        
        candidate = self.get_selected_candidate()
        if not candidate:
            return
            
        if self._pending_keypress and not self._processing_selection:
            self.action_select()
        else:
            if not self._processing_selection:
                self.post_message(self.Highlighted(candidate))

    def action_select(self) -> None:
        """Handle Enter key press - allow selection."""
        if not self._pending_keypress:
            return
        
        if self._processing_selection:
            return
        
        self._processing_selection = True
        try:
            candidate = self.get_selected_candidate()
            if candidate:
                self.post_message(self.Selected(candidate))
        finally:
            self._processing_selection = False
            self._pending_keypress = False

    def action_ignore(self) -> None:
        candidate = self.get_selected_candidate()
        if candidate:
            self.post_message(self.Ignored(candidate))

    def action_skip(self) -> None:
        candidate = self.get_selected_candidate()
        if candidate:
            self.post_message(self.Skipped(candidate))

    class Selected(Message):
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate

    class Ignored(Message):
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate

    class Skipped(Message):
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate

    class Highlighted(Message):
        """Message sent when a candidate is clicked (not Enter key)."""
        def __init__(self, candidate: CandidateTerm) -> None:
            super().__init__()
            self.candidate = candidate
