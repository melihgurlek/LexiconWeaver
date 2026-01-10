"""Candidate list widget for displaying potential terms."""

from textual.binding import Binding
from textual.message import Message
from textual.widgets import OptionList

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

    def set_candidates(self, candidates: list[CandidateTerm]) -> None:
        """Set the candidate terms to display."""
        self.candidates = candidates
        self.clear_options()

        for candidate in candidates:
            # Format: "term (confidence: 0.85, freq: 5)"
            label = (
                f"{candidate.term} "
                f"(conf: {candidate.confidence:.2f}, "
                f"freq: {candidate.frequency})"
            )
            self.add_option(label, id=candidate.term)

    def get_selected_candidate(self) -> CandidateTerm | None:
        """Get the currently selected candidate."""
        if self.highlighted is None:
            return None

        try:
            option = self.get_option_at_index(self.highlighted)
            if option is None:
                return None

            # Option.id should match the term
            for candidate in self.candidates:
                if candidate.term == option.id:
                    return candidate
        except Exception:
            return None

        return None

    def action_select(self) -> None:
        """Handle select action (Enter key)."""
        candidate = self.get_selected_candidate()
        if candidate:
            self.post_message(self.Selected(self, candidate))

    def action_ignore(self) -> None:
        """Handle ignore action (Delete key)."""
        candidate = self.get_selected_candidate()
        if candidate:
            self.post_message(self.Ignored(self, candidate))

    def action_skip(self) -> None:
        """Handle skip action (S key)."""
        candidate = self.get_selected_candidate()
        if candidate:
            self.post_message(self.Skipped(self, candidate))

    class Selected(Message):
        """Message sent when a candidate is selected."""

        def __init__(self, sender, candidate: CandidateTerm) -> None:
            super().__init__(sender)
            self.candidate = candidate

    class Ignored(Message):
        """Message sent when a candidate is ignored."""

        def __init__(self, sender, candidate: CandidateTerm) -> None:
            super().__init__(sender)
            self.candidate = candidate

    class Skipped(Message):
        """Message sent when a candidate is skipped."""

        def __init__(self, sender, candidate: CandidateTerm) -> None:
            super().__init__(sender)
            self.candidate = candidate
