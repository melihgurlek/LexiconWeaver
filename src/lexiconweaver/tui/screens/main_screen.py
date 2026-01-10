"""Main screen for the LexiconWeaver TUI."""

from pathlib import Path
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from lexiconweaver.config import Config
from lexiconweaver.database.models import GlossaryTerm, Project
from lexiconweaver.engines.scout import CandidateTerm, Scout
from lexiconweaver.tui.widgets.candidate_list import CandidateList
from lexiconweaver.tui.widgets.term_modal import TermModal
from lexiconweaver.tui.widgets.text_panel import TextPanel
from lexiconweaver.utils.highlighting import highlight_terms


class MainScreen(Screen):
    """Main screen displaying text panel and candidate list."""

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run_scout", "Run Scout"),
        ("t", "translate", "Translate"),
    ]

    DEFAULT_CSS = """
    MainScreen {
        background: $surface;
    }

    Horizontal {
        height: 1fr;
    }

    Vertical {
        width: 1fr;
    }

    #text_container {
        width: 2fr;
        border: solid $primary;
    }

    #candidate_container {
        width: 1fr;
        border: solid $primary;
    }

    #status_bar {
        height: 1;
        dock: bottom;
        background: $panel;
    }
    """

    def __init__(
        self,
        config: Config,
        project: Project,
        text: str = "",
        text_file: Path | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the main screen."""
        super().__init__(*args, **kwargs)
        self.config = config
        self.project = project
        self._text = text
        self._text_file = text_file
        self._scout: Scout | None = None
        self._candidates: list[CandidateTerm] = []
        self._confirmed_terms: set[str] = set()
        self._candidate_terms: set[str] = set()

    def compose(self):
        """Compose the screen widgets."""
        yield Header(show_clock=True)
        with Horizontal():
            with Vertical(id="text_container"):
                yield Static("Chapter Text", classes="section_title")
                yield TextPanel(self._text, id="text_panel")
            with Vertical(id="candidate_container"):
                yield Static("Candidate Terms", classes="section_title")
                yield CandidateList(id="candidate_list")
        yield Static("Ready", id="status_bar")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the screen is mounted."""
        self._load_text()
        self._load_confirmed_terms()
        self._update_highlights()
        self._initialize_scout()

    def _load_text(self) -> None:
        """Load text from file or use provided text."""
        if self._text_file and self._text_file.exists():
            try:
                with open(self._text_file, "r", encoding="utf-8") as f:
                    self._text = f.read()
            except Exception as e:
                self._update_status(f"Error loading file: {e}")
                return

        text_panel = self.query_one("#text_panel", TextPanel)
        text_panel.set_text(self._text)

    def _load_confirmed_terms(self) -> None:
        """Load confirmed terms from database."""
        try:
            terms = GlossaryTerm.select().where(
                GlossaryTerm.project == self.project
            )
            self._confirmed_terms = {term.source_term for term in terms}
        except Exception as e:
            self._update_status(f"Error loading terms: {e}")

    def _update_highlights(self) -> None:
        """Update text highlights based on confirmed and candidate terms."""
        text_panel = self.query_one("#text_panel", TextPanel)
        highlights = highlight_terms(
            self._text, self._confirmed_terms, self._candidate_terms
        )
        text_panel.set_highlights(highlights)

    def _initialize_scout(self) -> None:
        """Initialize the Scout engine."""
        self._scout = Scout(self.config, self.project)

    def action_run_scout(self) -> None:
        """Run the Scout to find candidate terms."""
        if self._scout is None:
            self._update_status("Scout not initialized")
            return

        self._update_status("Running Scout...")
        try:
            self._candidates = self._scout.process(self._text)
            self._candidate_terms = {c.term for c in self._candidates}

            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.set_candidates(self._candidates)

            self._update_highlights()
            self._update_status(f"Found {len(self._candidates)} candidate terms")
        except Exception as e:
            self._update_status(f"Scout error: {e}")

    def action_translate(self) -> None:
        """Start translation process."""
        self._update_status("Translation not yet implemented in TUI")

    def _update_status(self, message: str) -> None:
        """Update the status bar."""
        status_bar = self.query_one("#status_bar", Static)
        status_bar.update(message)

    def on_candidate_list_selected(self, message: CandidateList.Selected) -> None:
        """Handle candidate selection."""
        candidate = message.candidate
        self._show_term_modal(candidate.term)

    def on_candidate_list_ignored(self, message: CandidateList.Ignored) -> None:
        """Handle candidate ignore."""
        candidate = message.candidate
        # Add to ignored terms in database
        try:
            from lexiconweaver.database.models import IgnoredTerm

            IgnoredTerm.get_or_create(
                project=self.project, term=candidate.term
            )
            # Remove from candidate list
            self._candidates = [c for c in self._candidates if c.term != candidate.term]
            candidate_list = self.query_one("#candidate_list", CandidateList)
            candidate_list.set_candidates(self._candidates)
            self._update_status(f"Ignored term: {candidate.term}")
        except Exception as e:
            self._update_status(f"Error ignoring term: {e}")

    def on_candidate_list_skipped(self, message: CandidateList.Skipped) -> None:
        """Handle candidate skip."""
        candidate = message.candidate
        self._update_status(f"Skipped: {candidate.term}")

    def _show_term_modal(self, source_term: str) -> None:
        """Show the term editing modal."""
        modal = TermModal(source_term=source_term)
        self.mount(modal)

    def on_term_modal_term_saved(self, message: TermModal.TermSaved) -> None:
        """Handle term save from modal."""
        try:
            GlossaryTerm.get_or_create(
                project=self.project,
                source_term=message.source_term,
                defaults={
                    "target_term": message.target_term,
                    "category": message.category,
                    "is_regex": message.is_regex,
                },
            )
            self._confirmed_terms.add(message.source_term)
            self._candidate_terms.discard(message.source_term)
            self._update_highlights()
            self._update_status(f"Saved term: {message.source_term} -> {message.target_term}")
        except Exception as e:
            self._update_status(f"Error saving term: {e}")

    def on_term_modal_cancelled(self, message: TermModal.Cancelled) -> None:
        """Handle modal cancellation."""
        pass
