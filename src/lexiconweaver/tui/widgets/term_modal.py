"""Modal dialog for editing term definitions."""

from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Label, Select, Static

TERM_CATEGORIES = [
    ("Person", "Person"),
    ("Location", "Location"),
    ("Skill", "Skill"),
    ("Clan", "Clan"),
    ("Item", "Item"),
    ("Other", "Other"),
]


class TermModal(Container):
    """Modal dialog for adding/editing a term with optional LLM proposal display."""

    DEFAULT_CSS = """
    TermModal {
        width: 90;
        height: auto;
        max-height: 32;
        border: thick $primary;
        background: $surface;
    }

    TermModal Vertical {
        padding: 1;
    }

    TermModal Horizontal {
        height: 3;
        align: center middle;
    }

    TermModal .action-buttons {
        height: 5;
        align: center middle;
    }

    TermModal Input {
        width: 1fr;
        margin: 1 0;
    }

    TermModal Select {
        width: 1fr;
        margin: 1 0;
    }

    TermModal Button {
        margin: 0 1;
        width: 14;
    }

    TermModal .proposal-section {
        background: $boost;
        padding: 1;
        margin: 1 0;
        border: dashed $secondary;
    }

    TermModal .proposal-label {
        color: $secondary;
        text-style: bold;
    }

    TermModal .reasoning-text {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }

    TermModal .section-label {
        margin-top: 1;
        text-style: bold;
    }
    """

    def __init__(
        self,
        source_term: str,
        target_term: str = "",
        category: str = "",
        is_regex: bool = False,
        # LLM proposal fields (optional)
        proposed_translation: str | None = None,
        proposed_category: str | None = None,
        llm_reasoning: str | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize the term modal.
        
        Args:
            source_term: The source term (read-only).
            target_term: Current target term (if editing existing).
            category: Current category (if editing existing).
            is_regex: Whether this is a regex term.
            proposed_translation: LLM-proposed translation (if any).
            proposed_category: LLM-proposed category (if any).
            llm_reasoning: LLM's reasoning for the proposal (if any).
        """
        super().__init__(*args, **kwargs)
        self.source_term = source_term
        self._initial_target_term = target_term
        self._initial_category = category
        self._target_term_input: Input | None = None
        self._category_select: Select | None = None
        self._is_regex = is_regex
        
        # LLM proposal fields
        self._proposed_translation = proposed_translation
        self._proposed_category = proposed_category
        self._llm_reasoning = llm_reasoning
        self._has_proposal = proposed_translation is not None

    def compose(self):
        """Compose the modal widgets."""
        with Vertical():
            yield Label(f"Source Term: {self.source_term}", id="source_label")
            
            # Show LLM proposal section if available
            if self._has_proposal:
                with Container(classes="proposal-section"):
                    yield Label("LLM Proposal", classes="proposal-label")
                    yield Static(
                        f"Translation: {self._proposed_translation or 'N/A'}",
                        id="proposed_translation",
                    )
                    yield Static(
                        f"Category: {self._proposed_category or 'N/A'}",
                        id="proposed_category",
                    )
                    if self._llm_reasoning:
                        yield Label("Reasoning:", classes="section-label")
                        yield Static(
                            self._llm_reasoning[:200] + "..." if len(self._llm_reasoning or "") > 200 else self._llm_reasoning,
                            classes="reasoning-text",
                        )

            yield Label("Target Term:", classes="section-label")
            # Pre-fill with proposal if available, otherwise use existing value
            initial_value = self._proposed_translation if self._has_proposal and not self._initial_target_term else self._initial_target_term
            yield Input(
                value=initial_value or "",
                placeholder="Enter translation...",
                id="target_input",
            )
            
            yield Label("Category:", classes="section-label")
            yield Select(
                options=TERM_CATEGORIES,
                allow_blank=True,
                id="category_select",
            )
            
            yield Label("Scope: Global (all chapters)", id="scope_label")

            # Different button layout based on whether there's a proposal
            if self._has_proposal:
                with Horizontal(classes="action-buttons"):
                    yield Button("Approve", variant="success", id="approve_button")
                    yield Button("Modify", variant="primary", id="modify_button")
                    yield Button("Reject", variant="error", id="reject_button")
                    yield Button("Cancel", variant="default", id="cancel_button")
            else:
                with Horizontal(classes="action-buttons"):
                    yield Button("Save", variant="primary", id="save_button")
                    yield Button("Cancel", variant="default", id="cancel_button")

    def on_mount(self) -> None:
        """Called when the modal is mounted."""
        self._target_term_input = self.query_one("#target_input", Input)
        self._category_select = self.query_one("#category_select", Select)
        
        # Set the initial category value
        initial_category = self._proposed_category if self._has_proposal and not self._initial_category else self._initial_category
        if initial_category and self._category_select:
            for option_value, option_label in TERM_CATEGORIES:
                if option_label == initial_category:
                    self._category_select.value = option_value
                    break
        
        self._target_term_input.focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "save_button":
            self._save_term(action="save")
        elif button_id == "approve_button":
            self._save_term(action="approve")
        elif button_id == "modify_button":
            self._save_term(action="modify")
        elif button_id == "reject_button":
            self._reject_term()
        elif button_id == "cancel_button":
            self._cancel()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission (Enter key)."""
        if event.input.id == "target_input":
            if self._has_proposal:
                self._save_term(action="modify")
            else:
                self._save_term(action="save")

    def _get_current_values(self) -> tuple[str, str]:
        """Get current target term and category from inputs."""
        target_term = self._target_term_input.value if self._target_term_input else ""
        category = ""
        if self._category_select and self._category_select.value:
            select_value = self._category_select.value
            if isinstance(select_value, tuple):
                category = select_value[0]
            elif isinstance(select_value, str):
                category = select_value
        return target_term, category

    def _save_term(self, action: str = "save") -> None:
        """Save the term and close modal.
        
        Args:
            action: The action type - 'save', 'approve', or 'modify'.
        """
        target_term, category = self._get_current_values()
        
        if action == "approve":
            if not target_term and self._proposed_translation:
                target_term = self._proposed_translation
            if not category and self._proposed_category:
                category = self._proposed_category

        self.post_message(
            self.TermSaved(
                source_term=self.source_term,
                target_term=target_term,
                category=category,
                is_regex=self._is_regex,
                action=action,
                was_proposal=self._has_proposal,
            )
        )
        self.remove()

    def _reject_term(self) -> None:
        """Reject the proposed term."""
        self.post_message(
            self.TermRejected(
                source_term=self.source_term,
            )
        )
        self.remove()

    def _cancel(self) -> None:
        """Cancel and close modal."""
        self.post_message(self.Cancelled())
        self.remove()

    class TermSaved(Message):
        """Message sent when term is saved."""

        def __init__(
            self,
            source_term: str,
            target_term: str,
            category: str,
            is_regex: bool,
            action: str = "save",
            was_proposal: bool = False,
        ) -> None:
            super().__init__()
            self.source_term = source_term
            self.target_term = target_term
            self.category = category
            self.is_regex = is_regex
            self.action = action
            self.was_proposal = was_proposal

    class TermRejected(Message):
        """Message sent when a proposed term is rejected."""

        def __init__(self, source_term: str) -> None:
            super().__init__()
            self.source_term = source_term

    class Cancelled(Message):
        """Message sent when modal is cancelled."""

        def __init__(self) -> None:
            super().__init__()
