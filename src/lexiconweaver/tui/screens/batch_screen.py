"""Batch translation screen for the LexiconWeaver TUI.

This screen provides a user interface for batch translation workflow:
- Folder selection
- Global scout progress
- Parallel translation progress
- Merge progress

Note: For MVP, batch translation is fully functional via CLI.
This TUI screen is a stub for future enhancement.
Users can use: lexiconweaver batch-translate <workspace> --project <name>
"""

import asyncio
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, Static

from lexiconweaver.config import Config
from lexiconweaver.database.models import Project
from lexiconweaver.engines.batch_manager import BatchManager, BatchProgress
from lexiconweaver.logging_config import get_logger

logger = get_logger(__name__)


class BatchScreen(Screen):
    """
    Batch translation screen with folder selection and progress tracking.
    
    This is a basic implementation showing the structure.
    For full batch translation functionality, use the CLI:
        lexiconweaver batch-translate <workspace> --project <name>
    """
    
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("escape", "back", "Back"),
    ]
    
    DEFAULT_CSS = """
    BatchScreen {
        background: $surface;
    }
    
    #batch-container {
        padding: 1 2;
        height: 100%;
    }
    
    #info-panel {
        background: $panel;
        border: solid $primary;
        padding: 1 2;
        margin: 1 0;
    }
    
    #progress-panel {
        background: $panel;
        border: solid $accent;
        padding: 1 2;
        margin: 1 0;
        height: 1fr;
    }
    
    #button-container {
        height: auto;
        margin: 1 0;
    }
    
    Button {
        margin: 0 1;
    }
    """
    
    def __init__(
        self,
        config: Config,
        project: Project,
        workspace_path: Path | None = None,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ):
        """
        Initialize batch screen.
        
        Args:
            config: Application configuration
            project: Project context
            workspace_path: Optional workspace path
            name: Screen name
            id: Screen ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.config = config
        self.project = project
        self.workspace_path = workspace_path or Path.cwd()
        self.batch_manager: BatchManager | None = None
        self.is_processing = False
    
    def compose(self) -> ComposeResult:
        """Compose the batch screen layout."""
        yield Header()
        
        with ScrollableContainer(id="batch-container"):
            yield Static(
                "[bold cyan]Batch Translation Workflow[/bold cyan]\n\n"
                "Process multiple chapters with parallel translation, global scouting, and automatic merging.",
                id="title"
            )
            
            with Vertical(id="info-panel"):
                yield Label(f"Workspace: {self.workspace_path}", id="workspace-label")
                yield Label(f"Project: {self.project.title}", id="project-label")
                yield Label(f"Provider: {self.config.provider.primary}", id="provider-label")
                yield Label("Status: Ready", id="status-label")
            
            with Vertical(id="progress-panel"):
                yield Static("[dim]No active translation[/dim]", id="progress-text")
            
            with Horizontal(id="button-container"):
                yield Button("Select Workspace", variant="primary", id="select-workspace")
                yield Button("Start Translation", variant="success", id="start-translation")
                yield Button("Back", variant="default", id="back-button")
        
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "back-button":
            self.app.pop_screen()
        elif event.button.id == "select-workspace":
            self._select_workspace()
        elif event.button.id == "start-translation":
            if not self.is_processing:
                self._start_batch_translation()
    
    def _select_workspace(self) -> None:
        """Select workspace folder (stub)."""
        # TODO: Implement folder selection dialog
        # For now, show message directing to CLI
        progress_widget = self.query_one("#progress-text", Static)
        progress_widget.update(
            "[yellow]Folder selection in TUI is not yet implemented.[/yellow]\n\n"
            "[bold]Use CLI for batch translation:[/bold]\n"
            "  lexiconweaver batch-translate <workspace> --project <name>\n\n"
            "[bold]Example:[/bold]\n"
            "  lexiconweaver batch-translate ./my-novel --project 'MyNovel' --format epub"
        )
    
    def _start_batch_translation(self) -> None:
        """Start batch translation (stub)."""
        if self.is_processing:
            return
        
        progress_widget = self.query_one("#progress-text", Static)
        progress_widget.update(
            "[yellow]Batch translation in TUI is not yet implemented.[/yellow]\n\n"
            "[bold]Use CLI for full batch translation:[/bold]\n\n"
            "[cyan]Dry run (estimate costs):[/cyan]\n"
            "  lexiconweaver batch-translate ./workspace --project 'MyNovel' --dry-run\n\n"
            "[cyan]Full translation:[/cyan]\n"
            "  lexiconweaver batch-translate ./workspace --project 'MyNovel' --format epub\n\n"
            "[cyan]Resume interrupted translation:[/cyan]\n"
            "  lexiconweaver batch-translate ./workspace --project 'MyNovel' --resume\n\n"
            "[bold]Features:[/bold]\n"
            "  • Parallel chapter translation (5x speedup)\n"
            "  • Global scout with burst detection\n"
            "  • Checkpoint/resume support\n"
            "  • Cost estimation (--dry-run)\n"
            "  • Multi-chapter EPUB with TOC\n"
            "  • Styled chapter titles"
        )
    
    def action_back(self) -> None:
        """Go back to previous screen."""
        self.app.pop_screen()


# Future enhancement: Full implementation would include:
# - DirectoryTree widget for folder selection
# - Progress bars for each stage (scout, translate, merge)
# - Real-time chapter status display
# - Term approval interface for scout results
# - Cost estimation display
# - Resume/cancel controls
# - Error handling and retry options
