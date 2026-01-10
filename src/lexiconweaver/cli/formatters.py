"""CLI output formatting utilities."""

from rich.console import Console
from rich.table import Table

from lexiconweaver.engines.scout import CandidateTerm

console = Console()


def format_candidates_table(candidates: list[CandidateTerm]) -> Table:
    """Format candidate terms as a Rich table."""
    table = Table(title=f"Candidate Terms (Found: {len(candidates)})")
    table.add_column("Term", style="cyan")
    table.add_column("Confidence", style="green")
    table.add_column("Frequency", style="yellow")
    table.add_column("Pattern", style="blue")

    for candidate in candidates:
        table.add_row(
            candidate.term,
            f"{candidate.confidence:.2f}",
            str(candidate.frequency),
            candidate.context_pattern or "-",
        )

    return table
