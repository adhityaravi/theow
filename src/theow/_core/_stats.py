"""Stats display (meow)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from theow._core._chroma_store import ChromaStore


def meow(chroma: ChromaStore) -> None:
    """Print Theow stats."""
    console = Console()
    rules = chroma.get_all_rules_with_stats()

    if not rules:
        console.print("[dim]🐱 No rules yet[/dim]")
        return

    total_resolves = sum(r["success_count"] for r in rules)
    total_explores = sum(1 for r in rules if r["explored"])

    console.print(f"🐱 [bold]{total_resolves}[/] resolves, [bold]{total_explores}[/] explores")

    # Build unified table sorted by total hits descending
    active_rules = [r for r in rules if r["success_count"] + r["fail_count"] > 0]
    active_rules.sort(key=lambda r: r["success_count"] + r["fail_count"], reverse=True)

    if active_rules:
        table = Table(title="🐱 rules", show_header=True, header_style="bold")
        table.add_column("Rule", style="cyan")
        table.add_column("Hit", justify="right")
        table.add_column("Fix", justify="right")
        table.add_column("Rate", justify="right")

        for r in active_rules:
            hits = r["success_count"] + r["fail_count"]
            fixes = r["success_count"]
            rate = (fixes / hits * 100) if hits else 0
            rate_int = int(rate)

            if rate >= 70:
                rate_str = f"[green]{rate_int}%[/green]"
            elif rate >= 30:
                rate_str = f"[yellow]{rate_int}%[/yellow]"
            else:
                rate_str = f"[red]{rate_int}%[/red]"

            table.add_row(r["name"], str(hits), str(fixes), rate_str)

        console.print(table)
