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
    total_cost = sum(r["cost"] for r in rules)

    console.print(f"🐱 [bold]{total_resolves}[/] resolves, [bold]{total_explores}[/] explores")

    top_rules = sorted(rules, key=lambda r: r["success_count"], reverse=True)[:5]
    if top_rules and any(r["success_count"] > 0 for r in top_rules):
        table = Table(title="🐱 top rules", show_header=True, header_style="bold")
        table.add_column("Rule", style="cyan")
        table.add_column("Success", style="green", justify="right")
        table.add_column("Fail", style="red", justify="right")
        for r in top_rules:
            if r["success_count"] > 0:
                table.add_row(r["name"], str(r["success_count"]), str(r["fail_count"]))
        console.print(table)

    struggling = [r for r in rules if r["fail_count"] > r["success_count"]]
    if struggling:
        table = Table(title="🐱 struggling", show_header=True, header_style="bold")
        table.add_column("Rule", style="yellow")
        table.add_column("Success", justify="right")
        table.add_column("Fail", style="red", justify="right")
        for r in struggling:
            table.add_row(r["name"], str(r["success_count"]), str(r["fail_count"]))
        console.print(table)

    console.print(f"🐱 total cost: [bold green]${total_cost:.2f}[/]")
