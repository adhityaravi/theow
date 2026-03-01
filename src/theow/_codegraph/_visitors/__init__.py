"""Language visitor protocol and registry."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Protocol

from theow._codegraph._models import Edge, Node


class LanguageVisitor(Protocol):
    """Protocol for language-specific tree-sitter visitors."""

    extensions: list[str]

    def parse_file(
        self, path: Path, source: str, relative_path: str
    ) -> tuple[list[Node], list[Edge]]: ...


# Registry mapping language name -> (module_path, class_name)
BUILTIN_VISITORS: dict[str, tuple[str, str]] = {
    "python": ("theow._codegraph._visitors._python", "PythonVisitor"),
    "go": ("theow._codegraph._visitors._go", "GoVisitor"),
}


def load_visitors(languages: list[str]) -> list[LanguageVisitor]:
    """Load visitors for the specified languages.

    Args:
        languages: Language names to load (e.g. ["python", "go"]).

    Raises:
        ValueError: If a language name is not in the registry.
        ImportError: If a language's tree-sitter grammar is not installed.
    """
    visitors: list[LanguageVisitor] = []
    for lang in languages:
        entry = BUILTIN_VISITORS.get(lang)
        if entry is None:
            raise ValueError(
                f"Unknown language: {lang!r}. Available: {sorted(BUILTIN_VISITORS)}"
            )
        module_path, class_name = entry
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        visitors.append(cls())
    return visitors
