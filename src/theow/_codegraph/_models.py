"""Data models for the code graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Node:
    """A symbol in the code graph (module, class, or function)."""

    id: str
    kind: str  # "module" | "class" | "function"
    name: str
    file: str
    line: int
    end_line: int = 0
    signature: str = ""
    docstring: str = ""
    parent: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "file": self.file,
            "line": self.line,
        }
        if self.end_line:
            d["end_line"] = self.end_line
        if self.signature:
            d["signature"] = self.signature
        if self.docstring:
            d["docstring"] = self.docstring
        if self.parent:
            d["parent"] = self.parent
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Create Node from dictionary."""
        return cls(
            id=data["id"],
            kind=data["kind"],
            name=data["name"],
            file=data["file"],
            line=data["line"],
            end_line=data.get("end_line", 0),
            signature=data.get("signature", ""),
            docstring=data.get("docstring", ""),
            parent=data.get("parent", ""),
        )


@dataclass(frozen=True)
class Edge:
    """A relationship between two nodes in the code graph."""

    source: str
    target: str
    kind: str  # "defines" | "contains" | "calls" | "imports" | "inherits"
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "kind": self.kind,
        }
        if self.line:
            d["line"] = self.line
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Edge:
        """Create Edge from dictionary."""
        return cls(
            source=data["source"],
            target=data["target"],
            kind=data["kind"],
            line=data.get("line", 0),
        )


@dataclass(frozen=True)
class SearchResult:
    """A search result wrapping a Node with context."""

    node: Node
    context: str = ""
    relevance: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d: dict[str, Any] = self.node.to_dict()
        if self.context:
            d["context"] = self.context
        if self.relevance:
            d["relevance"] = self.relevance
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        """Create SearchResult from dictionary."""
        return cls(
            node=Node.from_dict(data),
            context=data.get("context", ""),
            relevance=data.get("relevance", ""),
        )
