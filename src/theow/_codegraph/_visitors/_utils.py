"""Shared tree-sitter utilities for language visitors."""

from __future__ import annotations

from tree_sitter import Node as TSNode


def text(node: TSNode) -> str:
    """Get node text as a decoded string."""
    return node.text.decode() if node.text else ""


def child_by_type(node: TSNode, type_name: str) -> TSNode | None:
    """Find first child of a given type."""
    for child in node.children:
        if child.type == type_name:
            return child
    return None


def last_child_by_type(node: TSNode, type_name: str) -> TSNode | None:
    """Find last child of a given type."""
    result = None
    for child in node.children:
        if child.type == type_name:
            result = child
    return result
