"""In-memory session cache for exploration deduplication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from theow._core._logging import get_logger
from theow._core._models import Rule

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    text: str
    rule: Rule


class SessionCache:
    """In-memory cache to deduplicate similar explorations within a session."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold
        self._entries: list[CacheEntry] = []

    def check(self, context: dict[str, Any]) -> Rule | None:
        """Return cached rule if similar context was explored this session."""
        if not self._entries:
            return None

        query_text = self._context_to_text(context)

        for entry in self._entries:
            similarity = SequenceMatcher(None, query_text, entry.text).ratio()
            if similarity >= self._threshold:
                logger.debug("Session cache hit", similarity=f"{similarity:.3f}")
                return entry.rule

        return None

    def store(self, context: dict[str, Any], rule: Rule) -> None:
        """Cache exploration result."""
        text = self._context_to_text(context)
        self._entries.append(CacheEntry(text=text, rule=rule))
        logger.debug("Cached exploration result", rule=rule.name)

    def _context_to_text(self, context: dict[str, Any]) -> str:
        """Convert context to text for comparison."""
        return json.dumps(context, sort_keys=True, default=str)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()

    def invalidate(self, rule_name: str) -> None:
        """Remove a cached entry by rule name."""
        self._entries = [e for e in self._entries if e.rule.name != rule_name]
        logger.debug("Invalidated cache entry", rule=rule_name)

    @property
    def size(self) -> int:
        return len(self._entries)
