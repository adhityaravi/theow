"""In-memory session cache for exploration deduplication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from theow._core._logging import get_logger
from theow._core._models import Rule

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    embedding: list[float]
    rule: Rule


class SessionCache:
    """In-memory cache to deduplicate similar explorations within a session."""

    def __init__(self, similarity_threshold: float = 0.85) -> None:
        self._threshold = similarity_threshold
        self._entries: list[CacheEntry] = []
        self._embedder: _SimpleEmbedder | None = None

    def check(self, context: dict[str, Any]) -> Rule | None:
        """Return cached rule if similar context was explored this session."""
        if not self._entries:
            return None

        query_embedding = self._embed(context)

        for entry in self._entries:
            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity >= self._threshold:
                logger.debug("Session cache hit", similarity=f"{similarity:.3f}")
                return entry.rule

        return None

    def store(self, context: dict[str, Any], rule: Rule) -> None:
        """Cache exploration result."""
        embedding = self._embed(context)
        self._entries.append(CacheEntry(embedding=embedding, rule=rule))
        logger.debug("Cached exploration result", rule=rule.name)

    def _embed(self, context: dict[str, Any]) -> list[float]:
        """Simple embedding via hashing (no external model dependency)."""
        if self._embedder is None:
            self._embedder = _SimpleEmbedder()
        return self._embedder.embed(self._context_to_text(context))

    def _context_to_text(self, context: dict[str, Any]) -> str:
        """Convert context to text for embedding."""
        return json.dumps(context, sort_keys=True, default=str)

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)


class _SimpleEmbedder:
    """Simple character n-gram based embedder for session deduplication.

    Not as good as a real embedding model, but avoids loading heavy
    models just for session deduplication. Uses 3-gram character hashing.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        text = text.lower()

        for i in range(len(text) - 2):
            ngram = text[i : i + 3]
            idx = hash(ngram) % self._dim
            vec[idx] += 1.0

        # Normalize
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec
