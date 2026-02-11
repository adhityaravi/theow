"""Rule matching and resolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from theow._core._logging import get_logger
from theow._core._models import Rule

if TYPE_CHECKING:
    from theow._core._chroma_store import ChromaStore
    from theow._core._decorators import ActionRegistry

logger = get_logger(__name__)


class Resolver:
    """Matches context against rules and returns bound rules for execution."""

    def __init__(
        self,
        chroma: ChromaStore,
        action_registry: ActionRegistry,
        rules_dir: Path,
    ) -> None:
        self._chroma = chroma
        self._action_registry = action_registry
        self._rules_dir = rules_dir
        self._rules_cache: dict[str, Rule] = {}

    def resolve(
        self,
        context: dict[str, Any],
        collection: str = "default",
        rules: list[str] | None = None,
        tags: list[str] | None = None,
        fallback: bool = True,
    ) -> Rule | None:
        """Match context against rules and return first match.

        Resolution order:
        1. Explicit rules by name (if specified)
        2. Rules matching tags (if specified)
        3. Vector search fallback (if enabled)
        """
        logger.debug("Resolving context", collection=collection)

        # Try explicit rules first
        if rules:
            for rule_name in rules:
                rule = self._try_rule(rule_name, context, collection)
                if rule:
                    return rule

        # Try tag-based matching
        if tags:
            candidates = self._find_by_tags(collection, tags)
            for rule_name in candidates:
                rule = self._try_rule(rule_name, context, collection)
                if rule:
                    return rule

        # Fall back to vector search
        if fallback:
            return self._vector_search(context, collection)

        return None

    def _try_rule(
        self,
        rule_name: str,
        context: dict[str, Any],
        collection: str,
    ) -> Rule | None:
        """Try to match a specific rule against context."""
        rule = self._load_rule(rule_name, collection)
        if not rule:
            logger.debug("Rule not found", rule=rule_name)
            return None

        return self._validate_and_bind(rule, context)

    def _validate_and_bind(
        self,
        rule: Rule,
        context: dict[str, Any],
    ) -> Rule | None:
        """Validate rule facts against context and bind if matched."""
        captures = rule.matches(context)
        if captures is None:
            logger.debug("Rule facts not matched", rule=rule.name)
            return None

        logger.debug("Rule matched", rule=rule.name)
        return rule.bind(captures, context, self._action_registry)

    def _find_by_tags(self, collection: str, tags: list[str]) -> list[str]:
        """Find rule names matching any of the given tags."""
        # Load all rules and filter by tags
        all_rules = self._chroma.list_rules(collection)
        matching = []

        for rule_name in all_rules:
            rule = self._load_rule(rule_name, collection)
            if rule and any(tag in rule.tags for tag in tags):
                matching.append(rule_name)

        return matching

    def _vector_search(
        self,
        context: dict[str, Any],
        collection: str,
    ) -> Rule | None:
        """Find matching rule via vector similarity search."""
        metadata_filter = self._extract_metadata_filter(context)
        query_text = self._extract_query_text(context)

        if not query_text:
            logger.debug("No query text in context")
            return None

        logger.debug("Vector search", query=query_text[:50])

        results = self._chroma.query_rules(
            collection=collection,
            query_text=query_text,
            metadata_filter=metadata_filter,
            n_results=10,
        )

        # Sort results: deterministic rules first, probabilistic last
        def rule_priority(result: tuple[str, float, dict]) -> tuple[int, float]:
            _name, distance, metadata = result
            type_priority = 0 if metadata.get("type") == "deterministic" else 1
            return (type_priority, distance)

        results = sorted(results, key=rule_priority)

        for rule_name, distance, metadata in results:
            logger.debug("Candidate rule", rule=rule_name, distance=f"{distance:.3f}")

            rule = self._load_rule(rule_name, collection)
            if not rule:
                continue

            bound_rule = self._validate_and_bind(rule, context)
            if bound_rule:
                return bound_rule

        return None

    def _extract_metadata_filter(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Extract filterable metadata from context."""
        known_keys = self._chroma.get_metadata_keys()
        filters = {}

        for key, value in context.items():
            if key in known_keys and isinstance(value, str):
                filters[key] = value

        return filters if filters else None

    def _extract_query_text(self, context: dict[str, Any]) -> str:
        """Extract query text from context (longest string value)."""
        longest = ""

        for value in context.values():
            if isinstance(value, str) and len(value) > len(longest):
                longest = value

        return longest

    def _load_rule(self, name: str, collection: str) -> Rule | None:
        """Load rule from cache or file."""
        cache_key = f"{collection}:{name}"

        if cache_key in self._rules_cache:
            return self._rules_cache[cache_key]

        rule_path = self._rules_dir / f"{name}.rule.yaml"
        if not rule_path.exists():
            return None

        try:
            rule = Rule.from_yaml(rule_path)
            self._rules_cache[cache_key] = rule
            return rule
        except Exception as e:
            logger.warning("Failed to load rule", rule=name, error=str(e))
            return None

    def clear_cache(self) -> None:
        """Clear the rules cache."""
        self._rules_cache.clear()
