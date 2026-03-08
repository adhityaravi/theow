"""Rule matching and resolution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from theow._core._chroma_store import extract_query_text
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
        n_results: int = 10,
        exclude_rules: list[str] | None = None,
    ) -> Rule | None:
        """Match context against rules and return first match.

        Resolution order:
        1. Explicit rules by name (if specified)
        2. Rules matching tags (if specified)
        3. Vector search fallback (if enabled)

        Args:
            context: The failure context to match against.
            collection: The rule collection to search.
            rules: Explicit rule names to try first.
            tags: Tags to filter rules by.
            fallback: Whether to use vector search if no explicit match.
            n_results: Max number of candidates to retrieve from vector search.
            exclude_rules: Rule names to skip (already tried and failed).
        """
        logger.debug("Resolving context", collection=collection)
        exclude_set = set(exclude_rules or [])

        if rules:
            for rule_name in rules:
                if rule_name in exclude_set:
                    continue
                rule = self._try_rule(rule_name, context, collection)
                if rule:
                    return rule

        if tags:
            candidates = self._find_by_tags(collection, tags)
            for rule_name in candidates:
                if rule_name in exclude_set:
                    continue
                rule = self._try_rule(rule_name, context, collection)
                if rule:
                    return rule

        if fallback:
            return self._vector_search(context, collection, n_results, exclude_set)

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

        return rule.bind(captures, context, self._action_registry)

    def _find_by_tags(self, collection: str, tags: list[str]) -> list[str]:
        """Find rule names matching any of the given tags.

        Deterministic rules are returned before probabilistic (LLM) rules
        so they get tried first.
        """
        all_rules = self._chroma.list_rules(collection)
        deterministic: list[str] = []
        probabilistic: list[str] = []

        for rule_name in all_rules:
            rule = self._load_rule(rule_name, collection)
            if rule and any(tag in rule.tags for tag in tags):
                if rule.type == "deterministic":
                    deterministic.append(rule_name)
                else:
                    probabilistic.append(rule_name)

        return deterministic + probabilistic

    def _vector_search(
        self,
        context: dict[str, Any],
        collection: str,
        n_results: int = 10,
        exclude_rules: set[str] | None = None,
    ) -> Rule | None:
        """Find matching rule via vector similarity search."""
        metadata_filter = self._extract_metadata_filter(context)
        query_text = self._extract_query_text(context)
        exclude_set = exclude_rules or set()

        if not query_text:
            logger.debug("No query text in context")
            return None

        logger.debug("Vector search", query=query_text[:50], n_results=n_results)

        results = self._chroma.query_rules(
            collection=collection,
            query_text=query_text,
            metadata_filter=metadata_filter,
            n_results=n_results + len(exclude_set),  # Fetch extra to account for exclusions
        )

        # Sort results: deterministic rules first, probabilistic last
        def rule_priority(result: tuple[str, float, dict]) -> tuple[int, float]:
            _name, distance, metadata = result
            type_priority = 0 if metadata.get("type") == "deterministic" else 1
            return (type_priority, distance)

        results = sorted(results, key=rule_priority)

        for rule_name, distance, metadata in results:
            if rule_name in exclude_set:
                logger.debug("Skipping excluded rule", rule=rule_name)
                continue

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
        return extract_query_text(context)

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
