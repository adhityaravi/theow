"""Core data models for Theow."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from theow._core._decorators import ActionRegistry


@dataclass
class Fact:
    """A condition in a rule's when: clause.

    Multiple facts in when: are ANDed - all must match.
    """

    fact: str
    equals: str | None = None
    contains: str | None = None
    regex: str | None = None
    examples: list[str] = field(default_factory=list)

    def matches(self, value: str | None) -> dict[str, str] | None:
        """Check if value matches this fact.

        Returns:
            Dict of captured groups (empty dict if no captures) on match,
            None if no match.
        """
        if value is None:
            return None

        if self.equals is not None:
            return {} if value == self.equals else None

        if self.contains is not None:
            return {} if self.contains in value else None

        if self.regex is not None:
            match = re.search(self.regex, value, re.MULTILINE | re.DOTALL)
            if match:
                return match.groupdict()
            return None

        return {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        d: dict[str, Any] = {"fact": self.fact}
        if self.equals is not None:
            d["equals"] = self.equals
        if self.contains is not None:
            d["contains"] = self.contains
        if self.regex is not None:
            d["regex"] = self.regex
        if self.examples:
            d["examples"] = self.examples
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fact:
        """Create Fact from dictionary."""
        # Normalize examples - YAML may parse multi-line strings with colons as dicts
        raw_examples = data.get("examples", [])
        examples = []
        for ex in raw_examples:
            if isinstance(ex, dict):
                # Convert dict back to multi-line string: "key: value\nkey2: value2"
                examples.append("\n".join(f"{k}: {v}" for k, v in ex.items()))
            else:
                examples.append(str(ex))

        return cls(
            fact=data["fact"],
            equals=data.get("equals"),
            contains=data.get("contains"),
            regex=data.get("regex"),
            examples=examples,
        )


@dataclass
class Action:
    """A step in a rule's then: clause.

    Actions execute sequentially, top to bottom.
    """

    action: str
    params: dict[str, Any] = field(default_factory=dict)

    def resolve_params(self, captures: dict[str, str], context: dict[str, Any]) -> dict[str, Any]:
        """Resolve param placeholders from regex captures and context.

        Placeholders use {name} syntax. Captures take precedence over context.
        """
        resolved = {}
        combined = {**context, **captures}

        for key, value in self.params.items():
            if isinstance(value, str) and "{" in value:
                # Replace all {placeholder} with values
                result = value
                for placeholder, replacement in combined.items():
                    result = result.replace(f"{{{placeholder}}}", str(replacement))
                resolved[key] = result
            else:
                resolved[key] = value

        return resolved

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        d: dict[str, Any] = {"action": self.action}
        if self.params:
            d["params"] = self.params
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        """Create Action from dictionary."""
        return cls(
            action=data["action"],
            params=data.get("params", {}),
        )


@dataclass
class LLMConfig:
    """Configuration for probabilistic rules that use LLM."""

    prompt_template: str
    tools: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    use_secondary: bool = False

    def get_prompt(
        self,
        base_path: Path | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Get the prompt content, loading from file if needed.

        Args:
            base_path: Base path for resolving file:// references.
            context: Dict of values to replace {placeholders} in the prompt.
        """
        if self.prompt_template.startswith("file://"):
            file_path = self.prompt_template[7:]
            if base_path:
                file_path = str(base_path / file_path)
            content = Path(file_path).read_text()
        else:
            content = self.prompt_template

        # Replace {placeholders} with context values
        if context:
            for key, value in context.items():
                content = content.replace(f"{{{key}}}", str(value))

        return content

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        d: dict[str, Any] = {"prompt_template": self.prompt_template}
        if self.tools:
            d["tools"] = self.tools
        if self.constraints:
            d["constraints"] = self.constraints
        if self.use_secondary:
            d["use_secondary"] = self.use_secondary
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMConfig:
        """Create LLMConfig from dictionary."""
        return cls(
            prompt_template=data["prompt_template"],
            tools=data.get("tools", []),
            constraints=data.get("constraints", {}),
            use_secondary=data.get("use_secondary", False),
        )


@dataclass
class Rule:
    """A production rule: when facts match, then act.

    Rules are the core knowledge artifact in Theow.
    """

    name: str
    description: str
    when: list[Fact]
    then: list[Action] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    collection: str = "default"
    llm_config: LLMConfig | None = None
    notes: str | None = None

    # Runtime state (not serialized)
    _resolved_params: dict[str, Any] = field(default_factory=dict, repr=False)
    _action_registry: ActionRegistry | None = field(default=None, repr=False)
    _source_path: Path | None = field(default=None, repr=False)
    _created_files: list[Path] = field(default_factory=list, repr=False)

    @property
    def is_ephemeral(self) -> bool:
        """Ephemeral rules are unproven, pending validation.

        A rule is ephemeral if it lives in the ephemeral/ subfolder.
        """
        if self._source_path:
            return "ephemeral" in self._source_path.parts
        return False

    @property
    def type(self) -> Literal["deterministic", "probabilistic"]:
        """Rule type based on whether it uses LLM."""
        return "probabilistic" if self.llm_config else "deterministic"

    def matches(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Check if all when: facts match the context.

        Returns:
            Combined captures from all facts on match, None otherwise.
        """
        all_captures: dict[str, Any] = {}

        for fact in self.when:
            value = context.get(fact.fact)
            captures = fact.matches(value)
            if captures is None:
                return None
            all_captures.update(captures)

        return all_captures

    def bind(
        self,
        captures: dict[str, str],
        context: dict[str, Any],
        action_registry: ActionRegistry | None = None,
    ) -> Rule:
        """Create a copy with resolved params and bound action registry."""
        bound = Rule(
            name=self.name,
            description=self.description,
            when=self.when,
            then=self.then,
            tags=list(self.tags),  # Copy to avoid mutation
            collection=self.collection,
            llm_config=self.llm_config,
            notes=self.notes,
        )
        bound._resolved_params = {**context, **captures}
        bound._action_registry = action_registry
        bound._source_path = self._source_path
        bound._created_files = list(self._created_files)
        return bound

    def act(self) -> list[Any]:
        """Execute the rule's actions with resolved params.

        Returns:
            List of results from each action.
        """
        if not self._action_registry:
            raise RuntimeError("Rule not bound to action registry. Call bind() first.")

        results = []
        for action in self.then:
            resolved = action.resolve_params(self._resolved_params, self._resolved_params)
            result = self._action_registry.call(action.action, resolved)
            results.append(result)

        return results

    def content_hash(self) -> str:
        """Compute hash of rule content for change detection."""
        content = self.to_yaml()
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_yaml(self, path: str | Path | None = None) -> str:
        """Serialize rule to YAML. Optionally write to path."""
        data: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "when": [f.to_dict() for f in self.when],
        }

        if self.then:
            data["then"] = [a.to_dict() for a in self.then]
        if self.tags:
            data["tags"] = self.tags
        if self.collection != "default":
            data["collection"] = self.collection
        if self.llm_config:
            data["llm_config"] = self.llm_config.to_dict()
        if self.notes:
            data["notes"] = self.notes

        yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

        if path:
            Path(path).write_text(yaml_str)

        return yaml_str

    @classmethod
    def from_yaml(cls, path: str | Path) -> Rule:
        """Load rule from a YAML file."""
        path = Path(path)
        content = path.read_text()
        rule = cls.from_yaml_string(content)
        rule._source_path = path
        return rule

    @classmethod
    def from_yaml_string(cls, content: str) -> Rule:
        """Load rule from a YAML string."""
        data = yaml.safe_load(content)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rule:
        """Create Rule from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            when=[Fact.from_dict(f) for f in data["when"]],
            then=[Action.from_dict(a) for a in data.get("then", [])],
            tags=data.get("tags", []),
            collection=data.get("collection", "default"),
            llm_config=LLMConfig.from_dict(data["llm_config"]) if data.get("llm_config") else None,
            notes=data.get("notes"),
        )

    def get_embedding_text(self) -> str:
        """Get text for vector embedding.

        Combines description with fact examples for better semantic matching.
        """
        parts = [self.description]

        for fact in self.when:
            if fact.examples:
                parts.append(" ||| ".join(fact.examples))

        return " ||| ".join(parts)

    def get_metadata(self) -> dict[str, Any]:
        """Get metadata for Chroma storage.

        Includes equals facts for filtering, plus tracking fields.
        """
        metadata: dict[str, Any] = {
            "type": self.type,
            "content_hash": self.content_hash(),
            "success_count": 0,
            "fail_count": 0,
            "explored": False,
            "cost": 0.0,
        }

        # Add equals facts as filterable metadata
        for fact in self.when:
            if fact.equals is not None:
                metadata[fact.fact] = fact.equals

        return metadata
