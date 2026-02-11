"""Tests for resolver."""

from unittest.mock import MagicMock

from theow._core._decorators import ActionRegistry
from theow._core._models import Action, Fact, Rule
from theow._core._resolver import Resolver


def test_resolver_explicit_rule_match(theow_dir):
    action_registry = ActionRegistry()

    @action_registry.register("fix_it")
    def fix_it(x: str):
        return {"fixed": x}

    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="my_rule",
        description="Test",
        when=[Fact(fact="problem_type", equals="build")],
        then=[Action(action="fix_it", params={"x": "{problem_type}"})],
    )
    rule.to_yaml(theow_dir / "rules" / "my_rule.rule.yaml")

    result = resolver.resolve(
        context={"problem_type": "build"},
        rules=["my_rule"],
    )

    assert result is not None
    assert result.name == "my_rule"


def test_resolver_explicit_rule_no_match(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="my_rule",
        description="Test",
        when=[Fact(fact="problem_type", equals="build")],
    )
    rule.to_yaml(theow_dir / "rules" / "my_rule.rule.yaml")

    result = resolver.resolve(
        context={"problem_type": "test"},
        rules=["my_rule"],
    )

    assert result is None


def test_resolver_fallback_disabled(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    result = resolver.resolve(
        context={"problem_type": "build"},
        rules=["nonexistent"],
        fallback=False,
    )

    assert result is None
    chroma.query_rules.assert_not_called()


def test_resolver_extract_query_text():
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=MagicMock(),
    )

    context = {"short": "x", "long": "this is the longest string value"}
    query = resolver._extract_query_text(context)
    assert query == "this is the longest string value"
