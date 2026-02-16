"""Tests for resolver."""

from unittest.mock import MagicMock

from theow._core._decorators import ActionRegistry
from theow._core._models import Action, Fact, LLMConfig, Rule
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


def test_resolver_tags_path(theow_dir):
    """Resolve via tags when no explicit rules given."""
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    chroma.list_rules.return_value = ["tagged_rule"]
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="tagged_rule",
        description="Test",
        when=[Fact(fact="problem_type", equals="build")],
        tags=["go", "dep"],
    )
    rule.to_yaml(theow_dir / "rules" / "tagged_rule.rule.yaml")

    result = resolver.resolve(
        context={"problem_type": "build"},
        tags=["go"],
        fallback=False,
    )

    assert result is not None
    assert result.name == "tagged_rule"


def test_resolver_tags_no_match(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    chroma.list_rules.return_value = ["tagged_rule"]
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="tagged_rule",
        description="Test",
        when=[Fact(fact="problem_type", equals="build")],
        tags=["python"],
    )
    rule.to_yaml(theow_dir / "rules" / "tagged_rule.rule.yaml")

    result = resolver.resolve(
        context={"problem_type": "build"},
        tags=["go"],
        fallback=False,
    )
    assert result is None


def test_resolver_vector_search_with_results(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    chroma.query_rules.return_value = [
        ("found_rule", 0.1, {"type": "deterministic"}),
    ]
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="found_rule",
        description="Test",
        when=[Fact(fact="stderr", contains="error happened")],
    )
    rule.to_yaml(theow_dir / "rules" / "found_rule.rule.yaml")

    result = resolver.resolve(
        context={"stderr": "some error happened here"},
    )

    assert result is not None
    assert result.name == "found_rule"


def test_resolver_vector_search_deterministic_first(theow_dir):
    """Deterministic rules should be tried before probabilistic ones."""
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    chroma.query_rules.return_value = [
        ("prob_rule", 0.05, {"type": "probabilistic"}),
        ("det_rule", 0.1, {"type": "deterministic"}),
    ]
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    det_rule = Rule(
        name="det_rule",
        description="Deterministic",
        when=[Fact(fact="stderr", contains="error")],
        then=[Action(action="fix")],
    )
    det_rule.to_yaml(theow_dir / "rules" / "det_rule.rule.yaml")

    prob_rule = Rule(
        name="prob_rule",
        description="Probabilistic",
        when=[Fact(fact="stderr", contains="error")],
        llm_config=LLMConfig(prompt_template="Fix"),
    )
    prob_rule.to_yaml(theow_dir / "rules" / "prob_rule.rule.yaml")

    result = resolver.resolve(context={"stderr": "some error here"})
    assert result is not None
    assert result.name == "det_rule"


def test_resolver_vector_search_no_query_text(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    result = resolver.resolve(context={})
    assert result is None
    chroma.query_rules.assert_not_called()


def test_resolver_exclude_rules(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="excluded",
        description="Test",
        when=[Fact(fact="problem_type", equals="build")],
    )
    rule.to_yaml(theow_dir / "rules" / "excluded.rule.yaml")

    result = resolver.resolve(
        context={"problem_type": "build"},
        rules=["excluded"],
        exclude_rules=["excluded"],
    )
    assert result is None


def test_resolver_load_rule_cache(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="cached",
        description="Test",
        when=[Fact(fact="x", equals="y")],
    )
    rule.to_yaml(theow_dir / "rules" / "cached.rule.yaml")

    # First load populates cache
    r1 = resolver._load_rule("cached", "default")
    assert r1 is not None

    # Second load from cache
    r2 = resolver._load_rule("cached", "default")
    assert r2 is r1


def test_resolver_load_rule_not_found(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    result = resolver._load_rule("nonexistent", "default")
    assert result is None


def test_resolver_load_rule_parse_error(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    (theow_dir / "rules" / "bad.rule.yaml").write_text("not: valid: [yaml")
    result = resolver._load_rule("bad", "default")
    assert result is None


def test_resolver_clear_cache(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(name="r", description="Test", when=[Fact(fact="x", equals="y")])
    rule.to_yaml(theow_dir / "rules" / "r.rule.yaml")

    resolver._load_rule("r", "default")
    assert len(resolver._rules_cache) == 1

    resolver.clear_cache()
    assert len(resolver._rules_cache) == 0


def test_resolver_extract_metadata_filter(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = {"problem_type"}
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    result = resolver._extract_metadata_filter({"problem_type": "build", "other": "val"})
    assert result == {"problem_type": "build"}


def test_resolver_extract_metadata_filter_empty(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    result = resolver._extract_metadata_filter({"x": "y"})
    assert result is None


def test_resolver_validate_and_bind(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    rule = Rule(
        name="r",
        description="Test",
        when=[Fact(fact="x", equals="y")],
    )

    bound = resolver._validate_and_bind(rule, {"x": "y"})
    assert bound is not None
    assert bound.name == "r"

    no_match = resolver._validate_and_bind(rule, {"x": "z"})
    assert no_match is None


def test_resolver_vector_search_excludes_rules(theow_dir):
    chroma = MagicMock()
    chroma.get_metadata_keys.return_value = set()
    chroma.query_rules.return_value = [
        ("excluded", 0.1, {"type": "deterministic"}),
        ("good", 0.2, {"type": "deterministic"}),
    ]
    action_registry = ActionRegistry()

    resolver = Resolver(
        chroma=chroma,
        action_registry=action_registry,
        rules_dir=theow_dir / "rules",
    )

    good_rule = Rule(
        name="good",
        description="Test",
        when=[Fact(fact="stderr", contains="error")],
    )
    good_rule.to_yaml(theow_dir / "rules" / "good.rule.yaml")

    result = resolver.resolve(
        context={"stderr": "some error"},
        exclude_rules=["excluded"],
    )
    assert result is not None
    assert result.name == "good"
