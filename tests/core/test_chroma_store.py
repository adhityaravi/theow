"""Tests for chroma store."""

import pytest

from theow._core._chroma_store import ChromaStore
from theow._core._models import Fact, Rule


def test_chroma_index_and_query_rule(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")

    rule = Rule(
        name="test_rule",
        description="Fix module rename issues",
        when=[Fact(fact="problem_type", equals="dep_resolution")],
        collection="default",
    )
    store.index_rule(rule)

    results = store.query_rules(
        collection="default",
        query_text="module rename problem",
        n_results=5,
    )

    assert len(results) >= 1
    assert results[0][0] == "test_rule"


def test_chroma_list_rules(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")

    rule1 = Rule(name="rule1", description="First", when=[], collection="default")
    rule2 = Rule(name="rule2", description="Second", when=[], collection="default")

    store.index_rule(rule1)
    store.index_rule(rule2)

    rules = store.list_rules("default")
    assert "rule1" in rules
    assert "rule2" in rules


def test_chroma_index_and_query_action(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")

    store.index_action("fix_rename", "Fix a module rename issue", "(workspace: str) -> dict")

    results = store.query_actions("rename module path")
    assert len(results) >= 1
    assert results[0]["name"] == "fix_rename"


def test_chroma_update_rule_stats(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")

    rule = Rule(name="test_rule", description="Test", when=[], collection="default")
    store.index_rule(rule)

    store.update_rule_stats("default", "test_rule", success=True, cost=0.01)
    store.update_rule_stats("default", "test_rule", success=True, cost=0.02)
    store.update_rule_stats("default", "test_rule", success=False, cost=0.01)

    stats = store.get_all_rules_with_stats()
    rule_stats = next(s for s in stats if s["name"] == "test_rule")

    assert rule_stats["success_count"] == 2
    assert rule_stats["fail_count"] == 1
    assert rule_stats["cost"] == pytest.approx(0.04)


def test_chroma_metadata_keys(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")

    rule = Rule(
        name="test",
        description="Test",
        when=[Fact(fact="problem_type", equals="build")],
        collection="default",
    )
    store.index_rule(rule)

    keys = store.get_metadata_keys()
    assert "problem_type" in keys
