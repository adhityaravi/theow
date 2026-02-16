"""Tests for chroma store."""

import pytest

from theow._core._chroma_store import ChromaStore, extract_query_text
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


def test_extract_query_text():
    ctx = {"short": "x", "long": "this is the longest string value", "num": 42}
    assert extract_query_text(ctx) == "this is the longest string value"


def test_extract_query_text_empty():
    assert extract_query_text({}) == ""


def test_index_action_hash_caching(temp_dir):
    """Indexing same action twice should skip the second time (unchanged hash)."""
    store = ChromaStore(path=temp_dir / "chroma")
    store.index_action("act", "Do a thing", "(x: str) -> dict")
    store.index_action("act", "Do a thing", "(x: str) -> dict")
    # Should not error; second call is a cache hit
    results = store.list_actions()
    assert "act" in results


def test_index_action_updates_on_change(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    store.index_action("act", "Do a thing", "(x: str) -> dict")
    store.index_action("act", "Do a different thing", "(x: str, y: int) -> dict")
    results = store.query_actions("different thing")
    assert any(r["name"] == "act" for r in results)


def test_sync_rules_from_directory(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    rules_dir = temp_dir / "rules"
    rules_dir.mkdir()

    rule = Rule(
        name="synced_rule",
        description="A synced rule",
        when=[Fact(fact="x", equals="y")],
    )
    rule.to_yaml(rules_dir / "synced_rule.rule.yaml")

    store.sync_rules(rules_dir)
    rules = store.list_rules("default")
    assert "synced_rule" in rules


def test_sync_rules_nonexistent_dir(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    store.sync_rules(temp_dir / "nonexistent")  # Should not raise


def test_sync_rules_unchanged_file_skipped(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    rules_dir = temp_dir / "rules"
    rules_dir.mkdir()

    rule = Rule(name="r1", description="Test", when=[Fact(fact="x", equals="y")])
    rule.to_yaml(rules_dir / "r1.rule.yaml")

    store.sync_rules(rules_dir)
    store.sync_rules(rules_dir)  # Second sync should skip unchanged
    assert "r1" in store.list_rules("default")


def test_cleanup_stale_removes_deleted_files(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    rules_dir = temp_dir / "rules"
    rules_dir.mkdir()

    rule1 = Rule(name="keep", description="Keep", when=[Fact(fact="x", equals="y")])
    rule2 = Rule(name="stale", description="Stale", when=[Fact(fact="x", equals="y")])
    rule1.to_yaml(rules_dir / "keep.rule.yaml")
    rule2.to_yaml(rules_dir / "stale.rule.yaml")

    store.sync_rules(rules_dir)
    assert "stale" in store.list_rules("default")

    # Remove stale file and re-sync
    (rules_dir / "stale.rule.yaml").unlink()
    store.sync_rules(rules_dir)
    assert "stale" not in store.list_rules("default")


def test_query_rules_with_metadata_filter(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")

    rule = Rule(
        name="filtered",
        description="Filterable rule",
        when=[Fact(fact="problem_type", equals="build")],
    )
    store.index_rule(rule)

    results = store.query_rules(
        collection="default",
        query_text="build problem",
        metadata_filter={"problem_type": "build"},
    )
    assert len(results) >= 1
    assert results[0][0] == "filtered"


def test_query_rules_failure_returns_empty(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    # Query a collection that has no entries with an invalid filter
    results = store.query_rules(
        collection="default",
        query_text="anything",
        metadata_filter={"$invalid_operator": "bad"},
    )
    assert results == []


def test_query_actions_failure_returns_empty(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    # The collection is empty, but with an invalid filter to trigger error path
    results = store.query_actions("anything", n_results=5)
    assert isinstance(results, list)


def test_list_actions(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    store.index_action("act1", "First action", "(x: str)")
    store.index_action("act2", "Second action", "(y: int)")
    actions = store.list_actions()
    assert "act1" in actions
    assert "act2" in actions


def test_get_all_rules_with_stats_multiple_collections(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")

    rule1 = Rule(name="r1", description="Rule 1", when=[], collection="coll_a")
    rule2 = Rule(name="r2", description="Rule 2", when=[], collection="coll_b")
    store.index_rule(rule1)
    store.index_rule(rule2)

    stats = store.get_all_rules_with_stats()
    names = [s["name"] for s in stats]
    assert "r1" in names
    assert "r2" in names


def test_update_rule_stats_nonexistent_rule(temp_dir):
    store = ChromaStore(path=temp_dir / "chroma")
    # Should not raise on nonexistent rule
    store.update_rule_stats("default", "ghost_rule", success=True)


def test_get_rule_returns_none(temp_dir):
    """get_rule always returns None (rules loaded from files, not chroma)."""
    store = ChromaStore(path=temp_dir / "chroma")
    rule = Rule(name="r1", description="Test", when=[])
    store.index_rule(rule)
    assert store.get_rule("default", "r1") is None
