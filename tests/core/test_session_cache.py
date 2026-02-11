"""Tests for session cache."""

from theow._core._models import Fact, Rule
from theow._core._session_cache import SessionCache


def test_session_cache_miss_on_empty():
    cache = SessionCache()
    result = cache.check({"problem": "something"})
    assert result is None


def test_session_cache_hit_on_similar():
    cache = SessionCache(similarity_threshold=0.8)

    rule = Rule(name="test", description="Test", when=[Fact(fact="x", equals="y")])
    cache.store({"problem": "module declares path as X"}, rule)

    result = cache.check({"problem": "module declares path as Y"})
    assert result is not None
    assert result.name == "test"


def test_session_cache_miss_on_different():
    cache = SessionCache(similarity_threshold=0.9)

    rule = Rule(name="test", description="Test", when=[])
    cache.store({"problem": "module path issue"}, rule)

    result = cache.check({"problem": "completely different error about networking"})
    assert result is None


def test_session_cache_clear():
    cache = SessionCache()

    rule = Rule(name="test", description="Test", when=[])
    cache.store({"problem": "x"}, rule)
    assert cache.size == 1

    cache.clear()
    assert cache.size == 0
