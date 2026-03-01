"""Tests for codegraph tool registration using theow's own source."""

import inspect

from theow._codegraph._graph import CodeGraph
from theow._gateway._base import build_tool_declaration


def test_search_code_callable(theow_src):
    """graph.search_code should be a callable that returns list[dict]."""
    graph = CodeGraph(root=theow_src)
    assert callable(graph.search_code)

    results = graph.search_code(query="Rule")
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


def test_search_code_has_docstring(theow_src):
    """search_code should have a docstring for LLM tool schema."""
    graph = CodeGraph(root=theow_src)
    doc = inspect.getdoc(graph.search_code)
    assert doc is not None
    assert "symbol" in doc.lower()


def test_build_tool_declaration_skips_self(theow_src):
    """build_tool_declaration should skip self for bound methods."""
    graph = CodeGraph(root=theow_src)
    decl = build_tool_declaration("search_code", graph.search_code)

    props = decl["input_schema"]["properties"]
    assert "self" not in props
    assert "query" in props
    assert "scope" in props
    assert "kind" in props
    assert "file" in props
    assert "line" in props
    assert "target" in props


def test_build_tool_declaration_types(theow_src):
    """Tool declaration should have correct JSON schema types."""
    graph = CodeGraph(root=theow_src)
    decl = build_tool_declaration("search_code", graph.search_code)

    props = decl["input_schema"]["properties"]
    assert props["query"]["type"] == "string"
    assert props["line"]["type"] == "integer"
    assert props["scope"]["type"] == "string"


def test_build_tool_declaration_no_required(theow_src):
    """All search_code params have defaults, so none should be required."""
    graph = CodeGraph(root=theow_src)
    decl = build_tool_declaration("search_code", graph.search_code)

    assert decl["input_schema"]["required"] == []


def test_build_tool_declaration_has_description(theow_src):
    """Tool declaration should carry the docstring as description."""
    graph = CodeGraph(root=theow_src)
    decl = build_tool_declaration("search_code", graph.search_code)

    assert len(decl["description"]) > 50


def test_public_import():
    """CodeGraph should be importable from the public API."""
    from theow.codegraph import CodeGraph as PublicCodeGraph

    assert PublicCodeGraph is CodeGraph


def test_search_code_returns_dicts_with_expected_keys(theow_src):
    """Results should contain the standard node fields."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(query="Rule", kind="class")

    assert len(results) > 0
    r = results[0]
    assert "id" in r
    assert "kind" in r
    assert "name" in r
    assert "file" in r
    assert "line" in r
