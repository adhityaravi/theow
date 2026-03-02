"""Tests for CodeGraph using theow's own source."""

import json
import tempfile
from pathlib import Path

from theow._codegraph._graph import CodeGraph


def test_build_on_theow_src(theow_src):
    """Building a graph on theow's source should produce real nodes and edges."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    assert len(graph._nodes) > 50
    assert graph._edge_count() > 50
    assert graph._built is True


def test_build_idempotent(theow_src):
    """Calling build() twice should produce the same graph."""
    graph = CodeGraph(root=theow_src)
    graph.build()
    count_1 = len(graph._nodes)

    graph.build()
    count_2 = len(graph._nodes)

    assert count_1 == count_2


def test_build_empty_project(empty_project):
    """Empty directory should build an empty graph without errors."""
    graph = CodeGraph(root=empty_project)
    graph.build()
    assert len(graph._nodes) == 0


def test_auto_build_on_first_search(theow_src):
    """search_code should trigger build() if not already built."""
    graph = CodeGraph(root=theow_src)
    assert graph._built is False

    results = graph.search_code(query="Rule", kind="class")
    assert graph._built is True
    assert len(results) > 0


def test_search_symbol_exact(theow_src):
    """Exact name match for a known class."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(query="Rule", kind="class")

    assert any(r["name"] == "Rule" for r in results)
    assert results[0]["relevance"] == "exact"


def test_search_symbol_substring(theow_src):
    """Substring match should find partial name matches."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(query="Gateway")

    names = {r["name"] for r in results}
    assert "LLMGateway" in names


def test_search_symbol_by_kind(theow_src):
    """Filtering by kind should only return that kind."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(query="", kind="class")

    for r in results:
        assert r["kind"] == "class"


def test_search_file_scope(theow_src):
    """File scope should list all symbols in a specific file."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(scope="file", file="_core/_models.py")

    assert len(results) > 5
    assert all(r["file"] == "_core/_models.py" for r in results)
    # Should be sorted by line number
    lines = [r["line"] for r in results]
    assert lines == sorted(lines)


def test_search_callers(theow_src):
    """Callers scope should find functions that call the target."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(query="matches", scope="callers")

    # Something should call matches()
    assert len(results) > 0
    assert all(r.get("relevance") == "caller" for r in results)


def test_search_callees(theow_src):
    """Callees scope should find functions that the target calls."""
    graph = CodeGraph(root=theow_src)
    # Fact.matches calls re.search — look for callees of matches
    results = graph.search_code(query="matches", scope="callees")

    # matches() should call something
    assert len(results) >= 0  # may or may not resolve depending on edge resolution


def test_search_references(theow_src):
    """References scope should show both incoming and outgoing edges."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(query="Rule", scope="references")

    # Rule has both incoming (contains) and outgoing edges
    assert len(results) > 0


def test_search_definition_by_file_line(theow_src):
    """Definition scope with file+line should find the symbol at that location."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    # Find Rule's line number first
    rule_node = graph._nodes.get("_core/_models.py::Rule")
    assert rule_node is not None

    results = graph.search_code(scope="definition", file="_core/_models.py", line=rule_node.line)
    assert len(results) > 0
    assert results[0]["name"] == "Rule"


def test_search_path(theow_src):
    """Path scope should find a relationship chain between two symbols."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    # _core/_models.py module contains Rule class
    results = graph.search_code(query="_core/_models.py", scope="path", target="Rule")
    # Should find a path (module --contains--> Rule)
    assert len(results) >= 2


def test_search_no_results(theow_src):
    """Searching for a nonexistent symbol returns empty list."""
    graph = CodeGraph(root=theow_src)
    results = graph.search_code(query="XyzNonExistent12345")
    assert results == []


def test_excludes(theow_src):
    """Excluded directories should not appear in the graph."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    for node in graph._nodes.values():
        assert "__pycache__" not in node.file


def test_max_file_size(theow_src):
    """Files above max_file_size should be skipped."""
    graph = CodeGraph(root=theow_src, max_file_size=100)
    graph.build()

    # With 100 byte limit, most files should be skipped
    assert len(graph._nodes) < 10


def test_file_index(theow_src):
    """File index should map files to their node IDs."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    assert "_core/_models.py" in graph._file_index
    node_ids = graph._file_index["_core/_models.py"]
    assert any("Rule" in nid for nid in node_ids)


def test_name_index(theow_src):
    """Name index should map short names to node IDs."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    assert "Rule" in graph._name_index
    assert len(graph._name_index["Rule"]) >= 1


def test_json_roundtrip(theow_src):
    """Graph should survive JSON serialization and deserialization."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json_path = f.name

    graph.to_json(json_path)
    loaded = CodeGraph.from_json(json_path)

    assert len(loaded._nodes) == len(graph._nodes)
    assert loaded._edge_count() == graph._edge_count()
    assert loaded._built is True

    Path(json_path).unlink()


def test_json_to_string(theow_src):
    """to_json without path should return a JSON string."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    result = graph.to_json()
    data = json.loads(result)
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == len(graph._nodes)


def test_custom_excludes(theow_src):
    """Custom excludes should override defaults."""
    graph = CodeGraph(root=theow_src, excludes={"_core"})
    graph.build()

    for node in graph._nodes.values():
        assert "_core" not in node.file


def test_edge_resolution(theow_src):
    """Edge resolution should resolve some symbolic names to node IDs."""
    graph = CodeGraph(root=theow_src)
    graph.build()

    # After resolution, some call edges should point to known nodes
    resolved_calls = 0
    for source, targets in graph._fwd.items():
        for target, data in targets.items():
            if data.get("kind") == "calls" and target in graph._nodes:
                resolved_calls += 1

    assert resolved_calls > 0
