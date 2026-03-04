"""Tests for codegraph data models."""

from theow._codegraph._models import Edge, Node, SearchResult


def test_node_to_dict_minimal():
    node = Node(id="mod.py::func", kind="function", name="func", file="mod.py", line=10)
    d = node.to_dict()
    assert d["id"] == "mod.py::func"
    assert d["kind"] == "function"
    assert d["line"] == 10
    assert "end_line" not in d
    assert "signature" not in d
    assert "docstring" not in d
    assert "parent" not in d


def test_node_to_dict_full():
    node = Node(
        id="mod.py::Cls.method",
        kind="function",
        name="method",
        file="mod.py",
        line=5,
        end_line=15,
        signature="def method(self, x)",
        docstring="Do stuff.",
        parent="mod.py::Cls",
    )
    d = node.to_dict()
    assert d["end_line"] == 15
    assert d["signature"] == "def method(self, x)"
    assert d["docstring"] == "Do stuff."
    assert d["parent"] == "mod.py::Cls"


def test_node_roundtrip():
    node = Node(
        id="a.py::X",
        kind="class",
        name="X",
        file="a.py",
        line=1,
        end_line=20,
        signature="",
        docstring="A class.",
        parent="a.py",
    )
    assert Node.from_dict(node.to_dict()) == node


def test_node_frozen():
    node = Node(id="x", kind="module", name="x", file="x.py", line=1)
    try:
        node.name = "y"  # type: ignore[misc]
        assert False, "Should be frozen"
    except AttributeError:
        pass


def test_edge_to_dict_minimal():
    edge = Edge(source="a", target="b", kind="calls")
    d = edge.to_dict()
    assert d == {"source": "a", "target": "b", "kind": "calls"}
    assert "line" not in d


def test_edge_to_dict_with_line():
    edge = Edge(source="a", target="b", kind="imports", line=7)
    d = edge.to_dict()
    assert d["line"] == 7


def test_edge_roundtrip():
    edge = Edge(source="a", target="b", kind="contains", line=3)
    assert Edge.from_dict(edge.to_dict()) == edge


def test_search_result_to_dict():
    node = Node(id="f.py::go", kind="function", name="go", file="f.py", line=1)
    sr = SearchResult(node=node, context="called by main", relevance="caller")
    d = sr.to_dict()
    assert d["id"] == "f.py::go"
    assert d["context"] == "called by main"
    assert d["relevance"] == "caller"


def test_search_result_to_dict_no_extras():
    node = Node(id="f.py", kind="module", name="f", file="f.py", line=1)
    sr = SearchResult(node=node)
    d = sr.to_dict()
    assert "context" not in d
    assert "relevance" not in d


def test_search_result_roundtrip():
    node = Node(id="x.py::Y", kind="class", name="Y", file="x.py", line=2)
    sr = SearchResult(node=node, context="ctx", relevance="exact")
    restored = SearchResult.from_dict(sr.to_dict())
    assert restored.node == node
    assert restored.context == "ctx"
    assert restored.relevance == "exact"
