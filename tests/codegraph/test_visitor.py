"""Tests for language visitors using theow's own source."""

import tempfile
from pathlib import Path

from theow._codegraph._visitors._go import GoVisitor
from theow._codegraph._visitors._python import PythonVisitor

GO_SOURCE = '''\
package main

import (
\t"fmt"
\t"strings"
)

type Animal interface {
\tSpeak() string
}

type Base struct {
\tName string
}

type Dog struct {
\tBase
\tBreed string
}

func NewDog(name string, breed string) *Dog {
\treturn &Dog{Base: Base{Name: name}, Breed: breed}
}

func (d *Dog) Speak() string {
\treturn fmt.Sprintf("Woof! I am %s", d.Name)
}

func (d *Dog) Fetch(item string) string {
\treturn strings.ToUpper(item)
}

func main() {
\tdog := NewDog("Rex", "Labrador")
\tfmt.Println(dog.Speak())
\tfmt.Println(dog.Fetch("ball"))
}
'''


def test_python_visitor_parses_models(theow_src):
    """Parse theow's _core/_models.py — has classes, methods, imports."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    nodes, edges = visitor.parse_file(path, source, relative)

    names = {n.name for n in nodes}
    assert "Rule" in names
    assert "Fact" in names
    assert "Action" in names
    assert "LLMConfig" in names

    # Module node exists
    kinds = {n.kind for n in nodes}
    assert "module" in kinds
    assert "class" in kinds
    assert "function" in kinds


def test_python_visitor_extracts_methods(theow_src):
    """Methods inside classes should have the class as parent."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    nodes, _ = visitor.parse_file(path, source, relative)

    # to_dict should be a method of Fact
    fact_methods = [n for n in nodes if n.parent == f"{relative}::Fact"]
    method_names = {n.name for n in fact_methods}
    assert "to_dict" in method_names
    assert "from_dict" in method_names
    assert "matches" in method_names


def test_python_visitor_extracts_imports(theow_src):
    """Module-level imports should produce import edges."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    _, edges = visitor.parse_file(path, source, relative)

    import_edges = [e for e in edges if e.kind == "imports"]
    import_targets = {e.target for e in import_edges}
    assert "yaml" in import_targets
    assert "re" in import_targets


def test_python_visitor_extracts_inheritance(theow_src):
    """Classes with bases should produce inherits edges."""
    visitor = PythonVisitor()
    # _gateway/_base.py has LLMGateway(ABC)
    path = theow_src / "_gateway" / "_base.py"
    source = path.read_text()
    relative = "_gateway/_base.py"

    _, edges = visitor.parse_file(path, source, relative)

    inherits = [e for e in edges if e.kind == "inherits"]
    assert any(e.target == "ABC" for e in inherits)


def test_python_visitor_extracts_docstrings(theow_src):
    """Functions and classes should have docstrings extracted."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    nodes, _ = visitor.parse_file(path, source, relative)

    rule_node = next(n for n in nodes if n.name == "Rule")
    assert "production rule" in rule_node.docstring.lower()


def test_python_visitor_extracts_signatures(theow_src):
    """Functions should have their signature extracted."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    nodes, _ = visitor.parse_file(path, source, relative)

    matches_node = next(n for n in nodes if n.name == "matches" and "Fact" in n.parent)
    assert "def matches" in matches_node.signature
    assert "self" in matches_node.signature


def test_python_visitor_extracts_calls(theow_src):
    """Call sites should produce call edges."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    _, edges = visitor.parse_file(path, source, relative)

    call_edges = [e for e in edges if e.kind == "calls"]
    call_targets = {e.target for e in call_edges}
    # Fact.matches uses re.search
    assert any("re.search" in t for t in call_targets)


def test_python_visitor_line_numbers(theow_src):
    """Node line numbers should be positive and end_line >= line."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    nodes, _ = visitor.parse_file(path, source, relative)

    for node in nodes:
        assert node.line >= 1
        if node.end_line:
            assert node.end_line >= node.line


def test_python_visitor_decorated_functions(theow_src):
    """Decorated definitions should still be extracted."""
    visitor = PythonVisitor()
    path = theow_src / "_core" / "_models.py"
    source = path.read_text()
    relative = "_core/_models.py"

    nodes, _ = visitor.parse_file(path, source, relative)

    # @property is_ephemeral on Rule
    node_names = {n.name for n in nodes}
    assert "is_ephemeral" in node_names


def test_python_visitor_extensions():
    visitor = PythonVisitor()
    assert visitor.extensions == [".py"]


def test_load_visitors_python():
    from theow._codegraph._visitors import load_visitors

    visitors = load_visitors(["python"])
    assert len(visitors) == 1
    assert visitors[0].extensions == [".py"]


def test_load_visitors_unknown():
    import pytest
    from theow._codegraph._visitors import load_visitors

    with pytest.raises(ValueError, match="Unknown language"):
        load_visitors(["rust"])


# --- Go visitor tests ---


def _parse_go(source=GO_SOURCE, filename="main.go"):
    visitor = GoVisitor()
    with tempfile.NamedTemporaryFile(suffix=".go", mode="w", delete=False) as f:
        f.write(source)
        path = Path(f.name)
    nodes, edges = visitor.parse_file(path, source, filename)
    path.unlink()
    return nodes, edges


def test_go_visitor_extensions():
    assert GoVisitor().extensions == [".go"]


def test_go_visitor_extracts_functions():
    nodes, _ = _parse_go()
    names = {n.name for n in nodes if n.kind == "function"}
    assert "NewDog" in names
    assert "main" in names


def test_go_visitor_extracts_methods():
    """Methods should have the receiver type as parent."""
    nodes, _ = _parse_go()
    speak = next(n for n in nodes if n.name == "Speak")
    assert "Dog" in speak.parent
    assert "func (*Dog) Speak" in speak.signature or "func (Dog) Speak" in speak.signature


def test_go_visitor_extracts_structs():
    nodes, _ = _parse_go()
    struct_names = {n.name for n in nodes if n.kind == "class"}
    assert "Dog" in struct_names
    assert "Base" in struct_names


def test_go_visitor_extracts_interfaces():
    nodes, _ = _parse_go()
    assert any(n.name == "Animal" and n.kind == "class" for n in nodes)


def test_go_visitor_extracts_imports():
    _, edges = _parse_go()
    import_edges = [e for e in edges if e.kind == "imports"]
    targets = {e.target for e in import_edges}
    assert "fmt" in targets
    assert "strings" in targets


def test_go_visitor_extracts_calls():
    _, edges = _parse_go()
    call_edges = [e for e in edges if e.kind == "calls"]
    targets = {e.target for e in call_edges}
    assert "NewDog" in targets
    # Method calls like dog.Speak()
    assert any("Speak" in t for t in targets)


def test_go_visitor_extracts_embedding():
    """Struct embedding (Base in Dog) should produce inherits edge."""
    _, edges = _parse_go()
    inherits = [e for e in edges if e.kind == "inherits"]
    assert any(e.target == "Base" and "Dog" in e.source for e in inherits)


def test_go_visitor_function_signatures():
    nodes, _ = _parse_go()
    new_dog = next(n for n in nodes if n.name == "NewDog")
    assert "func NewDog" in new_dog.signature
    assert "string" in new_dog.signature


def test_go_visitor_method_signatures():
    nodes, _ = _parse_go()
    fetch = next(n for n in nodes if n.name == "Fetch")
    assert "func" in fetch.signature
    assert "Dog" in fetch.signature
    assert "item" in fetch.signature


def test_go_visitor_line_numbers():
    nodes, _ = _parse_go()
    for node in nodes:
        assert node.line >= 1
        if node.end_line:
            assert node.end_line >= node.line


def test_go_visitor_contains_edges():
    """Module should contain functions and types."""
    _, edges = _parse_go()
    contains = [e for e in edges if e.kind == "contains"]
    assert len(contains) > 0
    # Module contains functions
    assert any(e.source == "main.go" for e in contains)


def test_load_visitors_go():
    from theow._codegraph._visitors import load_visitors

    visitors = load_visitors(["go"])
    assert len(visitors) == 1
    assert visitors[0].extensions == [".go"]


def test_load_visitors_both():
    from theow._codegraph._visitors import load_visitors

    visitors = load_visitors(["python", "go"])
    assert len(visitors) == 2
