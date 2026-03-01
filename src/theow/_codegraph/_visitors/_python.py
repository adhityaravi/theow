"""Python language visitor using tree-sitter."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node as TSNode, Parser

from theow._codegraph._models import Edge, Node
from theow._codegraph._visitors._utils import child_by_type, text

PY_LANGUAGE = Language(tspython.language())


class PythonVisitor:
    """Extract nodes and edges from Python source files."""

    extensions: list[str] = [".py"]

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def parse_file(
        self, path: Path, source: str, relative_path: str
    ) -> tuple[list[Node], list[Edge]]:
        """Parse a Python file and extract structural nodes and edges."""
        tree = self._parser.parse(source.encode())
        nodes: list[Node] = []
        edges: list[Edge] = []

        module_id = relative_path
        nodes.append(
            Node(
                id=module_id,
                kind="module",
                name=Path(relative_path).stem,
                file=relative_path,
                line=1,
                end_line=source.count("\n") + 1,
            )
        )

        self._walk(tree.root_node, module_id, relative_path, nodes, edges)
        return nodes, edges

    def _walk(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        nodes: list[Node],
        edges: list[Edge],
    ) -> None:
        """Recursively walk the AST and extract symbols."""
        for child in node.children:
            if child.type == "function_definition":
                self._extract_function(child, parent_id, relative_path, nodes, edges)
            elif child.type == "decorated_definition":
                for inner in child.children:
                    if inner.type == "function_definition":
                        self._extract_function(
                            inner, parent_id, relative_path, nodes, edges
                        )
                    elif inner.type == "class_definition":
                        self._extract_class(
                            inner, parent_id, relative_path, nodes, edges
                        )
            elif child.type == "class_definition":
                self._extract_class(child, parent_id, relative_path, nodes, edges)
            elif child.type == "import_statement":
                self._extract_import(child, parent_id, relative_path, edges)
            elif child.type == "import_from_statement":
                self._extract_import_from(child, parent_id, relative_path, edges)
            elif child.type == "expression_statement":
                expr = child.children[0] if child.children else None
                if expr and expr.type == "call":
                    self._extract_call(expr, parent_id, relative_path, edges)

    def _extract_function(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        nodes: list[Node],
        edges: list[Edge],
    ) -> None:
        name_node = child_by_type(node, "identifier")
        if not name_node:
            return

        name = text(name_node)
        func_id = f"{parent_id}::{name}"

        params_node = child_by_type(node, "parameters")
        signature = f"def {name}{text(params_node)}" if params_node else f"def {name}()"

        docstring = _extract_docstring(node)

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        nodes.append(
            Node(
                id=func_id,
                kind="function",
                name=name,
                file=relative_path,
                line=start_line,
                end_line=end_line,
                signature=signature,
                docstring=docstring,
                parent=parent_id,
            )
        )
        edges.append(Edge(source=parent_id, target=func_id, kind="contains", line=start_line))

        body = child_by_type(node, "block")
        if body:
            self._extract_calls_in_body(body, func_id, relative_path, edges)

    def _extract_class(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        nodes: list[Node],
        edges: list[Edge],
    ) -> None:
        name_node = child_by_type(node, "identifier")
        if not name_node:
            return

        name = text(name_node)
        class_id = f"{parent_id}::{name}"

        arg_list = child_by_type(node, "argument_list")
        if arg_list:
            for arg in arg_list.children:
                if arg.type == "identifier":
                    edges.append(
                        Edge(
                            source=class_id,
                            target=text(arg),
                            kind="inherits",
                            line=arg.start_point[0] + 1,
                        )
                    )

        docstring = _extract_docstring(node)
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        nodes.append(
            Node(
                id=class_id,
                kind="class",
                name=name,
                file=relative_path,
                line=start_line,
                end_line=end_line,
                docstring=docstring,
                parent=parent_id,
            )
        )
        edges.append(Edge(source=parent_id, target=class_id, kind="contains", line=start_line))

        body = child_by_type(node, "block")
        if body:
            self._walk(body, class_id, relative_path, nodes, edges)

    def _extract_import(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        edges: list[Edge],
    ) -> None:
        """Extract `import foo` statements."""
        for child in node.children:
            if child.type == "dotted_name":
                edges.append(
                    Edge(
                        source=parent_id,
                        target=text(child),
                        kind="imports",
                        line=node.start_point[0] + 1,
                    )
                )

    def _extract_import_from(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        edges: list[Edge],
    ) -> None:
        """Extract `from foo import bar` statements."""
        module_name = ""
        for child in node.children:
            if child.type == "dotted_name" and not module_name:
                module_name = text(child)
            elif child.type == "relative_import":
                module_name = text(child)

        if module_name:
            edges.append(
                Edge(
                    source=parent_id,
                    target=module_name,
                    kind="imports",
                    line=node.start_point[0] + 1,
                )
            )

    def _extract_calls_in_body(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        edges: list[Edge],
    ) -> None:
        """Recursively find call expressions in a body block."""
        for child in node.children:
            if child.type == "call":
                self._extract_call(child, parent_id, relative_path, edges)
            elif child.type == "expression_statement":
                for inner in child.children:
                    if inner.type == "call":
                        self._extract_call(inner, parent_id, relative_path, edges)
                    else:
                        self._extract_calls_in_body(inner, parent_id, relative_path, edges)
            else:
                self._extract_calls_in_body(child, parent_id, relative_path, edges)

    def _extract_call(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        edges: list[Edge],
    ) -> None:
        """Extract a call expression into an edge."""
        func = node.children[0] if node.children else None
        if not func:
            return

        if func.type == "identifier":
            target = text(func)
        elif func.type == "attribute":
            target = text(func)
        else:
            return

        edges.append(
            Edge(
                source=parent_id,
                target=target,
                kind="calls",
                line=node.start_point[0] + 1,
            )
        )


def _extract_docstring(node: TSNode) -> str:
    """Extract docstring from a function or class definition."""
    body = child_by_type(node, "block")
    if not body:
        return ""

    children = body.children
    if not children:
        return ""

    first_stmt = children[0]
    if first_stmt.type == "expression_statement":
        expr = first_stmt.children[0] if first_stmt.children else None
        if expr and expr.type == "string":
            raw = text(expr)
            # Strip triple quotes
            for q in ('"""', "'''"):
                if raw.startswith(q) and raw.endswith(q):
                    return raw[3:-3].strip()
            # Strip single quotes
            for q in ('"', "'"):
                if raw.startswith(q) and raw.endswith(q):
                    return raw[1:-1].strip()
            return raw
    return ""
