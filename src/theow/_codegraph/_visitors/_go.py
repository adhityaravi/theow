"""Go language visitor using tree-sitter."""

from __future__ import annotations

from pathlib import Path

import tree_sitter_go as tsgo
from tree_sitter import Language, Node as TSNode, Parser

from theow._codegraph._models import Edge, Node
from theow._codegraph._visitors._utils import child_by_type, last_child_by_type, text

GO_LANGUAGE = Language(tsgo.language())


class GoVisitor:
    """Extract nodes and edges from Go source files."""

    extensions: list[str] = [".go"]

    def __init__(self) -> None:
        self._parser = Parser(GO_LANGUAGE)

    def parse_file(
        self, path: Path, source: str, relative_path: str
    ) -> tuple[list[Node], list[Edge]]:
        """Parse a Go file and extract structural nodes and edges."""
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

        for child in tree.root_node.children:
            if child.type == "function_declaration":
                self._extract_function(child, module_id, relative_path, nodes, edges)
            elif child.type == "method_declaration":
                self._extract_method(child, module_id, relative_path, nodes, edges)
            elif child.type == "type_declaration":
                self._extract_type_declaration(child, module_id, relative_path, nodes, edges)
            elif child.type == "import_declaration":
                self._extract_imports(child, module_id, relative_path, edges)

        return nodes, edges

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

        params = child_by_type(node, "parameter_list")
        result = child_by_type(node, "result")
        sig = f"func {name}"
        if params:
            sig += text(params)
        if result:
            sig += " " + text(result)

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
                signature=sig,
                parent=parent_id,
            )
        )
        edges.append(Edge(source=parent_id, target=func_id, kind="contains", line=start_line))

        body = child_by_type(node, "block")
        if body:
            self._extract_calls_in_body(body, func_id, relative_path, edges)

    def _extract_method(
        self,
        node: TSNode,
        module_id: str,
        relative_path: str,
        nodes: list[Node],
        edges: list[Edge],
    ) -> None:
        name_node = child_by_type(node, "field_identifier")
        if not name_node:
            return

        name = text(name_node)

        receiver_node = child_by_type(node, "parameter_list")
        receiver_type = ""
        if receiver_node:
            for ch in receiver_node.children:
                if ch.type == "parameter_declaration":
                    type_node = last_child_by_type(
                        ch, "type_identifier"
                    ) or last_child_by_type(ch, "pointer_type")
                    if type_node:
                        receiver_type = text(type_node).lstrip("*")
                    break

        receiver_id = f"{module_id}::{receiver_type}" if receiver_type else module_id
        method_id = f"{receiver_id}::{name}"

        params_nodes = [c for c in node.children if c.type == "parameter_list"]
        params_text = text(params_nodes[1]) if len(params_nodes) > 1 else "()"
        result = child_by_type(node, "result")
        sig = f"func ({receiver_type}) {name}{params_text}"
        if result:
            sig += " " + text(result)

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        nodes.append(
            Node(
                id=method_id,
                kind="function",
                name=name,
                file=relative_path,
                line=start_line,
                end_line=end_line,
                signature=sig,
                parent=receiver_id,
            )
        )
        edges.append(
            Edge(source=receiver_id, target=method_id, kind="contains", line=start_line)
        )

        body = child_by_type(node, "block")
        if body:
            self._extract_calls_in_body(body, method_id, relative_path, edges)

    def _extract_type_declaration(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        nodes: list[Node],
        edges: list[Edge],
    ) -> None:
        for child in node.children:
            if child.type == "type_spec":
                self._extract_type_spec(child, parent_id, relative_path, nodes, edges)

    def _extract_type_spec(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        nodes: list[Node],
        edges: list[Edge],
    ) -> None:
        name_node = child_by_type(node, "type_identifier")
        if not name_node:
            return

        name = text(name_node)
        type_id = f"{parent_id}::{name}"

        struct_type = child_by_type(node, "struct_type")
        kind = "class"

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        nodes.append(
            Node(
                id=type_id,
                kind=kind,
                name=name,
                file=relative_path,
                line=start_line,
                end_line=end_line,
                parent=parent_id,
            )
        )
        edges.append(Edge(source=parent_id, target=type_id, kind="contains", line=start_line))

        if struct_type:
            field_list = child_by_type(struct_type, "field_declaration_list")
            if field_list:
                for field in field_list.children:
                    if field.type == "field_declaration":
                        children = [c for c in field.children if c.type != "comment"]
                        if len(children) == 1 and children[0].type == "type_identifier":
                            embedded = text(children[0])
                            edges.append(
                                Edge(
                                    source=type_id,
                                    target=embedded,
                                    kind="inherits",
                                    line=field.start_point[0] + 1,
                                )
                            )

    def _extract_imports(
        self,
        node: TSNode,
        parent_id: str,
        relative_path: str,
        edges: list[Edge],
    ) -> None:
        """Extract import declarations."""
        for child in node.children:
            if child.type == "import_spec":
                path_node = child_by_type(child, "interpreted_string_literal")
                if path_node:
                    import_path = text(path_node).strip('"')
                    edges.append(
                        Edge(
                            source=parent_id,
                            target=import_path,
                            kind="imports",
                            line=child.start_point[0] + 1,
                        )
                    )
            elif child.type == "import_spec_list":
                for spec in child.children:
                    if spec.type == "import_spec":
                        path_node = child_by_type(spec, "interpreted_string_literal")
                        if path_node:
                            import_path = text(path_node).strip('"')
                            edges.append(
                                Edge(
                                    source=parent_id,
                                    target=import_path,
                                    kind="imports",
                                    line=spec.start_point[0] + 1,
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
            if child.type == "call_expression":
                self._extract_call(child, parent_id, relative_path, edges)
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
        elif func.type == "selector_expression":
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

        arg_list = child_by_type(node, "argument_list")
        if arg_list:
            self._extract_calls_in_body(arg_list, parent_id, relative_path, edges)
