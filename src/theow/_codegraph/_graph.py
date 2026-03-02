"""CodeGraph: tree-sitter based code structure graph."""

import json
from collections import deque
from pathlib import Path

from theow._codegraph._models import Edge, Node, SearchResult
from theow._codegraph._visitors import LanguageVisitor, load_visitors
from theow._core._logging import get_logger

logger = get_logger(__name__)

DEFAULT_EXCLUDES = frozenset(
    {
        "__pycache__",
        ".git",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
)


class CodeGraph:
    """A queryable graph of code symbols and relationships.

    Parses source files with tree-sitter visitors and builds a directed graph
    of modules, classes, functions, and their relationships (calls, imports,
    inheritance, containment).

    Args:
        root: Root directory of the codebase to index.
        languages: Language names to load visitors for (e.g. ["python", "go"]).
            Defaults to ["python"].
        visitors: Pre-built visitor instances. Overrides ``languages`` if given.
        excludes: Directory names to skip during traversal.
        max_file_size: Skip files larger than this (bytes).
    """

    def __init__(
        self,
        root: str | Path,
        languages: list[str] | None = None,
        visitors: list[LanguageVisitor] | None = None,
        excludes: set[str] | None = None,
        max_file_size: int = 1_000_000,
    ) -> None:
        self._root = Path(root).resolve()
        self._roots: list[Path] = [self._root]
        self._excludes = excludes if excludes is not None else set(DEFAULT_EXCLUDES)
        self._max_file_size = max_file_size
        self._built = False

        if visitors is not None:
            self._visitors = list(visitors)
        else:
            self._visitors = load_visitors(languages or ["python"])

        # Build extension -> visitor lookup
        self._ext_map: dict[str, LanguageVisitor] = {}
        for v in self._visitors:
            for ext in v.extensions:
                self._ext_map[ext] = v

        # Internal state — adjacency lists for directed edges
        self._fwd: dict[str, dict[str, dict]] = {}  # source -> {target -> edge_data}
        self._rev: dict[str, dict[str, dict]] = {}  # target -> {source -> edge_data}
        self._nodes: dict[str, Node] = {}
        self._file_index: dict[str, list[str]] = {}
        self._name_index: dict[str, list[str]] = {}

    def build(self) -> None:
        """Parse all files and build the graph. Idempotent."""
        if self._built:
            self._fwd.clear()
            self._rev.clear()
            self._nodes.clear()
            self._file_index.clear()
            self._name_index.clear()

        files_parsed = 0
        for root in self._roots:
            for path in self._iter_files_for_root(root):
                self._build_file(path, root)
                files_parsed += 1

        resolved, unresolved = self._resolve_edges()
        self._built = True
        logger.info(
            "Code graph built",
            files=files_parsed,
            nodes=len(self._nodes),
            edges=self._edge_count(),
            resolved_refs=resolved,
            unresolved_refs=unresolved,
        )

    def add_root(self, root: str | Path) -> None:
        """Add an additional source root (e.g. a dependency) to the graph.

        If the graph is already built, the new root is parsed immediately
        and edges are re-resolved.
        """
        resolved = Path(root).resolve()
        self._roots.append(resolved)

        if self._built:
            files_parsed = 0
            for path in self._iter_files_for_root(resolved):
                self._build_file(path, resolved)
                files_parsed += 1
            resolved_refs, unresolved_refs = self._resolve_edges()
            logger.info(
                "Root added to code graph",
                root=str(resolved),
                files=files_parsed,
                nodes=len(self._nodes),
                edges=self._edge_count(),
                resolved_refs=resolved_refs,
                unresolved_refs=unresolved_refs,
            )

    def build_file(self, path: Path) -> None:
        """Parse a single file and add its nodes/edges to the graph."""
        self._build_file(path, self._root)

    def _build_file(self, path: Path, root: Path) -> None:
        """Parse a single file relative to a given root."""
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return

        relative_path = str(path.relative_to(root))
        ext = path.suffix
        visitor = self._ext_map.get(ext)
        if not visitor:
            return

        try:
            nodes, edges = visitor.parse_file(path, source, relative_path)
        except Exception as e:
            logger.warning("Parse failed", file=relative_path, error=str(e))
            return

        for node in nodes:
            self._nodes[node.id] = node
            self._file_index.setdefault(node.file, []).append(node.id)
            self._name_index.setdefault(node.name, []).append(node.id)

        for edge in edges:
            self._add_edge(edge.source, edge.target, kind=edge.kind, line=edge.line)

    def _iter_files(self):
        """Walk all roots, yielding files that match visitor extensions."""
        for root in self._roots:
            yield from self._iter_files_for_root(root)

    def _iter_files_for_root(self, root: Path):
        """Walk a single root directory, yielding matching files."""
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in self._excludes for part in path.relative_to(root).parts):
                continue
            if path.stat().st_size > self._max_file_size:
                logger.debug("Skipping large file", file=str(path.relative_to(root)))
                continue
            if path.suffix in self._ext_map:
                yield path

    def _resolve_edges(self) -> tuple[int, int]:
        """Resolve symbolic call/import targets to fully qualified node IDs.

        For call edges where the target is a short name (e.g. "helper"),
        try to resolve it to a fully qualified node ID. Same-file preference
        breaks ambiguity.

        Returns:
            (resolved_count, unresolved_count) tuple.
        """
        edges_to_resolve = []
        for source, targets in self._fwd.items():
            for target, data in targets.items():
                if data.get("kind") in ("calls", "inherits") and target not in self._nodes:
                    edges_to_resolve.append((source, target, data))

        resolved_count = 0
        for source, target, data in edges_to_resolve:
            resolved = self._resolve_name(target, source)
            if resolved and resolved != target:
                self._remove_edge(source, target)
                self._add_edge(source, resolved, **data)
                resolved_count += 1

        return resolved_count, len(edges_to_resolve) - resolved_count

    def _resolve_name(self, name: str, context_id: str) -> str | None:
        """Resolve a short name to a node ID, preferring same-file then same-root."""
        # For attribute access like "self.method" or "obj.method", use the last part
        short = name.rsplit(".", 1)[-1] if "." in name else name

        candidates = self._name_index.get(short, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        context_file = self._nodes[context_id].file if context_id in self._nodes else ""

        # Prefer same-file
        for cid in candidates:
            if cid in self._nodes and self._nodes[cid].file == context_file:
                return cid

        # Prefer same-root (files sharing a common path prefix)
        if context_file:
            context_parts = Path(context_file).parts
            for cid in candidates:
                if cid in self._nodes:
                    cid_parts = Path(self._nodes[cid].file).parts
                    if context_parts and cid_parts and context_parts[0] == cid_parts[0]:
                        return cid

        return candidates[0]

    def _add_edge(self, source: str, target: str, **data) -> None:
        self._fwd.setdefault(source, {})[target] = data
        self._rev.setdefault(target, {})[source] = data

    def _remove_edge(self, source: str, target: str) -> None:
        if source in self._fwd:
            self._fwd[source].pop(target, None)
        if target in self._rev:
            self._rev[target].pop(source, None)

    def _edge_count(self) -> int:
        return sum(len(targets) for targets in self._fwd.values())

    def _bfs_path(self, source: str, target: str) -> list[str] | None:
        """BFS shortest path from source to target."""
        if source == target:
            return [source]
        visited = {source}
        queue: deque[list[str]] = deque([[source]])
        while queue:
            path = queue.popleft()
            for neighbor in self._fwd.get(path[-1], {}):
                if neighbor == target:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])
        return None

    def search_code(
        self,
        query: str = "",
        kind: str = "",
        scope: str = "symbol",
        file: str = "",
        line: int = 0,
        target: str = "",
    ) -> list[dict]:
        """Search the codebase structure. Navigate classes, functions, imports, and call relationships.

        Args:
            query: Symbol name or substring to search for.
            kind: Filter by type: "function", "class", "module", or "import".
            scope: What to search for:
                - "symbol": Find symbols by name (default)
                - "callers": Who calls this symbol?
                - "callees": What does this symbol call?
                - "references": All incoming/outgoing relationships
                - "definition": Where is this symbol defined?
                - "file": List all symbols in a file
                - "path": Find relationship path between query and target
            file: Filter to a specific file, or target file for "file" scope.
            line: Find the symbol at this line number in file.
            target: Target symbol for "path" scope.
        """
        if not self._built:
            self.build()

        logger.debug("search_code", scope=scope, query=query, kind=kind, file=file)

        if scope == "file":
            return self._search_file(file or query)
        if scope == "callers":
            return self._search_callers(query, kind)
        if scope == "callees":
            return self._search_callees(query, kind)
        if scope == "references":
            return self._search_references(query)
        if scope == "definition":
            return self._search_definition(query, file, line)
        if scope == "path":
            return self._search_path(query, target)
        # Default: symbol search
        return self._search_symbol(query, kind, file)

    def _search_symbol(self, query: str, kind: str, file: str) -> list[dict]:
        """Find symbols matching query by name."""
        results: list[SearchResult] = []
        for node_id, node in self._nodes.items():
            if kind and node.kind != kind:
                continue
            if file and node.file != file:
                continue
            if query and query.lower() not in node.name.lower() and query.lower() not in node_id.lower():
                continue
            relevance = "exact" if node.name == query else "substring"
            results.append(SearchResult(node=node, relevance=relevance))

        # Sort: exact matches first, then by file + line
        results.sort(key=lambda r: (r.relevance != "exact", r.node.file, r.node.line))
        return [r.to_dict() for r in results[:50]]

    def _search_file(self, file: str) -> list[dict]:
        """List all symbols in a file."""
        node_ids = self._file_index.get(file, [])
        results = []
        for nid in node_ids:
            node = self._nodes[nid]
            results.append(SearchResult(node=node).to_dict())
        results.sort(key=lambda r: r["line"])
        return results

    def _search_callers(self, query: str, kind: str) -> list[dict]:
        """Find symbols that call the queried symbol."""
        target_ids = self._find_node_ids(query, kind)
        results: list[dict] = []
        for tid in target_ids:
            for pred, edge_data in self._rev.get(tid, {}).items():
                if edge_data.get("kind") == "calls" and pred in self._nodes:
                    results.append(
                        SearchResult(
                            node=self._nodes[pred],
                            context=f"calls {tid}",
                            relevance="caller",
                        ).to_dict()
                    )
        return results

    def _search_callees(self, query: str, kind: str) -> list[dict]:
        """Find symbols that the queried symbol calls."""
        source_ids = self._find_node_ids(query, kind)
        results: list[dict] = []
        for sid in source_ids:
            for succ, edge_data in self._fwd.get(sid, {}).items():
                if edge_data.get("kind") == "calls" and succ in self._nodes:
                    results.append(
                        SearchResult(
                            node=self._nodes[succ],
                            context=f"called by {sid}",
                            relevance="callee",
                        ).to_dict()
                    )
        return results

    def _search_references(self, query: str) -> list[dict]:
        """Find all incoming/outgoing relationships for a symbol."""
        node_ids = self._find_node_ids(query, "")
        results: list[dict] = []
        for nid in node_ids:
            # Incoming
            for pred, edge_data in self._rev.get(nid, {}).items():
                edge_kind = edge_data.get("kind", "unknown")
                if pred in self._nodes:
                    results.append(
                        SearchResult(
                            node=self._nodes[pred],
                            context=f"{edge_kind} -> {nid}",
                            relevance="incoming",
                        ).to_dict()
                    )
            # Outgoing
            for succ, edge_data in self._fwd.get(nid, {}).items():
                edge_kind = edge_data.get("kind", "unknown")
                if succ in self._nodes:
                    results.append(
                        SearchResult(
                            node=self._nodes[succ],
                            context=f"{nid} {edge_kind} ->",
                            relevance="outgoing",
                        ).to_dict()
                    )
        return results

    def _search_definition(self, query: str, file: str, line: int) -> list[dict]:
        """Find the definition of a symbol."""
        if file and line:
            # Find symbol at file:line
            node_ids = self._file_index.get(file, [])
            for nid in node_ids:
                node = self._nodes[nid]
                if node.line <= line <= (node.end_line or node.line):
                    if node.kind != "module":
                        return [SearchResult(node=node, relevance="exact").to_dict()]

        # Fall back to name search
        return self._search_symbol(query, "", file)

    def _search_path(self, query: str, target: str) -> list[dict]:
        """Find relationship path between two symbols."""
        source_ids = self._find_node_ids(query, "")
        target_ids = self._find_node_ids(target, "")

        for sid in source_ids:
            for tid in target_ids:
                path = self._bfs_path(sid, tid)
                if path is None:
                    continue

                results: list[dict] = []
                for i, nid in enumerate(path):
                    if nid in self._nodes:
                        ctx = ""
                        if i < len(path) - 1:
                            edge_data = self._fwd.get(nid, {}).get(path[i + 1], {})
                            ctx = f"--{edge_data.get('kind', '?')}--> {path[i + 1]}"
                        results.append(
                            SearchResult(
                                node=self._nodes[nid],
                                context=ctx,
                                relevance=f"step {i}",
                            ).to_dict()
                        )
                return results

        return []

    def _find_node_ids(self, query: str, kind: str) -> list[str]:
        """Find node IDs matching a query string."""
        # Exact ID match
        if query in self._nodes:
            return [query]

        # Exact name match
        candidates = self._name_index.get(query, [])
        if candidates:
            if kind:
                candidates = [c for c in candidates if self._nodes[c].kind == kind]
            return candidates

        # Substring search
        results = []
        for nid, node in self._nodes.items():
            if kind and node.kind != kind:
                continue
            if query.lower() in node.name.lower():
                results.append(nid)
        return results

    def to_json(self, path: str | Path | None = None) -> str:
        """Serialize graph to JSON."""
        edges = []
        for source, targets in self._fwd.items():
            for target, data in targets.items():
                edges.append(
                    Edge(
                        source=source,
                        target=target,
                        kind=data.get("kind", ""),
                        line=data.get("line", 0),
                    ).to_dict()
                )
        data = {
            "root": str(self._root),
            "nodes": [n.to_dict() for n in self._nodes.values()],
            "edges": edges,
        }
        result = json.dumps(data, indent=2)
        if path:
            Path(path).write_text(result)
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "CodeGraph":
        """Load a cached graph from JSON."""
        raw = json.loads(Path(path).read_text())
        graph = cls(root=raw["root"], visitors=[])
        for node_data in raw["nodes"]:
            node = Node.from_dict(node_data)
            graph._nodes[node.id] = node
            graph._file_index.setdefault(node.file, []).append(node.id)
            graph._name_index.setdefault(node.name, []).append(node.id)
        for edge_data in raw["edges"]:
            edge = Edge.from_dict(edge_data)
            graph._add_edge(edge.source, edge.target, kind=edge.kind, line=edge.line)
        graph._built = True
        logger.info(
            "Code graph loaded from cache",
            nodes=len(graph._nodes),
            edges=graph._edge_count(),
        )
        return graph
