"""Visualize theow's code graph with pyvis. Run: uv run --extra codegraph --with pyvis assets/codegraph/visualize.py"""

from pathlib import Path

from pyvis.network import Network

from theow.codegraph import CodeGraph

COLORS = {
    "module": "#bd93f9",
    "class": "#ffb86c",
    "function": "#50fa7b",
}

EDGE_COLORS = {
    "contains": "#6272a4",
    "calls": "#f1fa8c",
    "imports": "#6272a4",
    "inherits": "#ff5555",
}

project_root = Path(__file__).resolve().parent.parent.parent

graph = CodeGraph(root=project_root / "src" / "theow")
graph.build()

# Add a dependency to demonstrate add_root
import structlog

structlog_root = Path(structlog.__file__).parent
graph.add_root(structlog_root)

net = Network(height="100vh", width="100%", directed=True, bgcolor="#282a36", font_color="#f8f8f2")
net.barnes_hut(gravity=-3000, spring_length=150)

for node in graph._nodes.values():
    label = node.name
    title = f"{node.id}\n{node.kind}\n{node.file}:{node.line}"
    if node.signature:
        title += f"\n{node.signature}"
    if node.docstring:
        title += f"\n\n{node.docstring[:200]}"

    size = {"module": 25, "class": 18, "function": 12}.get(node.kind, 10)
    net.add_node(
        node.id,
        label=label,
        title=title,
        color=COLORS.get(node.kind, "#999"),
        size=size,
        group=node.kind,
    )

for source, targets in graph._fwd.items():
    for target, data in targets.items():
        if target not in graph._nodes:
            continue
        kind = data.get("kind", "")
        net.add_edge(
            source,
            target,
            title=kind,
            color=EDGE_COLORS.get(kind, "#ccc"),
            arrows="to",
            width=2 if kind in ("calls", "inherits") else 1,
        )

out = Path(__file__).resolve().parent / "theow_graph.html"
net.save_graph(str(out))
print(f"Graph saved to {out}")
print(f"Nodes: {len(graph._nodes)}, Edges: {graph._edge_count()}")
