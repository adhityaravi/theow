"""Code graph for consumer opt-in registration.

Usage:
    from theow.codegraph import CodeGraph

    graph = CodeGraph(root="./src")
    engine = Theow(theow_dir=".theow", llm="anthropic/claude-sonnet-4-20250514")
    engine.tool()(graph.search_code)
"""

from theow._codegraph import CodeGraph

__all__ = ["CodeGraph"]
