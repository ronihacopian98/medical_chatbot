from functools import lru_cache

from langgraph.graph.state import CompiledStateGraph

from app.core.graph import build_graph


@lru_cache
def get_graph() -> CompiledStateGraph:
    """Build the LangGraph workflow once, reuse on every request.

    Same pattern as config.py — @lru_cache means
    "build it the first time, then return the same one forever."
    """
    return build_graph()
