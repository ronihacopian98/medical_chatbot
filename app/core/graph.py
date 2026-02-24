from langgraph.graph import END, StateGraph

from app.core.nodes import (
    generate_response,
    retrieve_documents,
    route_query,
    web_search,
)
from app.core.state import GraphState


def _decide_route(state: GraphState) -> str:
    """Conditional edge: based on the route, decide which nodes to run next.

    This is the "traffic controller" of the graph.
    After the router decides the strategy, this function
    tells LangGraph which path to take.
    """
    route = state["route"]
    if route == "rag_only":
        return "retrieve"
    elif route == "web_only":
        return "web_search"
    else:
        return "both"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph workflow.

    The graph looks like this:

    [route_query]
         │
         ├── "retrieve"  → [retrieve_documents] → [generate_response] → END
         │
         ├── "web_search" → [web_search] → [generate_response] → END
         │
         └── "both" → [retrieve_documents] → [web_search] → [generate_response] → END

    Returns a compiled graph you can call with .invoke()
    """
    # Create the graph with our state schema
    graph = StateGraph(GraphState)

    # Add nodes (the workers)
    graph.add_node("route_query", route_query)
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_response", generate_response)

    # Set the entry point — always start with routing
    graph.set_entry_point("route_query")

    # Add conditional edges — the "if/else" logic
    graph.add_conditional_edges(
        "route_query",  # After this node...
        _decide_route,  # ...run this function to decide where to go
        {
            # If it returns "retrieve", go to retrieve_documents
            "retrieve": "retrieve_documents",
            # If it returns "web_search", go to web_search
            "web_search": "web_search",
            # If it returns "both", go to retrieve first
            "both": "retrieve_documents",
        },
    )

    # After retrieve_documents:
    # - If route was "both", also do web search
    # - Otherwise, go straight to generate
    graph.add_conditional_edges(
        "retrieve_documents",
        lambda state: "web_search" if state["route"] == "rag_and_web" else "generate",
        {
            "web_search": "web_search",
            "generate": "generate_response",
        },
    )

    # After web_search, always generate
    graph.add_edge("web_search", "generate_response")

    # After generate, we're done
    graph.add_edge("generate_response", END)

    return graph.compile()
