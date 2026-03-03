from langgraph.graph import END, StateGraph

from app.core.nodes import (
    add_disclaimer,
    check_hallucination,
    check_safety,
    generate_response,
    grade_documents,
    rerank_documents,
    retrieve_documents,
    web_search,
)
from app.core.state import GraphState


def _needs_web_search(state: GraphState) -> str:
    """After grading: does the answer need web search too?"""
    if state["needs_web_search"]:
        return "web_search"
    return "generate"


def build_graph() -> StateGraph:
    """Build and compile the full production LangGraph workflow.

    Flow:
    1. Search books (Qdrant)
    2. Rerank results (keep only relevant chunks)
    3. Grade (are results good enough?)
    4. Maybe search web (Tavily)
    5. Generate answer (LLM)
    6. Check hallucinations (is answer grounded in sources?)
    7. Check safety (no dangerous medical advice?)
    8. Add disclaimer + handle bad answers
    """
    graph = StateGraph(GraphState)

    # Add all nodes
    graph.add_node("retrieve_documents", retrieve_documents)
    graph.add_node("rerank_documents", rerank_documents)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate_response", generate_response)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("check_safety", check_safety)
    graph.add_node("add_disclaimer", add_disclaimer)

    # Wire it all together
    graph.set_entry_point("retrieve_documents")
    graph.add_edge("retrieve_documents", "rerank_documents")
    graph.add_edge("rerank_documents", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        _needs_web_search,
        {
            "web_search": "web_search",
            "generate": "generate_response",
        },
    )

    graph.add_edge("web_search", "generate_response")
    graph.add_edge("generate_response", "check_hallucination")
    graph.add_edge("check_hallucination", "check_safety")
    graph.add_edge("check_safety", "add_disclaimer")
    graph.add_edge("add_disclaimer", END)

    return graph.compile()
