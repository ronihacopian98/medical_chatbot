from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """The shared notepad that all graph nodes read/write."""

    # The user's original question
    query: str

    # Documents retrieved from Qdrant (our medical books)
    documents: list[Document]

    # Did the grader say we need web search?
    needs_web_search: bool

    # Results from Tavily web search
    web_results: list[Document]

    # The generated answer (before safety checks)
    answer: str

    # Is the answer grounded in the context? (hallucination check)
    is_grounded: bool

    # Is the answer safe? (no dangerous medical advice)
    is_safe: bool

    # Source citations to show the user
    sources: list[dict[str, str]]
