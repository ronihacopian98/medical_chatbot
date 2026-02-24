from typing import TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """The shared notepad that all graph nodes read/write.

    Think of it like a form being passed around a team:
    - Person 1 writes the question
    - Person 2 fills in "should we search books or web?"
    - Person 3 finds relevant documents
    - Person 4 writes the final answer

    Each field is one section of the form.
    """

    # The user's original question
    query: str

    # "rag_only", "web_only", or "rag_and_web"
    # Decided by the router node
    route: str

    # Documents retrieved from Qdrant (our medical books)
    documents: list[Document]

    # Results from Tavily web search
    web_results: list[Document]

    # The final generated answer
    answer: str

    # Source citations to show the user
    sources: list[dict[str, str]]
