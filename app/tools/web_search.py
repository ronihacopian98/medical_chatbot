from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from app.config import get_settings


def search_web(query: str, max_results: int = 3) -> list[Document]:
    """Search the web using Tavily and return results as Documents.

    Tavily is a search engine API designed for AI apps.
    It returns clean, structured results (not raw HTML).

    We convert results to LangChain Documents so they work
    seamlessly with the rest of our pipeline.
    """
    settings = get_settings()

    if not settings.tavily_api_key:
        return []

    tool = TavilySearch(
        api_key=settings.tavily_api_key,
        max_results=max_results,
    )

    results = tool.invoke({"query": query})

    # Convert Tavily results to LangChain Documents
    documents: list[Document] = []
    if isinstance(results, list):
        for result in results:
            documents.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "source": result.get("url", "web"),
                        "title": result.get("title", ""),
                    },
                )
            )
    return documents
