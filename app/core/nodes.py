from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.core.prompts import (
    COMBINED_PROMPT,
    MEDICAL_SYSTEM_PROMPT,
    RAG_PROMPT,
    ROUTER_PROMPT,
    WEB_PROMPT,
)
from app.core.state import GraphState
from app.rag.retriever import get_retriever
from app.tools.web_search import search_web


def get_llm() -> ChatOllama:
    """Create the LLM instance (Ollama running locally)."""
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0,  # 0 = deterministic, no randomness
    )


# --- Node 1: Route the query ---

def route_query(state: GraphState) -> GraphState:
    """Decide: should we search books, web, or both?

    The LLM reads the question and picks a strategy.
    Example:
      "What is diabetes?" → rag_only (textbook knowledge)
      "Latest COVID vaccine news" → web_only (current info)
      "Diabetes treatments in 2025" → rag_and_web (both)
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | llm

    result = chain.invoke({"query": state["query"]})
    route = result.content.strip().lower()

    # Default to rag_only if LLM gives unexpected output
    if route not in ("rag_only", "web_only", "rag_and_web"):
        route = "rag_only"

    return {**state, "route": route}


# --- Node 2: Retrieve from vector store ---

def retrieve_documents(state: GraphState) -> GraphState:
    """Search Qdrant for relevant medical book chunks.

    Takes the user's question, embeds it, finds the
    closest chunks in our vector database.
    """
    retriever = get_retriever()
    documents = retriever.invoke(state["query"])
    return {**state, "documents": documents}


# --- Node 3: Web search ---

def web_search(state: GraphState) -> GraphState:
    """Search the web via Tavily for current medical info."""
    results = search_web(state["query"])
    return {**state, "web_results": results}


# --- Node 4: Generate the final answer ---

def generate_response(state: GraphState) -> GraphState:
    """Take all the context and generate the final answer.

    Picks the right prompt based on the route:
    - rag_only → uses only book context
    - web_only → uses only web results
    - rag_and_web → uses both
    """
    llm = get_llm()
    route = state["route"]
    query = state["query"]

    if route == "rag_only":
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            ("human", RAG_PROMPT),
        ])
        result = prompt | llm
        response = result.invoke({"context": context, "query": query})

    elif route == "web_only":
        context = "\n\n".join(doc.page_content for doc in state["web_results"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            ("human", WEB_PROMPT),
        ])
        result = prompt | llm
        response = result.invoke({"context": context, "query": query})

    else:  # rag_and_web
        rag_context = "\n\n".join(doc.page_content for doc in state["documents"])
        web_context = "\n\n".join(doc.page_content for doc in state["web_results"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            ("human", COMBINED_PROMPT),
        ])
        result = prompt | llm
        response = result.invoke({
            "rag_context": rag_context,
            "web_context": web_context,
            "query": query,
        })

    # Collect source citations
    sources: list[dict[str, str]] = []
    for doc in state.get("documents", []):
        sources.append({
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "unknown"),
        })
    for doc in state.get("web_results", []):
        sources.append({
            "content": doc.page_content[:200],
            "source": doc.metadata.get("source", "web"),
        })

    return {**state, "answer": response.content, "sources": sources}
