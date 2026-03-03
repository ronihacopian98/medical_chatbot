import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.config import get_settings
from app.core.prompts import (
    COMBINED_PROMPT,
    GRADER_PROMPT,
    HALLUCINATION_PROMPT,
    MEDICAL_DISCLAIMER,
    MEDICAL_SYSTEM_PROMPT,
    RAG_PROMPT,
    RERANK_PROMPT,
    SAFETY_PROMPT,
)
from app.core.state import GraphState
from app.rag.retriever import get_retriever
from app.tools.web_search import search_web

logger = logging.getLogger(__name__)


def get_llm() -> ChatOllama:
    """Create the LLM instance (Ollama running locally)."""
    settings = get_settings()
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=0,
    )


# --- Node 1: Search books ---

def retrieve_documents(state: GraphState) -> GraphState:
    """Search Qdrant for relevant medical book chunks."""
    logger.info("Searching books for: %s", state["query"])
    retriever = get_retriever()
    documents = retriever.invoke(state["query"])
    logger.info("Found %d chunks", len(documents))
    return {**state, "documents": documents}


# --- Node 2: Rerank results ---

def rerank_documents(state: GraphState) -> GraphState:
    """Score each chunk and keep only the most relevant ones.

    Why rerank? Qdrant finds "close" vectors, but close doesn't
    always mean "best answer". The LLM reads each chunk and scores
    it 0-10. We keep only chunks that score 6+.
    """
    if not state["documents"]:
        return state

    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(RERANK_PROMPT)
    chain = prompt | llm

    scored_docs = []
    for doc in state["documents"]:
        result = chain.invoke({
            "query": state["query"],
            "document": doc.page_content,
        })
        try:
            score = int(result.content.strip())
        except ValueError:
            score = 5  # Default if LLM gives bad output

        if score >= 6:
            scored_docs.append(doc)

    logger.info("Reranked: %d → %d chunks", len(state["documents"]), len(scored_docs))
    return {**state, "documents": scored_docs}


# --- Node 3: Grade the results ---

def grade_documents(state: GraphState) -> GraphState:
    """Ask the LLM: are these book results good enough?"""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(GRADER_PROMPT)
    chain = prompt | llm

    context = "\n\n".join(doc.page_content for doc in state["documents"])
    result = chain.invoke({"query": state["query"], "context": context})
    grade = result.content.strip().lower()

    needs_web = "no" in grade or not state["documents"]
    logger.info("Grade result: needs_web_search=%s", needs_web)
    return {**state, "needs_web_search": needs_web}


# --- Node 4: Web search (only if needed) ---

def web_search(state: GraphState) -> GraphState:
    """Search the web via Tavily for current medical info."""
    logger.info("Searching web for: %s", state["query"])
    results = search_web(state["query"])
    logger.info("Found %d web results", len(results))
    return {**state, "web_results": results}


# --- Node 5: Generate the answer ---

def generate_response(state: GraphState) -> GraphState:
    """Take all the context and generate the answer."""
    llm = get_llm()
    query = state["query"]

    has_docs = bool(state.get("documents"))
    has_web = bool(state.get("web_results"))

    if has_docs and has_web:
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
    else:
        context = "\n\n".join(doc.page_content for doc in state["documents"])
        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_SYSTEM_PROMPT),
            ("human", RAG_PROMPT),
        ])
        result = prompt | llm
        response = result.invoke({"context": context, "query": query})

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

    logger.info("Generated response: %d chars", len(response.content))
    return {**state, "answer": response.content, "sources": sources}


# --- Node 6: Check for hallucinations ---

def check_hallucination(state: GraphState) -> GraphState:
    """Verify the answer is actually supported by the context.

    If the LLM made something up that isn't in the sources,
    this catches it.
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(HALLUCINATION_PROMPT)
    chain = prompt | llm

    all_context = "\n\n".join(doc.page_content for doc in state["documents"])
    if state.get("web_results"):
        all_context += "\n\n" + "\n\n".join(
            doc.page_content for doc in state["web_results"]
        )

    result = chain.invoke({"context": all_context, "answer": state["answer"]})
    is_grounded = "yes" in result.content.strip().lower()

    logger.info("Hallucination check: grounded=%s", is_grounded)
    return {**state, "is_grounded": is_grounded}


# --- Node 7: Safety filter ---

def check_safety(state: GraphState) -> GraphState:
    """Check if the answer contains dangerous medical advice."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(SAFETY_PROMPT)
    chain = prompt | llm

    result = chain.invoke({"answer": state["answer"]})
    is_safe = "yes" in result.content.strip().lower()

    logger.info("Safety check: safe=%s", is_safe)
    return {**state, "is_safe": is_safe}


# --- Node 8: Add disclaimer ---

def add_disclaimer(state: GraphState) -> GraphState:
    """Add medical disclaimer and handle unsafe/hallucinated answers."""
    answer = state["answer"]

    if not state.get("is_grounded", True):
        answer = (
            "I found some relevant information but I'm not fully confident "
            "in my answer. Please verify with a healthcare professional.\n\n"
            + answer
        )

    if not state.get("is_safe", True):
        answer = (
            "**Important: The following information is general in nature. "
            "Please consult your doctor before making any medical decisions.**\n\n"
            + answer
        )

    answer += MEDICAL_DISCLAIMER

    logger.info("Final answer prepared: %d chars", len(answer))
    return {**state, "answer": answer}
