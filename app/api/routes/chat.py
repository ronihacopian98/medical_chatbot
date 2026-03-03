import uuid

from fastapi import APIRouter

from app.api.dependencies import get_graph
from app.models.schemas import ChatRequest, ChatResponse, Source

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Chat endpoint.

    What happens:
    1. User sends a question (ChatRequest)
    2. We run the LangGraph workflow
    3. LangGraph decides: books, web, or both
    4. Searches for context
    5. LLM writes the answer
    6. We send it back (ChatResponse)
    """
    graph = get_graph()

    # Run the full LangGraph workflow
    result = graph.invoke({
        "query": request.query,
        "documents": [],
        "needs_web_search": False,
        "web_results": [],
        "answer": "",
        "is_grounded": True,
        "is_safe": True,
        "sources": [],
    })

    # Convert raw sources to our Source schema
    sources = [
        Source(
            content=s.get("content", ""),
            source=s.get("source", "unknown"),
        )
        for s in result.get("sources", [])
    ]

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        conversation_id=request.conversation_id or str(uuid.uuid4()),
    )
