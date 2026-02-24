from fastapi import APIRouter

router = APIRouter()


@router.post("/chat")
async def chat() -> dict[str, str]:
    """Chat endpoint — will be implemented in Phase 4."""
    return {"message": "Chat endpoint coming soon"}
