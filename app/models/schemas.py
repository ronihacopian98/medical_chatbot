from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """What the client sends to POST /api/v1/chat."""

    query: str = Field(
        ...,  # ... means required
        min_length=1,
        max_length=2000,
        description="The medical question to ask",
        examples=["What are the symptoms of diabetes?"],
    )
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation ID for multi-turn chat",
    )


class Source(BaseModel):
    """A single source document used to generate the answer."""

    content: str = Field(description="The relevant text chunk")
    source: str = Field(description="Where this came from (PDF name or URL)")
    page: int | None = Field(default=None, description="Page number if from PDF")


class ChatResponse(BaseModel):
    """What the API returns from POST /api/v1/chat."""

    answer: str = Field(description="The generated answer")
    sources: list[Source] = Field(
        default_factory=list,
        description="Sources used to generate the answer",
    )
    conversation_id: str = Field(description="Conversation ID for follow-ups")


class IngestRequest(BaseModel):
    """Metadata for a PDF ingestion request."""

    collection_name: str | None = Field(
        default=None,
        description="Override the default Qdrant collection name",
    )


class IngestResponse(BaseModel):
    """Result of a PDF ingestion."""

    filename: str
    chunks_created: int
    message: str = "Ingestion successful"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
