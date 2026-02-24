from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.config import get_settings
from app.rag.embeddings import get_embedding_model


def get_qdrant_client() -> QdrantClient:
    """Raw Qdrant client — used for admin operations like creating collections."""
    settings = get_settings()
    return QdrantClient(url=settings.qdrant_url)


def get_vector_store() -> QdrantVectorStore:
    """LangChain-wrapped Qdrant vector store.

    This is the main interface for adding and searching documents.
    LangChain wraps Qdrant so we can use it with the rest of
    the LangChain ecosystem (retrievers, chains, etc.)
    """
    settings = get_settings()
    return QdrantVectorStore.from_existing_collection(
        collection_name=settings.qdrant_collection_name,
        embedding=get_embedding_model(),
        url=settings.qdrant_url,
    )


def ensure_collection_exists() -> None:
    """Create the Qdrant collection if it doesn't exist yet.

    A 'collection' in Qdrant is like a 'table' in SQL.
    We need to tell it the vector size (768 for nomic-embed-text)
    and the distance metric (Cosine = angle between vectors).
    """
    from qdrant_client.models import Distance, VectorParams

    settings = get_settings()
    client = get_qdrant_client()

    try:
        client.get_collection(settings.qdrant_collection_name)
    except (UnexpectedResponse, Exception):
        client.create_collection(
            collection_name=settings.qdrant_collection_name,
            vectors_config=VectorParams(
                size=768,  # nomic-embed-text output dimension
                distance=Distance.COSINE,
            ),
        )


def add_documents(documents: list[Document]) -> None:
    """Embed documents and store them in Qdrant.

    This does two things in one call:
    1. Sends each document's text to Ollama → gets vector back
    2. Stores the vector + original text + metadata in Qdrant
    """
    settings = get_settings()
    QdrantVectorStore.from_documents(
        documents=documents,
        embedding=get_embedding_model(),
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
    )
