from langchain_core.vectorstores import VectorStoreRetriever

from app.config import get_settings
from app.rag.vectorstore import get_vector_store


def get_retriever() -> VectorStoreRetriever:
    """Create a retriever that searches Qdrant for relevant chunks.

    How it works:
    1. User asks: "What are symptoms of diabetes?"
    2. Retriever embeds that question → vector
    3. Finds the top_k closest vectors in Qdrant
    4. Returns the original text chunks

    top_k=5 means "return the 5 most relevant chunks".
    Too few → might miss important context.
    Too many → floods the LLM with noise.
    5 is a good balance.
    """
    settings = get_settings()
    vector_store = get_vector_store()
    return vector_store.as_retriever(
        search_kwargs={"k": settings.retriever_top_k},
    )
