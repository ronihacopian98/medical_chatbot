from functools import lru_cache

from langchain_ollama import OllamaEmbeddings

from app.config import get_settings


@lru_cache
def get_embedding_model() -> OllamaEmbeddings:
    """Get the embedding model (cached — created once, reused).

    OllamaEmbeddings talks to your local Ollama server.
    The model 'nomic-embed-text' converts text → 768-dim vectors.

    Make sure Ollama is running and the model is pulled:
        ollama pull nomic-embed-text
    """
    settings = get_settings()
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )
