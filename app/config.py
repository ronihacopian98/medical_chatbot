from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    pydantic-settings automatically reads from:
    1. Environment variables (highest priority)
    2. .env file (fallback)

    This means in production you set real env vars,
    and locally you just use a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "Medical Chatbot API"
    app_version: str = "0.1.0"
    debug: bool = False

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    ollama_embedding_model: str = "nomic-embed-text"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "medical_docs"

    # --- Tavily ---
    tavily_api_key: str = ""

    # --- RAG ---
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance — created once, reused everywhere.

    Why @lru_cache? We don't want to re-read .env on every request.
    The settings are immutable at runtime, so caching is safe.
    """
    return Settings()
