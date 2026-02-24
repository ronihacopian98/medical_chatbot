from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Runs on startup and shutdown.

    Startup: initialize connections (Qdrant, Ollama, etc.)
    Shutdown: clean up resources.
    We'll add real logic here in later phases.
    """
    # --- Startup ---
    yield
    # --- Shutdown ---


def create_app() -> FastAPI:
    """Application factory pattern.

    Why a factory? It lets us:
    1. Create different app instances for testing vs production
    2. Keep configuration flexible
    3. Avoid circular imports
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-powered medical chatbot with RAG and web search",
        lifespan=lifespan,
    )

    # CORS — allows the Streamlit frontend (different port) to call our API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, lock this to your domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register route modules
    _register_routes(app)

    return app


def _register_routes(app: FastAPI) -> None:
    """Register all API route modules."""
    from app.api.routes import chat, ingest

    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])


app = create_app()


@app.get("/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Health check endpoint — ALB and Docker health checks hit this."""
    return {"status": "healthy"}
