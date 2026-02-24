from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF file and return raw pages as LangChain Documents.

    Each page becomes one Document with metadata like:
    - source: the file path
    - page: the page number
    """
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding.

    Why RecursiveCharacterTextSplitter?
    It tries to split on natural boundaries in this order:
    1. Double newlines (paragraphs)
    2. Single newlines
    3. Spaces (words)
    4. Characters (last resort)

    This keeps paragraphs and sentences intact when possible.

    chunk_size=1000  → each chunk is ~1000 characters (~200 words)
    chunk_overlap=200 → 200 chars overlap between chunks
                        so context isn't lost at boundaries
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,  # Track where each chunk starts in the original
    )
    return splitter.split_documents(documents)


def load_and_split_pdf(file_path: str | Path) -> list[Document]:
    """Full pipeline: load PDF → split into chunks.

    This is the main function other code will call.
    Returns a list of Document chunks ready for embedding.
    """
    documents = load_pdf(file_path)
    return split_documents(documents)
