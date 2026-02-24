import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.models.schemas import IngestResponse
from app.rag.loader import load_and_split_pdf
from app.rag.vectorstore import add_documents, ensure_collection_exists

router = APIRouter()


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
async def ingest(file: UploadFile) -> IngestResponse:
    """Upload a PDF and ingest it into the vector store.

    What happens step by step:
    1. Validate it's a PDF
    2. Save to a temp file (PyPDFLoader needs a file path)
    3. Load PDF → split into chunks
    4. Embed chunks → store in Qdrant
    5. Return how many chunks were created
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Save uploaded file to temp location
    # (PyPDFLoader reads from disk, not memory)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Make sure the Qdrant collection exists
        ensure_collection_exists()

        # Load and chunk the PDF
        chunks = load_and_split_pdf(tmp_path)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text content found in PDF",
            )

        # Embed and store in Qdrant
        add_documents(chunks)

        return IngestResponse(
            filename=file.filename,
            chunks_created=len(chunks),
        )
    finally:
        # Always clean up the temp file
        tmp_path.unlink(missing_ok=True)
