# backend/api/routes.py
# ─────────────────────────────────────────────────────
# PURPOSE: Define all FastAPI route handlers.
# Each route is a function that:
#   1. Receives an HTTP request
#   2. Validates input (Pydantic does this automatically)
#   3. Calls the RAG pipeline
#   4. Returns a structured HTTP response
#
# We separate routes from main.py to keep the
# entry point clean. Routes can be imported and
# mounted as a router — good for scaling to many
# endpoint groups.
# ─────────────────────────────────────────────────────

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    # APIRouter: groups related endpoints together.
    # Like a "section" of your API.
    # Keeps main.py clean — just mounts routers.

    File,
    # File: FastAPI's marker for file upload parameters.
    # Tells FastAPI to parse multipart form data.

    UploadFile,
    # UploadFile: The actual uploaded file object.
    # Has properties: .filename, .content_type, .file
    # .file is a file-like object you can read from.

    HTTPException,
    # HTTPException: raise this to return HTTP error
    # responses (404, 400, 500) with JSON body.
    # FastAPI catches it and formats the response.

    Depends,
    # Depends: FastAPI's dependency injection system.
    # We use it to share the RAG pipeline instance
    # across all route handlers without global variables.

    status,
    # status: HTTP status code constants.
    # status.HTTP_200_OK, status.HTTP_400_BAD_REQUEST etc.
    # More readable than magic numbers like 400, 404.
)

from fastapi.responses import JSONResponse
# JSONResponse: explicitly return JSON with custom
# status codes. FastAPI auto-converts dicts to JSON,
# but JSONResponse gives more control.

from pydantic import BaseModel, Field
# BaseModel: base class for Pydantic data models.
# Define it once → automatic validation, serialization,
# and OpenAPI schema generation (for /docs).
#
# Field: adds metadata to Pydantic fields —
# descriptions, examples, constraints.
# These appear in the auto-generated /docs UI.

from backend.pipeline import RAGPipeline
from backend.config import UPLOAD_DIR

logger = logging.getLogger(__name__)

router = APIRouter()
# Creates a router — a mini FastAPI app that groups
# related routes. Mounted in main.py with a prefix.


# ══════════════════════════════════════════════════════
# Pydantic Models — Request + Response Schemas
# ══════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    """
    Schema for POST /query request body.

    Pydantic automatically:
      - Parses incoming JSON into this model
      - Validates field types and constraints
      - Returns 422 Unprocessable Entity if invalid
      - Generates OpenAPI schema for /docs
    """

    question: str = Field(
        ...,
        # ... means REQUIRED — no default value.

        min_length=3,
        # Reject questions shorter than 3 characters.
        # Prevents accidental empty or nonsense queries.

        max_length=1000,
        # Prevent extremely long inputs that could be
        # prompt injection attempts or accidental pastes.

        description="The question to ask about your documents.",
        example="What is the battery range of the Model X?",
        # description and example appear in /docs UI.
        # Makes your API self-documenting.
    )

    strategy: Optional[str] = Field(
        default="mmr",
        description="Retrieval strategy: 'mmr', 'similarity', or 'threshold'",
        example="mmr",
    )

    top_k: Optional[int] = Field(
        default=4,
        ge=1,
        # ge = "greater than or equal to"
        # Rejects top_k=0 (nonsensical) automatically.
        le=10,
        # le = "less than or equal to"
        # Cap at 10 — more chunks = more cost + noise.
        description="Number of chunks to retrieve (1-10)",
    )


class QueryResponse(BaseModel):
    """
    Schema for POST /query response body.
    Defines exactly what clients will receive.
    """

    success: bool
    question: str
    answer: str
    sources: list
    retrieval_strategy: str
    chunks_retrieved: int
    top_retrieval_score: Optional[float]
    retrieval_time: float
    generation_time: float
    total_time: float
    is_uncertain: bool
    token_estimate: int
    model_used: str
    error_message: Optional[str] = None


class UploadResponse(BaseModel):
    """Schema for POST /upload response."""

    success: bool
    file_name: str
    chunks_created: int
    total_vectors: int
    ingestion_time: float
    pages_loaded: int
    message: str
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Schema for GET /health response."""

    status: str
    index_stats: dict
    retrieval_strategy: str
    llm_model: str
    embedding_model: dict


# ══════════════════════════════════════════════════════
# Dependency Injection — Pipeline Singleton
# ══════════════════════════════════════════════════════

# Module-level pipeline instance — created once at startup.
# All requests share this single instance.
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """
    FastAPI dependency that provides the RAG pipeline.

    HOW DEPENDENCY INJECTION WORKS IN FASTAPI:
    Instead of creating a new RAGPipeline() for every
    request (expensive!), we create it ONCE and inject
    the same instance into every route handler.

    Usage in route:
      @router.post("/query")
      def query(request: QueryRequest,
                pipeline: RAGPipeline = Depends(get_pipeline)):
          # pipeline is the shared singleton here

    FastAPI calls get_pipeline() for each request,
    but we return the same _pipeline object every time.
    This is the "singleton via dependency injection" pattern.
    """

    global _pipeline
    if _pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Pipeline not initialized. "
                   "Server may still be starting up.",
        )
    return _pipeline


def initialize_pipeline(**kwargs) -> RAGPipeline:
    """
    Create the pipeline singleton.
    Called ONCE from main.py at server startup.
    Never called during request handling.
    """

    global _pipeline
    logger.info("Creating RAG Pipeline singleton...")
    _pipeline = RAGPipeline(**kwargs)
    logger.info("✅ Pipeline singleton ready.")
    return _pipeline


# ══════════════════════════════════════════════════════
# Route: GET /
# ══════════════════════════════════════════════════════

@router.get(
    "/",
    summary="Root",
    description="Confirms the API is running.",
)
def root():
    """
    Simple liveness check.
    Return immediately — no pipeline needed.
    Used by: load balancers, uptime monitors.
    """

    return {
        "message": "RAG System API is running. "
                   "Visit /docs for interactive documentation.",
        "version": "1.0.0",
        "endpoints": ["/health", "/upload", "/query",
                      "/documents", "/reset"],
    }


# ══════════════════════════════════════════════════════
# Route: GET /health
# ══════════════════════════════════════════════════════

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    description="Returns pipeline status, index stats, and model info.",
)
def health_check(
    pipeline: RAGPipeline = Depends(get_pipeline),
    # Depends(get_pipeline): FastAPI calls get_pipeline()
    # and injects the returned object as `pipeline`.
    # If get_pipeline() raises HTTPException, FastAPI
    # returns that error response immediately —
    # the route function never runs.
):
    """
    Readiness check — more detailed than root.
    Returns: whether index has documents, which models
    are loaded, how many vectors are indexed.

    Used by: frontend to show system status,
             DevOps for monitoring,
             CI/CD pipelines before running tests.
    """

    status_data = pipeline.get_system_status()

    return HealthResponse(
        status=status_data["status"],
        index_stats=status_data["index"],
        retrieval_strategy=status_data["retrieval_strategy"],
        llm_model=status_data["llm_model"],
        embedding_model=status_data["embedding_model"],
    )


# ══════════════════════════════════════════════════════
# Route: POST /upload
# ══════════════════════════════════════════════════════

@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    # 201 Created is more semantically correct than
    # 200 OK for resource creation operations.
    summary="Upload and Index a Document",
    description="Upload a PDF or TXT file to be indexed for querying.",
)
async def upload_document(
    file: UploadFile = File(...),
    # UploadFile: FastAPI reads the multipart form data
    # and gives us this object.
    # ... means required — no file = 422 error.

    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Upload a document file and ingest it into the RAG system.

    Process:
      1. Validate file type
      2. Save file to disk (data/uploads/)
      3. Run ingestion pipeline (load → chunk → embed → index)
      4. Return indexing statistics

    Accepts: .pdf, .txt files
    Max size: handled by server config (default: no limit in dev)
    """

    # ── Validate file type ────────────────────────────
    allowed_extensions = {".pdf", ".txt"}
    file_ext = Path(file.filename).suffix.lower()
    # Extract extension from original filename.
    # .lower() handles "File.PDF" edge case.

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type: '{file_ext}'. "
                f"Allowed: {', '.join(allowed_extensions)}"
            ),
        )
        # HTTP 400: Bad Request — client sent invalid data.
        # Raising HTTPException short-circuits the function.
        # FastAPI catches it and returns:
        # {"detail": "Unsupported file type: ..."}

    # ── Save uploaded file to disk ────────────────────
    upload_dir = Path(UPLOAD_DIR)
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Ensure upload directory exists.

    save_path = upload_dir / file.filename
    # Path division operator creates: data/uploads/file.pdf
    # Cleaner than: os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(save_path, "wb") as f:
            # "wb" = write binary mode.
            # Files (PDFs, etc.) are binary data — not text.
            # Using "w" (text mode) would corrupt binary files.

            content = await file.read()
            # await: because file.read() is async in FastAPI.
            # The `async def` route handler allows us to await.
            # Without await, we'd block the entire server
            # while reading the file.

            f.write(content)
            # Write raw bytes to disk.

    except Exception as e:
        logger.error(f"Failed to save file: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}",
        )
        # HTTP 500: server-side error — not the client's fault.

    logger.info(
        f"📁 File saved: {save_path} "
        f"({len(content)/1024:.1f} KB)"
    )

    # ── Run ingestion pipeline ────────────────────────
    result = pipeline.ingest_document(str(save_path))
    # Calls our Phase 8 orchestrator.
    # This is the heavy operation: load → chunk → embed → index.
    # In production with large files, this should be
    # offloaded to a background task (Celery, FastAPI
    # BackgroundTasks) to avoid blocking the HTTP response.

    if not result.success:
        # Ingestion failed after saving file.
        # Clean up the saved file to avoid orphan files.
        if save_path.exists():
            os.remove(save_path)

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.error_message,
        )
        # HTTP 422: the file was received but couldn't be
        # processed (e.g., image-only PDF with no text).

    return UploadResponse(
        success=True,
        file_name=result.file_name,
        chunks_created=result.chunks_created,
        total_vectors=result.total_vectors,
        ingestion_time=result.ingestion_time,
        pages_loaded=result.pages_loaded,
        message=(
            f"Successfully indexed '{result.file_name}'. "
            f"Created {result.chunks_created} chunks from "
            f"{result.pages_loaded} pages. "
            f"Ready for queries."
        ),
    )


# ══════════════════════════════════════════════════════
# Route: POST /query
# ══════════════════════════════════════════════════════

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Query the RAG System",
    description="Ask a question and get an answer grounded in your documents.",
)
async def query_documents(
    request: QueryRequest,
    # Pydantic parses + validates the JSON body automatically.
    # If "question" is missing or too short → 422 error.
    # No manual validation code needed.

    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Submit a natural language question.
    The system retrieves relevant document chunks and
    generates a grounded answer using the LLM.

    Returns the answer, source attribution, and
    performance metrics (retrieval time, generation time).
    """

    # ── Validate retrieval strategy ───────────────────
    valid_strategies = {"mmr", "similarity", "threshold"}
    if request.strategy not in valid_strategies:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid strategy '{request.strategy}'. "
                f"Valid options: {valid_strategies}"
            ),
        )

    # ── Check if index exists ─────────────────────────
    index_stats = pipeline.vectorstore.get_index_stats()
    if index_stats["total_vectors"] == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No documents indexed yet. "
                "Please upload a document first via POST /upload"
            ),
        )
    # Fail fast with a clear message instead of returning
    # "I don't have enough information" — that's confusing
    # when the real problem is no documents are loaded.

    # ── Run query pipeline ────────────────────────────
    result = pipeline.query(
        question=request.question,
        strategy=request.strategy,
        top_k=request.top_k,
    )
    # Phase 8 orchestrator handles everything:
    # retrieve → format → generate → return QueryResult

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.error_message,
        )

    return QueryResponse(**result.to_dict())
    # **result.to_dict() unpacks the dict as keyword args.
    # Equivalent to: QueryResponse(success=True, answer="...", ...)
    # Cleaner than listing every field manually.


# ══════════════════════════════════════════════════════
# Route: GET /documents
# ══════════════════════════════════════════════════════

@router.get(
    "/documents",
    summary="List Indexed Documents",
    description="Returns all documents currently in the upload folder.",
)
def list_documents():
    """
    List all files in the upload directory.

    Simple utility endpoint — the frontend uses this
    to show users what documents are currently indexed.
    """

    upload_dir = Path(UPLOAD_DIR)

    if not upload_dir.exists():
        return {"documents": [], "total": 0}

    # Collect all supported files in the upload directory
    documents = []
    for file_path in upload_dir.iterdir():
        # iterdir(): yields Path objects for every item
        # in the directory (files and subdirectories).

        if file_path.is_file() and file_path.suffix.lower() \
                in {".pdf", ".txt"}:
            # is_file(): skip subdirectories.
            # suffix check: skip hidden files like .DS_Store

            stat = file_path.stat()
            # stat(): filesystem metadata for the file.
            # .st_size: file size in bytes.
            # .st_mtime: last modified time (Unix timestamp).

            documents.append({
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "size_kb":   round(stat.st_size / 1024, 2),
                "modified":  stat.st_mtime,
            })

    # Sort by modification time, newest first
    documents.sort(key=lambda x: x["modified"], reverse=True)
    # lambda: anonymous function.
    # key=lambda x: x["modified"] means:
    # sort by the "modified" field of each dict.
    # reverse=True: descending order (newest first).

    return {
        "documents": documents,
        "total":     len(documents),
    }


# ══════════════════════════════════════════════════════
# Route: DELETE /reset
# ══════════════════════════════════════════════════════

@router.delete(
    "/reset",
    summary="Reset Index",
    description="⚠️ DESTRUCTIVE: Clears the entire FAISS index.",
    status_code=status.HTTP_200_OK,
)
def reset_index(
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Delete all vectors from the FAISS index.
    Documents on disk are NOT deleted — only the index.
    You can re-ingest documents after resetting.

    ⚠️ This operation cannot be undone.
    In production: require an admin API key header here.
    """

    result = pipeline.reset_index()
    logger.warning("🗑️  Index reset via API call.")

    return {
        "status":  "success",
        "message": "Index cleared. Re-upload documents to rebuild.",
        "detail":  result,
    }