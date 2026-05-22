# backend/main.py
# ─────────────────────────────────────────────────────
# PURPOSE: FastAPI application entry point.
# Creates the app, configures middleware,
# registers routers, and handles startup/shutdown.
# ─────────────────────────────────────────────────────

import logging
from contextlib import asynccontextmanager
# asynccontextmanager: decorator to create async
# context managers. FastAPI uses this for lifespan
# events (startup + shutdown in one function).

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# CORSMiddleware: Cross-Origin Resource Sharing.
# Browsers block requests from one domain to another
# by default (security feature).
# Our React/Streamlit frontend runs on localhost:8501,
# our FastAPI runs on localhost:8000.
# Without CORS middleware: browser blocks all requests!
# With it: we explicitly allow our frontend's origin.

from backend.api.routes import router, initialize_pipeline

# ── Logging configuration ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    # %(levelname)-8s: pads level name to 8 chars.
    # INFO     → "INFO    "  (aligned columns → readable logs)
    # WARNING  → "WARNING "
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# Application Lifespan — Startup + Shutdown
# ══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown.

    FastAPI's modern lifespan approach replaces the old
    @app.on_event("startup") decorator.

    Everything BEFORE yield → runs at startup.
    Everything AFTER yield  → runs at shutdown.

    WHY initialize here instead of at module level?
    - Module-level init runs at IMPORT time — before
      the server is fully ready, before env vars are
      definitely loaded, before logging is configured.
    - Lifespan runs AFTER all setup — safer and more
      predictable. Errors here show up clearly in logs.
    """

    # ── STARTUP ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("🚀 RAG System API Starting Up")
    logger.info("=" * 60)

    # Initialize the RAG pipeline ONCE.
    # All routes share this instance via get_pipeline().
    initialize_pipeline(
        retrieval_strategy="mmr",
        top_k=4,
        score_threshold=0.5,
        mmr_lambda=0.5,
        llm_temperature=0.0,
        max_tokens=1024,
    )

    logger.info("✅ API ready to serve requests.")
    logger.info("📖 API docs available at: http://localhost:8000/docs")
    logger.info("=" * 60)

    yield
    # Control passes to the running application here.
    # Everything below runs at SHUTDOWN.

    # ── SHUTDOWN ──────────────────────────────────────
    logger.info("🛑 RAG System API Shutting Down...")
    # Clean up resources here if needed:
    # - Close database connections
    # - Flush log buffers
    # - Save any in-memory state
    # For our system: FAISS is file-based, nothing to close.
    logger.info("Goodbye.")


# ══════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════

app = FastAPI(
    title="RAG System API",
    description="""
## Retrieval-Augmented Generation API

Upload documents and ask questions grounded in their content.

### Features
- 📄 Upload PDF and TXT documents
- 🔍 Semantic search with MMR retrieval
- 🤖 GPT-4o-mini powered answers
- 📚 Full source attribution
- 📊 Performance metrics per query

### Workflow
1. Upload a document via **POST /upload**
2. Ask questions via **POST /query**
3. Get answers with citations and confidence scores
    """,
    version="1.0.0",
    lifespan=lifespan,
    # Register our startup/shutdown handler.
    # FastAPI calls lifespan(app) and manages the context.
)


# ══════════════════════════════════════════════════════
# Middleware
# ══════════════════════════════════════════════════════

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",    # React dev server
        "http://localhost:8501",    # Streamlit
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        # In production: replace with your actual domain
        # e.g., "https://your-app.vercel.app"
    ],

    allow_credentials=True,
    # Allow cookies and auth headers cross-origin.

    allow_methods=["*"],
    # Allow all HTTP methods (GET, POST, DELETE, etc.)
    # In production: restrict to only what you use.

    allow_headers=["*"],
    # Allow all headers.
    # In production: restrict to: ["Content-Type",
    #   "Authorization", "X-API-Key"]
)


# ══════════════════════════════════════════════════════
# Register Routes
# ══════════════════════════════════════════════════════

app.include_router(
    router,
    prefix="/api/v1",
    # All routes get prefixed with /api/v1
    # /health  →  /api/v1/health
    # /upload  →  /api/v1/upload
    # /query   →  /api/v1/query
    # Versioning: when you make breaking changes,
    # release /api/v2 without breaking /api/v1 clients.

    tags=["RAG System"],
    # Groups all these routes under "RAG System"
    # in the /docs UI — keeps docs organized.
)


# ══════════════════════════════════════════════════════
# Run the server (for development)
# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    # uvicorn: ASGI server that runs FastAPI.
    # ASGI (Async Server Gateway Interface) is the
    # async evolution of WSGI. Required for async FastAPI.

    uvicorn.run(
        "backend.main:app",
        # "module:variable" — tells uvicorn where to
        # find the FastAPI app object.

        host="0.0.0.0",
        # Listen on all network interfaces.
        # "127.0.0.1" = only localhost (more secure for dev)
        # "0.0.0.0"   = accessible from other machines on LAN
        # Use 0.0.0.0 for Docker, 127.0.0.1 for local dev.

        port=8000,
        # Standard port for FastAPI/uvicorn.

        reload=True,
        # Hot reload: server restarts when code changes.
        # ONLY use in development — causes brief downtime.
        # Remove in production.

        log_level="info",
    )
