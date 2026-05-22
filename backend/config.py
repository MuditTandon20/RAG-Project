# backend/config.py
# ─────────────────────────────────────────────────────
# PURPOSE: Central configuration file.
# Instead of hardcoding API keys anywhere in the code,
# we load them from a .env file. This is standard
# production practice — keys never touch your codebase.
# ─────────────────────────────────────────────────────

import os                          
# os: Python's built-in module to interact with the 
# operating system — here we use it to read env vars

from dotenv import load_dotenv     
# load_dotenv: reads the .env file and injects its
# key=value pairs into os.environ, making them available
# anywhere in your app via os.getenv()

load_dotenv()
# Executes the .env loading. Must be called BEFORE
# any os.getenv() calls. Best practice: call it once
# at the top of config.py, not in every file.

# ── LLM Settings ──────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Reads OPENAI_API_KEY from environment.
# Returns None if not set — we handle that gracefully below.

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
# Second argument is the DEFAULT value.
# gpt-4o-mini is chosen because:
#   - 8x cheaper than gpt-4o
#   - Fast enough for RAG (the heavy lifting is retrieval)
#   - Good instruction following for grounded responses

# ── Embedding Settings ─────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    "text-embedding-3-small"
)
# text-embedding-3-small: OpenAI's latest embedding model
# - 1536-dimensional vectors
# - 5x cheaper than ada-002 (old model)
# - Better performance on retrieval benchmarks

# ── Chunking Settings ──────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
# How many characters per chunk.
# 500 is a good default — we'll explain the trade-offs
# in depth in Phase 3.

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
# How many characters overlap between adjacent chunks.
# Prevents losing context at chunk boundaries.

# ── Retrieval Settings ─────────────────────────────────
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
# How many chunks to retrieve per query.
# 4 is a tested sweet spot — enough context, not too
# much noise for the LLM to handle.

# ── File Storage ───────────────────────────────────────
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "data/uploads")
# Where uploaded documents are stored.

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore_index")
# Where FAISS saves its index files to disk.

# ── Validation ─────────────────────────────────────────
if not OPENAI_API_KEY:
    # Fail loudly at startup, not silently mid-request.
    # This is called "fail-fast" — a core production principle.
    raise ValueError(
        "❌ OPENAI_API_KEY not found. "
        "Add it to your .env file: OPENAI_API_KEY=sk-..."
    )
