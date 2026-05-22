# frontend/app.py
# ─────────────────────────────────────────────────────
# PURPOSE: Complete Streamlit frontend for the RAG system.
# Communicates with our FastAPI backend (Phase 9) via
# HTTP requests. Displays: upload UI, chat interface,
# source attribution, and performance metrics.
#
# Run with: streamlit run frontend/app.py
# ─────────────────────────────────────────────────────

import streamlit as st
# streamlit: The entire UI framework. Every UI element
# is a function call: st.title(), st.button(), etc.
# Streamlit re-runs this ENTIRE script on every user
# interaction. st.session_state persists data between runs.

import requests
# requests: Python's most popular HTTP library.
# We use it to call our FastAPI backend endpoints.
# Alternative: httpx (async), aiohttp (async).
# requests is sync — fine for Streamlit (which is also sync).

import time
import json
from typing import Optional
from pathlib import Path

# ══════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════

# Backend URL — where our FastAPI server is running.
# In development: localhost:8000
# In production: your deployed backend URL
API_BASE_URL = "http://localhost:8000/api/v1"


def extract_api_error(response: requests.Response) -> str:
    """Return the most useful error text from a FastAPI response."""

    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    detail = payload.get("detail")
    if isinstance(detail, str):
        return detail
    if detail:
        return json.dumps(detail)

    message = payload.get("error_message") or payload.get("message")
    if message:
        return str(message)

    return f"HTTP {response.status_code}: {response.reason}"

# Page configuration — MUST be the first Streamlit call.
# Any st.* call before this raises an error.
st.set_page_config(
    page_title="RAG Document Assistant",
    # Tab title in the browser.

    page_icon="🧠",
    # Browser tab icon. Accepts emoji or image path.

    layout="wide",
    # "wide": uses full browser width.
    # "centered": narrower, more readable for text.
    # Wide is better for our sidebar + main panel layout.

    initial_sidebar_state="expanded",
    # Show sidebar open by default.
)


# ══════════════════════════════════════════════════════
# Custom CSS Styling
# ══════════════════════════════════════════════════════

# st.markdown with unsafe_allow_html=True lets us inject
# raw HTML/CSS into the Streamlit page.
# We use this sparingly — only for things Streamlit
# can't do natively (custom colors, spacing, etc.)
st.markdown("""
    <style>
    /* Make the main header visually distinct */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    /* Style for individual source cards in results */
    .source-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.875rem;
    }

    /* Metric boxes for performance numbers */
    .metric-box {
        background: #eff6ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
        font-size: 0.875rem;
    }

    /* Style the user chat bubbles */
    .user-message {
        background: #3b82f6;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }

    /* Style the assistant chat bubbles */
    .assistant-message {
        background: #f1f5f9;
        color: #1e293b;
        padding: 0.75rem 1rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
    }

    /* Uncertain/warning response styling */
    .uncertain-message {
        background: #fef9c3;
        border-left: 4px solid #eab308;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
    }
    </style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
# Session State Initialization
# ══════════════════════════════════════════════════════

# st.session_state is a dict-like object that persists
# across reruns of the script (within the same browser session).
# Without it, every button click would reset all variables.
#
# We initialize defaults here using .setdefault() —
# which sets a value ONLY if the key doesn't already exist.
# This means on the FIRST run, defaults are set.
# On SUBSEQUENT runs (after user interactions), existing
# values are preserved.

st.session_state.setdefault("chat_history", [])
# List of dicts: [{"role": "user"/"assistant", "content": "...",
#                  "sources": [...], "metrics": {...}}, ...]
# Grows as conversation progresses.

st.session_state.setdefault("documents_indexed", [])
# List of document names successfully indexed.
# Used to show "currently indexed" status.

st.session_state.setdefault("system_status", None)
# Cached health check result — avoids hitting /health
# on every single rerun.

st.session_state.setdefault("last_status_check", 0)
# Unix timestamp of last status check.
# We refresh status max once every 30 seconds.


# ══════════════════════════════════════════════════════
# API Helper Functions
# ══════════════════════════════════════════════════════
# These functions are the "API client layer" —
# they isolate all HTTP communication from the UI code.
# If the API URL or request format changes, you only
# update these functions, not the UI components.

def check_system_health() -> Optional[dict]:
    """
    Call GET /health and return the response dict.
    Returns None if the backend is unreachable.

    We cache the result in session_state and refresh
    only every 30 seconds to avoid hammering the API.
    """

    now = time.time()
    cache_age = now - st.session_state.last_status_check

    # Use cached status if it's less than 30 seconds old
    if cache_age < 30 and st.session_state.system_status:
        return st.session_state.system_status

    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=5,
            # timeout=5: give up after 5 seconds.
            # Without timeout, a dead backend causes
            # the UI to hang indefinitely.
        )
        response.raise_for_status()
        # raise_for_status(): raises an exception if
        # status code is 4xx or 5xx.
        # Forces us to handle API errors, not silently
        # ignore them.

        status_data = response.json()
        # Parse JSON response body into a Python dict.

        # Cache the result and timestamp
        st.session_state.system_status = status_data
        st.session_state.last_status_check = now

        return status_data

    except requests.exceptions.ConnectionError:
        # Backend is not running — common during development.
        return None
    except Exception as e:
        # Other errors: timeout, JSON parse failure, etc.
        return None


def upload_document(file) -> dict:
    """
    Call POST /upload with the file and return the result dict.

    Args:
        file: Streamlit UploadedFile object from st.file_uploader()

    Returns:
        API response dict with success, chunks_created, etc.
    """

    try:
        files = {
            "file": (
                file.name,
                # Original filename — sent as Content-Disposition header.

                file.getvalue(),
                # Raw bytes of the file.
                # getvalue() returns bytes from UploadedFile.

                file.type or "application/octet-stream",
                # MIME type. file.type is set by Streamlit
                # based on file extension. Fallback to generic
                # binary type if not detected.
            )
        }
        # This dict format tells requests to send a
        # multipart/form-data request — the same format
        # browsers use when submitting <input type="file">.

        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files,
            timeout=120,
            # 120 seconds: ingestion can be slow for large docs.
            # Embedding 200 chunks takes 10-30 seconds.
            # A generous timeout prevents premature failure.
        )
        if not response.ok:
            return {
                "success": False,
                "error_message": extract_api_error(response),
            }
        return response.json()

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_message": (
                "Upload timed out. The document may be too large. "
                "Try splitting it into smaller files."
            )
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error_message": (
                "Cannot connect to backend. "
                "Is the FastAPI server running on port 8000?"
            )
        }
    except Exception as e:
        return {"success": False, "error_message": str(e)}


def query_rag(
    question: str,
    strategy: str,
    top_k: int,
) -> dict:
    """
    Call POST /query and return the full result dict.

    Args:
        question: User's natural language question
        strategy: Retrieval strategy ("mmr", etc.)
        top_k:    Number of chunks to retrieve
    """

    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                # json= parameter: serializes dict to JSON
                # and sets Content-Type: application/json.
                # FastAPI's QueryRequest Pydantic model
                # parses this automatically.
                "question": question,
                "strategy": strategy,
                "top_k": top_k,
            },
            timeout=60,
            # 60 seconds: LLM generation can be slow.
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error_message": "Query timed out. Try a simpler question.",
            "answer": "⏱️ Request timed out.",
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error_message": "Backend unreachable.",
            "answer": "❌ Cannot connect to the RAG backend.",
        }
    except Exception as e:
        return {
            "success": False,
            "error_message": str(e),
            "answer": f"❌ Error: {str(e)}",
        }


def get_documents() -> list:
    """Call GET /documents and return list of indexed docs."""

    try:
        response = requests.get(
            f"{API_BASE_URL}/documents",
            timeout=5,
        )
        response.raise_for_status()
        return response.json().get("documents", [])
    except Exception:
        return []


# ══════════════════════════════════════════════════════
# UI Components — Reusable Pieces
# ══════════════════════════════════════════════════════
# Breaking the UI into functions is the same principle
# as component-based design in React.
# Each function renders one visual section.

def render_status_badge(status_data: Optional[dict]):
    """
    Render a colored status indicator in the sidebar.
    Green = ready, Yellow = empty, Red = offline.
    """

    if status_data is None:
        st.error("🔴 Backend Offline")
        st.caption("Start the FastAPI server: uvicorn backend.main:app")
        return

    status = status_data.get("status", "unknown")
    vectors = status_data.get("index", {}).get("total_vectors", 0)

    if status == "ready" and vectors > 0:
        st.success(f"🟢 System Ready — {vectors} chunks indexed")
    elif status == "empty" or vectors == 0:
        st.warning("🟡 Online — No documents indexed yet")
    else:
        st.error("🔴 System Error")

    # Show model info compactly
    llm = status_data.get("llm_model", "unknown")
    emb = status_data.get("embedding_model", {}).get(
        "model_name", "unknown"
    )
    st.caption(f"LLM: `{llm}` | Embeddings: `{emb}`")


def render_source_cards(sources: list):
    """
    Render retrieved source chunks as styled cards.
    Each card shows: filename, page, score, and preview.
    """

    if not sources:
        return

    st.markdown("**📚 Sources Retrieved:**")

    for i, source in enumerate(sources, 1):
        # Build a colored score badge based on score value
        score = source.get("score", 0)
        if score >= 0.85:
            score_color = "#22c55e"   # green — highly relevant
        elif score >= 0.65:
            score_color = "#f59e0b"   # amber — moderately relevant
        else:
            score_color = "#ef4444"   # red — low relevance

        st.markdown(f"""
            <div class="source-card">
                <strong>Source {i}:</strong>
                {source.get('file_name', 'unknown')} &nbsp;|&nbsp;
                Page {source.get('page', '?')} &nbsp;|&nbsp;
                <span style="color:{score_color}; font-weight:600;">
                    Score: {score:.3f}
                </span>
                <br>
                <span style="color:#64748b; font-style:italic;">
                    "{source.get('chunk_preview', '')}"
                </span>
            </div>
        """, unsafe_allow_html=True)


def render_metrics_row(metrics: dict):
    """
    Render performance metrics in a horizontal row.
    Shows: retrieval time, generation time, total time,
    chunks retrieved, and token estimate.
    """

    cols = st.columns(5)
    # st.columns(n): creates n equally-spaced columns.
    # Returns a list of column objects.
    # Use `with col:` to render inside each column.

    with cols[0]:
        st.metric(
            "⚡ Retrieval",
            f"{metrics.get('retrieval_time', 0):.2f}s",
            # st.metric(label, value): renders a big
            # number with a label — great for dashboards.
        )
    with cols[1]:
        st.metric(
            "🤖 Generation",
            f"{metrics.get('generation_time', 0):.2f}s",
        )
    with cols[2]:
        st.metric(
            "🕐 Total",
            f"{metrics.get('total_time', 0):.2f}s",
        )
    with cols[3]:
        st.metric(
            "📄 Chunks",
            metrics.get("chunks_retrieved", 0),
        )
    with cols[4]:
        st.metric(
            "🔢 Tokens ~",
            metrics.get("token_estimate", 0),
        )


def render_chat_message(message: dict):
    """
    Render a single chat message with role-based styling.
    User messages appear right-aligned, assistant left-aligned.
    Sources and metrics are shown below assistant messages.
    """

    role = message["role"]
    content = message["content"]

    if role == "user":
        # User message — right-aligned blue bubble
        st.markdown(
            f'<div class="user-message">👤 {content}</div>',
            unsafe_allow_html=True,
        )

    else:
        # Assistant message — check if uncertain
        is_uncertain = message.get("is_uncertain", False)

        if is_uncertain:
            st.markdown(
                f'<div class="uncertain-message">'
                f'⚠️ {content}'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-message">'
                f'🧠 {content}'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Show sources if available
        if message.get("sources"):
            with st.expander(
                f"📚 View {len(message['sources'])} Sources",
                expanded=False,
                # expanded=False: collapsed by default.
                # User can click to expand.
                # Keeps the chat clean.
            ):
                render_source_cards(message["sources"])

        # Show metrics if available
        if message.get("metrics"):
            with st.expander("⚡ Performance Metrics", expanded=False):
                render_metrics_row(message["metrics"])

        st.markdown("---")
        # Horizontal rule between messages for visual separation.


# ══════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════

with st.sidebar:
    # Everything inside `with st.sidebar:` renders
    # in the left sidebar panel.

    st.markdown("## 🧠 RAG Assistant")
    st.markdown("---")

    # ── System Status ─────────────────────────────────
    st.markdown("### 📊 System Status")
    health_data = check_system_health()
    render_status_badge(health_data)

    # Manual refresh button — forces a new /health call
    if st.button("🔄 Refresh Status", use_container_width=True):
        st.session_state.last_status_check = 0
        # Reset timestamp → next check_system_health()
        # call will fetch fresh data instead of using cache.
        st.rerun()
        # st.rerun(): immediately re-runs the script.
        # The refresh will happen at the top of the next run.

    st.markdown("---")

    # ── Document Upload ───────────────────────────────
    st.markdown("### 📄 Upload Document")

    uploaded_file = st.file_uploader(
        label="Choose a PDF or TXT file",
        type=["pdf", "txt"],
        # type: restricts file picker to these extensions.
        # User CAN'T select other file types.
        # Our API validates again server-side (defense in depth).

        help="Upload a document to ask questions about it.",
        # help: small tooltip shown below the uploader.
        # Great for guiding users without cluttering the UI.
    )

    if uploaded_file is not None:
        # File has been selected (but not yet uploaded to API).
        # Show file info as a preview.
        file_size_kb = len(uploaded_file.getvalue()) / 1024
        st.info(
            f"📎 **{uploaded_file.name}**\n\n"
            f"Size: {file_size_kb:.1f} KB | "
            f"Type: {uploaded_file.type}"
        )

        if st.button(
            "🚀 Index Document",
            use_container_width=True,
            type="primary",
            # type="primary": renders as a filled blue button.
            # Default is "secondary" (outlined).
        ):
            # Show a progress spinner while uploading.
            with st.spinner(
                f"Indexing '{uploaded_file.name}'... "
                "This may take 15-30 seconds."
            ):
                # st.spinner: shows a loading animation
                # while the code inside the `with` block runs.
                # Prevents users from thinking the app froze.
                result = upload_document(uploaded_file)

            if result.get("success"):
                st.success(
                    f"✅ Indexed successfully!\n\n"
                    f"Pages: {result.get('pages_loaded')} | "
                    f"Chunks: {result.get('chunks_created')} | "
                    f"Time: {result.get('ingestion_time')}s"
                )
                # Invalidate status cache so it refreshes
                st.session_state.last_status_check = 0

            else:
                st.error(
                    f"❌ Failed to index:\n\n"
                    f"{result.get('error_message', 'Unknown error')}"
                )

    st.markdown("---")

    # ── Retrieval Settings ────────────────────────────
    st.markdown("### ⚙️ Retrieval Settings")

    retrieval_strategy = st.selectbox(
        "Search Strategy",
        options=["mmr", "similarity", "threshold"],
        index=0,
        # index=0: "mmr" is selected by default.
        help=(
            "MMR: Diverse + relevant (recommended)\n"
            "Similarity: Pure cosine similarity\n"
            "Threshold: Filters low-confidence chunks"
        ),
    )

    top_k = st.slider(
        "Chunks to Retrieve",
        min_value=1,
        max_value=10,
        value=4,
        # value: default position of the slider.
        help="More chunks = more context but slower + pricier.",
    )

    st.markdown("---")

    # ── Indexed Documents ─────────────────────────────
    st.markdown("### 📁 Indexed Documents")
    docs = get_documents()

    if docs:
        for doc in docs:
            st.markdown(
                f"• **{doc['file_name']}** "
                f"({doc['size_kb']} KB)"
            )
    else:
        st.caption("No documents indexed yet.")

    st.markdown("---")

    # ── Danger Zone ───────────────────────────────────
    with st.expander("⚠️ Danger Zone"):
        # Wrap destructive action in expander so it's
        # hidden by default — prevents accidental clicks.

        st.warning(
            "Resetting will delete ALL indexed documents "
            "and vectors. This cannot be undone."
        )

        if st.button(
            "🗑️ Reset Index",
            use_container_width=True,
            type="secondary",
        ):
            try:
                response = requests.delete(
                    f"{API_BASE_URL}/reset",
                    timeout=10,
                )
                response.raise_for_status()
                st.session_state.chat_history = []
                st.session_state.last_status_check = 0
                st.success("Index cleared.")
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")


# ══════════════════════════════════════════════════════
# MAIN PANEL
# ══════════════════════════════════════════════════════

st.markdown(
    '<h1 class="main-header">🧠 RAG Document Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "Ask questions about your uploaded documents. "
    "Every answer is grounded in your content — "
    "no hallucinations."
)
st.markdown("---")

# ── Chat History Display ──────────────────────────────
# Render all past messages in order.
# This loop runs on EVERY rerun — so new messages
# appended to session_state.chat_history automatically
# appear on the next rerun.

if not st.session_state.chat_history:
    # Show a helpful empty state if no conversation yet.
    st.markdown("""
        <div style="text-align:center; padding: 3rem 1rem;
                    color: #94a3b8;">
            <h3>💬 No conversation yet</h3>
            <p>Upload a document using the sidebar,
               then ask a question below.</p>
            <p><em>Example: "What is the main topic of
               this document?"</em></p>
        </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.chat_history:
        render_chat_message(message)

# ── Chat Input ────────────────────────────────────────
# st.chat_input: special Streamlit widget that stays
# fixed at the bottom of the screen like a real chat app.
# It returns the text when user presses Enter,
# or None if they haven't submitted yet.

user_input = st.chat_input(
    placeholder="Ask a question about your documents...",
)

if user_input:
    # User submitted a question — process it.

    # ── Check prerequisites ───────────────────────────
    if health_data is None:
        st.error(
            "❌ Cannot connect to the backend. "
            "Please start the FastAPI server first:\n\n"
            "`uvicorn backend.main:app --reload --port 8000`"
        )
        st.stop()
        # st.stop(): halts script execution here.
        # Prevents running query logic when backend is down.

    if health_data.get("index", {}).get("total_vectors", 0) == 0:
        st.warning(
            "⚠️ No documents indexed yet. "
            "Please upload a document using the sidebar first."
        )
        st.stop()

    # ── Add user message to history ───────────────────
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
    })
    # Appending to chat_history adds it to persistent state.
    # On the next rerun (which happens immediately after
    # this block), the loop above will render it.

    # ── Call RAG pipeline and show spinner ────────────
    with st.spinner("🔍 Searching documents and generating answer..."):
        result = query_rag(
            question=user_input,
            strategy=retrieval_strategy,
            # Uses the sidebar selectbox value.
            top_k=top_k,
            # Uses the sidebar slider value.
        )

    # ── Add assistant response to history ────────────
    assistant_message = {
        "role":         "assistant",
        "content":      result.get("answer", "No answer returned."),
        "sources":      result.get("sources", []),
        "is_uncertain": result.get("is_uncertain", False),
        "metrics": {
            "retrieval_time":  result.get("retrieval_time", 0),
            "generation_time": result.get("generation_time", 0),
            "total_time":      result.get("total_time", 0),
            "chunks_retrieved":result.get("chunks_retrieved", 0),
            "token_estimate":  result.get("token_estimate", 0),
        },
    }

    st.session_state.chat_history.append(assistant_message)

    # ── Trigger a rerun to display new messages ───────
    st.rerun()
    # After appending to chat_history, we call st.rerun()
    # to immediately re-execute the script.
    # The chat history loop at the top will now render
    # the new user + assistant messages.
    # Without st.rerun(), the messages would only appear
    # after the NEXT user interaction.
