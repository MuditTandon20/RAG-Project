# backend/pipeline.py
# ─────────────────────────────────────────────────────
# PURPOSE: The central orchestrator of the entire
# RAG system. Wires together every component from
# Phases 2-7 into two clean public methods:
#   - ingest_document()  → index a new document
#   - query()            → answer a user question
#
# This is the ONLY file the API layer (Phase 9) talks
# to. It knows nothing about FAISS internals, prompt
# templates, or chunking strategies — that's the
# point. Each layer knows only its neighbors.
# ─────────────────────────────────────────────────────

import logging
import time
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# ── Import every component we built ──────────────────
from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_documents
from backend.embeddings.embedder import EmbeddingManager
from backend.vectorstore.store import VectorStore
from backend.retrieval.retriever import Retriever, RetrievalResult
from backend.generation.chain import LLMChain, GenerationResult
from backend.config import (
    UPLOAD_DIR,
    VECTORSTORE_DIR,
    TOP_K_RESULTS,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════
# Result Containers
# ══════════════════════════════════════════════════════

@dataclass
class IngestionResult:
    """
    Result of indexing a document into the system.
    Returned to the API layer after upload + processing.
    """

    success: bool
    # True if ingestion completed without errors.

    file_name: str
    # Name of the processed file (e.g., "manual.pdf")

    chunks_created: int = 0
    # How many chunks were generated from this document.
    # Useful for understanding document coverage.

    total_vectors: int = 0
    # Total vectors now in the FAISS index (all documents
    # combined, not just this one).

    ingestion_time: float = 0.0
    # Total seconds taken for load + chunk + embed + index.
    # If this is > 30s, consider async processing.

    error_message: Optional[str] = None
    # If success=False, this explains what went wrong.
    # Always populate this — silent failures are the worst.

    pages_loaded: int = 0
    # How many pages were extracted from the document.

    def to_dict(self) -> dict:
        """Serialize to dict for JSON API responses."""
        return {
            "success":        self.success,
            "file_name":      self.file_name,
            "chunks_created": self.chunks_created,
            "total_vectors":  self.total_vectors,
            "ingestion_time": self.ingestion_time,
            "pages_loaded":   self.pages_loaded,
            "error_message":  self.error_message,
        }


@dataclass
class QueryResult:
    """
    Result of a user query through the full RAG pipeline.
    The richest response object in the system —
    carries answer, sources, and all performance metrics.
    """

    success: bool
    question: str
    answer: str = ""

    sources: list = field(default_factory=list)
    # List of dicts: [{file_name, page, chunk_preview}, ...]
    # Tells the user WHERE the answer came from.

    retrieval_strategy: str = "mmr"
    chunks_retrieved: int = 0
    top_retrieval_score: Optional[float] = None

    # ── Performance metrics ───────────────────────────
    retrieval_time: float = 0.0
    # Seconds spent on vector search.

    generation_time: float = 0.0
    # Seconds spent on LLM generation.

    total_time: float = 0.0
    # End-to-end latency (retrieval + generation).

    # ── Quality signals ───────────────────────────────
    is_uncertain: bool = False
    # True if LLM couldn't answer from retrieved context.
    # Signals: either wrong document indexed, or query
    # is out of scope for this document corpus.

    token_estimate: int = 0
    # Approximate tokens used — for cost tracking.

    error_message: Optional[str] = None
    model_used: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for JSON API responses."""
        return {
            "success":              self.success,
            "question":             self.question,
            "answer":               self.answer,
            "sources":              self.sources,
            "retrieval_strategy":   self.retrieval_strategy,
            "chunks_retrieved":     self.chunks_retrieved,
            "top_retrieval_score":  self.top_retrieval_score,
            "retrieval_time":       self.retrieval_time,
            "generation_time":      self.generation_time,
            "total_time":           self.total_time,
            "is_uncertain":         self.is_uncertain,
            "token_estimate":       self.token_estimate,
            "model_used":           self.model_used,
            "error_message":        self.error_message,
        }


# ══════════════════════════════════════════════════════
# The RAG Pipeline Orchestrator
# ══════════════════════════════════════════════════════

class RAGPipeline:
    """
    Central orchestrator for the entire RAG system.

    Initialization (happens once at server startup):
      1. Creates EmbeddingManager (loads model config)
      2. Creates VectorStore (loads saved FAISS index)
      3. Creates Retriever (configures search strategy)
      4. Creates LLMChain (connects to OpenAI)

    After init, two public methods are available:
      - ingest_document(file_path) → IngestionResult
      - query(question)            → QueryResult

    WHY initialize everything in __init__?
    These objects are expensive to create:
      - EmbeddingManager: sets up API connection
      - VectorStore: loads FAISS index from disk (I/O)
      - LLMChain: configures model parameters
    
    By creating them ONCE at startup and reusing,
    each query/ingestion just calls methods on
    already-warm objects. This is the "warm object"
    pattern — critical for API performance.
    """

    def __init__(
        self,
        retrieval_strategy: str = "mmr",
        top_k: int = TOP_K_RESULTS,
        score_threshold: float = 0.5,
        mmr_lambda: float = 0.5,
        llm_temperature: float = 0.0,
        max_tokens: int = 1024,
    ):
        """
        Args:
            retrieval_strategy: "mmr" | "similarity" | "threshold"
            top_k:              Chunks to retrieve per query.
            score_threshold:    Min similarity score (threshold strategy).
            mmr_lambda:         MMR relevance-diversity balance.
            llm_temperature:    LLM randomness. Always 0.0 for RAG.
            max_tokens:         Max LLM response length.
        """

        logger.info("🚀 Initializing RAG Pipeline...")
        init_start = time.time()

        self.retrieval_strategy = retrieval_strategy

        # ── Step 1: Embedding Manager ─────────────────────
        logger.info("  [1/4] Loading embedding model...")
        self.embedding_manager = EmbeddingManager()
        # EmbeddingManager from Phase 4.
        # Shared between VectorStore (for indexing) and
        # Retriever (for query embedding) — same model
        # guaranteed on both sides. Critical for correctness.

        # ── Step 2: Vector Store ──────────────────────────
        logger.info("  [2/4] Loading vector store...")
        self.vectorstore = VectorStore(
            embedding_manager=self.embedding_manager,
            persist_dir=VECTORSTORE_DIR,
        )
        # Try to load existing index from disk.
        # Returns True if found, False if first run.
        index_loaded = self.vectorstore.load_index()

        if index_loaded:
            stats = self.vectorstore.get_index_stats()
            logger.info(
                f"  ✅ Loaded existing index with "
                f"{stats['total_vectors']} vectors."
            )
        else:
            logger.info(
                "  ⚠️  No existing index found. "
                "Upload a document to begin."
            )

        # ── Step 3: Retriever ─────────────────────────────
        logger.info("  [3/4] Configuring retriever...")
        self.retriever = Retriever(
            vectorstore=self.vectorstore,
            top_k=top_k,
            score_threshold=score_threshold,
            mmr_lambda=mmr_lambda,
        )
        # Retriever from Phase 6.
        # Wraps VectorStore with strategy logic.

        # ── Step 4: LLM Chain ─────────────────────────────
        logger.info("  [4/4] Connecting LLM...")
        self.llm_chain = LLMChain(
            temperature=llm_temperature,
            max_tokens=max_tokens,
        )
        # LLMChain from Phase 7.
        # Prompt template + OpenAI connection.

        total_init = time.time() - init_start
        logger.info(
            f"✅ RAG Pipeline ready in {total_init:.2f}s"
        )

    # ══════════════════════════════════════════════════
    # PUBLIC METHOD 1: Ingest a Document
    # ══════════════════════════════════════════════════

    def ingest_document(
        self,
        file_path: str,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ) -> IngestionResult:
        """
        Index a new document into the RAG system.

        Full pipeline:
          file_path
            → load_document()     [Phase 2]
            → chunk_documents()   [Phase 3]
            → vectorstore.add_documents() [Phase 4+5]
            → save_index()        [Phase 5]

        Args:
            file_path:     Path to the document file.
            chunk_size:    Characters per chunk (tunable).
            chunk_overlap: Overlap between chunks (tunable).

        Returns:
            IngestionResult with stats and status.
        """

        file_name = Path(file_path).name
        logger.info(f"📥 Starting ingestion: '{file_name}'")
        start_time = time.time()

        # ── STAGE 1: Load Document ────────────────────────
        try:
            logger.info("  [Stage 1/3] Loading document...")
            stage_start = time.time()

            documents = load_document(file_path)
            # Phase 2: returns List[Document], one per page.

            pages_loaded = len(documents)
            logger.info(
                f"  ✅ Loaded {pages_loaded} pages "
                f"in {time.time()-stage_start:.2f}s"
            )

        except FileNotFoundError as e:
            # File doesn't exist — user error, not system error.
            # Return a clean IngestionResult, don't crash.
            return IngestionResult(
                success=False,
                file_name=file_name,
                error_message=f"File not found: {e}",
            )

        except Exception as e:
            # Unexpected error — log full traceback for debugging.
            logger.error(
                f"  ❌ Load failed: {e}", exc_info=True
            )
            # exc_info=True: appends full stack trace to log.
            # Invaluable when debugging production errors.
            return IngestionResult(
                success=False,
                file_name=file_name,
                error_message=f"Load error: {str(e)}",
            )

        # ── STAGE 2: Chunk Document ───────────────────────
        try:
            logger.info("  [Stage 2/3] Chunking document...")
            stage_start = time.time()

            chunks = chunk_documents(
                documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            # Phase 3: returns List[Document] of small chunks.

            chunks_created = len(chunks)
            logger.info(
                f"  ✅ Created {chunks_created} chunks "
                f"in {time.time()-stage_start:.2f}s"
            )

            if chunks_created == 0:
                # Document loaded but produced no chunks.
                # Possible cause: PDF with only images (no text).
                return IngestionResult(
                    success=False,
                    file_name=file_name,
                    pages_loaded=pages_loaded,
                    error_message=(
                        "No text could be extracted. "
                        "The document may be image-only or scanned."
                    ),
                )

        except Exception as e:
            logger.error(f"  ❌ Chunk failed: {e}", exc_info=True)
            return IngestionResult(
                success=False,
                file_name=file_name,
                pages_loaded=pages_loaded,
                error_message=f"Chunking error: {str(e)}",
            )

        # ── STAGE 3: Embed + Index ────────────────────────
        try:
            logger.info(
                f"  [Stage 3/3] Embedding + indexing "
                f"{chunks_created} chunks..."
            )
            stage_start = time.time()

            self.vectorstore.add_documents(chunks)
            # Phase 4+5 combined:
            # add_documents() internally calls OpenAI embeddings
            # API, adds vectors to FAISS, saves to disk.

            stats = self.vectorstore.get_index_stats()
            total_vectors = stats["total_vectors"]

            logger.info(
                f"  ✅ Indexed in {time.time()-stage_start:.2f}s | "
                f"Total vectors in DB: {total_vectors}"
            )

        except Exception as e:
            logger.error(f"  ❌ Index failed: {e}", exc_info=True)
            return IngestionResult(
                success=False,
                file_name=file_name,
                pages_loaded=pages_loaded,
                chunks_created=chunks_created,
                error_message=f"Indexing error: {str(e)}",
            )

        # ── SUCCESS ───────────────────────────────────────
        total_time = time.time() - start_time

        logger.info(
            f"🎉 Ingestion complete: '{file_name}' | "
            f"{pages_loaded} pages → {chunks_created} chunks | "
            f"Total time: {total_time:.2f}s"
        )

        return IngestionResult(
            success=True,
            file_name=file_name,
            chunks_created=chunks_created,
            total_vectors=total_vectors,
            ingestion_time=round(total_time, 3),
            pages_loaded=pages_loaded,
        )

    # ══════════════════════════════════════════════════
    # PUBLIC METHOD 2: Query the System
    # ══════════════════════════════════════════════════

    def query(
        self,
        question: str,
        strategy: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> QueryResult:
        """
        Answer a question using the full RAG pipeline.

        Full pipeline:
          question
            → retriever.retrieve()    [Phase 6]
            → retriever.format_context()
            → llm_chain.generate()   [Phase 7]
            → QueryResult

        Args:
            question: User's natural language question.
            strategy: Override retrieval strategy for this query.
                      None = use pipeline's default strategy.
            top_k:    Override number of chunks for this query.

        Returns:
            QueryResult with answer, sources, and metrics.
        """

        if not question or not question.strip():
            return QueryResult(
                success=False,
                question=question,
                error_message="Question cannot be empty.",
            )

        strategy = strategy or self.retrieval_strategy
        logger.info(
            f"❓ Query received: '{question[:60]}' | "
            f"strategy={strategy}"
        )

        pipeline_start = time.time()

        # ── STAGE 1: Retrieve Relevant Chunks ────────────
        try:
            logger.info("  [Stage 1/2] Retrieving context...")
            retrieval_start = time.time()

            retrieval_result: RetrievalResult = (
                self.retriever.retrieve(
                    query=question,
                    strategy=strategy,
                    top_k=top_k,
                )
            )
            # Phase 6: returns RetrievalResult with
            # documents, scores, and metadata.

            retrieval_time = time.time() - retrieval_start

            # ── Handle empty retrieval ────────────────────
            if retrieval_result.is_empty:
                # No relevant chunks found — either no document
                # indexed yet, or query is completely out of scope.
                logger.warning(
                    "  ⚠️  No relevant chunks retrieved."
                )
                return QueryResult(
                    success=True,
                    # success=True because system worked correctly
                    # — it correctly identified nothing relevant.
                    question=question,
                    answer=(
                        "I couldn't find any relevant information "
                        "in the uploaded documents to answer your "
                        "question. Please ensure you have uploaded "
                        "a relevant document."
                    ),
                    chunks_retrieved=0,
                    retrieval_time=round(retrieval_time, 3),
                    total_time=round(
                        time.time() - pipeline_start, 3
                    ),
                    retrieval_strategy=strategy,
                    is_uncertain=True,
                )

            # Format chunks into context string
            context = self.retriever.format_context(
                retrieval_result
            )
            # Phase 6's format_context() produces:
            # "[Source: file.pdf | Page: 3 | Score: 0.91]\ntext...\n\n
            #  [Source: file.pdf | Page: 5 | Score: 0.87]\ntext..."

            logger.info(
                f"  ✅ Retrieved {len(retrieval_result.documents)} "
                f"chunks in {retrieval_time:.2f}s | "
                f"top_score={retrieval_result.top_score:.4f}"
            )

        except Exception as e:
            logger.error(
                f"  ❌ Retrieval failed: {e}", exc_info=True
            )
            return QueryResult(
                success=False,
                question=question,
                error_message=f"Retrieval error: {str(e)}",
            )

        # ── STAGE 2: Generate Answer ──────────────────────
        try:
            logger.info("  [Stage 2/2] Generating answer...")
            gen_start = time.time()

            gen_result: GenerationResult = (
                self.llm_chain.generate(
                    question=question,
                    context=context,
                )
            )
            # Phase 7: sends formatted context + question
            # to OpenAI, returns GenerationResult.

            generation_time = time.time() - gen_start

            logger.info(
                f"  ✅ Generated in {generation_time:.2f}s | "
                f"uncertain={gen_result.is_uncertain}"
            )

        except Exception as e:
            logger.error(
                f"  ❌ Generation failed: {e}", exc_info=True
            )
            return QueryResult(
                success=False,
                question=question,
                chunks_retrieved=len(
                    retrieval_result.documents
                ),
                retrieval_time=round(retrieval_time, 3),
                error_message=f"Generation error: {str(e)}",
            )

        # ── Build Source Attribution ──────────────────────
        sources = [
            {
                "file_name": doc.metadata.get(
                    "file_name", "unknown"
                ),
                "page": doc.metadata.get("page", "?"),
                "score": round(score, 4),
                "chunk_preview": (
                    doc.page_content[:150] + "..."
                    if len(doc.page_content) > 150
                    else doc.page_content
                ),
                # Ternary expression: value_if_true if condition
                # else value_if_false.
                # Truncate long chunks to 150 chars for preview.
            }
            for doc, score in zip(
                retrieval_result.documents,
                retrieval_result.scores,
            )
        ]
        # List comprehension over zipped (doc, score) pairs.
        # zip() pairs each document with its parallel score.

        # ── SUCCESS ───────────────────────────────────────
        total_time = time.time() - pipeline_start

        logger.info(
            f"🎉 Query complete in {total_time:.2f}s | "
            f"retrieval={retrieval_time:.2f}s | "
            f"generation={generation_time:.2f}s"
        )

        return QueryResult(
            success=True,
            question=question,
            answer=gen_result.answer,
            sources=sources,
            retrieval_strategy=strategy,
            chunks_retrieved=len(retrieval_result.documents),
            top_retrieval_score=retrieval_result.top_score,
            retrieval_time=round(retrieval_time, 3),
            generation_time=round(generation_time, 3),
            total_time=round(total_time, 3),
            is_uncertain=gen_result.is_uncertain,
            token_estimate=gen_result.token_estimate,
            model_used=gen_result.model_used,
        )

    # ══════════════════════════════════════════════════
    # Utility Methods
    # ══════════════════════════════════════════════════

    def get_system_status(self) -> dict:
        """
        Return health and stats of the entire pipeline.
        Used by the API's /health endpoint (Phase 9).
        """

        index_stats = self.vectorstore.get_index_stats()

        return {
            "status": "ready" if index_stats[
                "total_vectors"
            ] > 0 else "empty",
            "index": index_stats,
            "retrieval_strategy": self.retrieval_strategy,
            "llm_model": self.llm_chain.model_name,
            "embedding_model": (
                self.embedding_manager.get_model_info()
            ),
        }

    def reset_index(self) -> dict:
        """
        Clear the FAISS index. Removes all indexed documents.
        Useful for: starting fresh, testing, or replacing
        the entire document corpus.

        ⚠️ DESTRUCTIVE — cannot be undone.
        The API layer should require confirmation before calling.
        """

        import shutil
        # shutil: high-level file operations module.
        # shutil.rmtree() deletes a directory and all contents.

        persist_dir = self.vectorstore.persist_dir

        if persist_dir.exists():
            shutil.rmtree(persist_dir)
            # Deletes the entire vectorstore_index/ directory.
            persist_dir.mkdir(parents=True, exist_ok=True)
            # Recreate empty directory immediately.

        # Reset the vectorstore object in memory
        self.vectorstore.vectorstore = None
        # Set to None so is_empty checks work correctly.

        logger.info("🗑️  Index reset. All documents cleared.")

        return {"status": "reset", "message": "Index cleared."}