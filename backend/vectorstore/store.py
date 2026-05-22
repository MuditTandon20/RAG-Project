# backend/vectorstore/store.py
# ─────────────────────────────────────────────────────
# PURPOSE: Take embedded document chunks and store
# them in a FAISS index for fast similarity search.
# Also handles saving to disk and loading back —
# so you don't re-embed every time the server restarts.
# ─────────────────────────────────────────────────────

import os
import logging
import pickle
# pickle: Python's built-in serialization module.
# Converts Python objects (dicts, lists, custom classes)
# into bytes that can be saved to disk and loaded back.
# We use it to save chunk metadata alongside FAISS index.
# ⚠️ Never unpickle data from untrusted sources —
# malicious pickle files can execute arbitrary code.

from typing import List, Optional, Tuple
from pathlib import Path

import faiss
# faiss: Facebook AI Similarity Search library.
# Written in C++ with Python bindings — that's why
# it's so fast. The Python API is a thin wrapper
# over highly optimized C++ code.
# Install: pip install faiss-cpu
# For GPU:  pip install faiss-gpu (requires CUDA)

import numpy as np
# numpy: We need this to convert Python lists to
# the float32 arrays that FAISS requires.
# FAISS is picky: it ONLY accepts np.float32, not
# float64 (Python's default) or Python lists.

from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS as LangChainFAISS
# LangChainFAISS: LangChain's wrapper around raw FAISS.
# It bundles together:
#   - The FAISS index (vectors)
#   - The document store (original text + metadata)
#   - The embedding model (for query-time embedding)
# This is much more convenient than managing them
# separately — which we'll show for learning purposes,
# then use the wrapper for the actual pipeline.

from backend.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    VECTORSTORE_DIR,
    TOP_K_RESULTS,
)
from backend.embeddings.embedder import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages the FAISS vector index for the RAG system.

    Responsibilities:
      1. Build index from document chunks + their vectors
      2. Save index to disk (persistence)
      3. Load index from disk (avoid re-embedding on restart)
      4. Search index for a query vector (retrieval)

    Design decision — why wrap LangChain's FAISS?
    LangChain's FAISS wrapper handles the pairing of
    vectors ↔ documents automatically. Raw FAISS only
    stores vectors — you'd manage document lookup yourself.
    We use LangChain's wrapper for the pipeline but also
    show raw FAISS internals for learning.
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        persist_dir: str = VECTORSTORE_DIR,
    ):
        """
        Args:
            embedding_manager: Our EmbeddingManager from Phase 4.
                               Needed to embed queries at search time.
            persist_dir:       Directory to save/load FAISS index.
        """

        self.embedding_manager = embedding_manager
        self.persist_dir = Path(persist_dir)
        # Convert to Path object for cleaner path operations.

        self.vectorstore: Optional[LangChainFAISS] = None
        # Will hold the LangChain FAISS instance after
        # build_index() or load_index() is called.
        # Optional means it can be None (before initialization).

        # Create persist directory if it doesn't exist
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        # mkdir(parents=True): creates parent dirs too.
        #   e.g., if "a/b/c" doesn't exist, creates a, b, c.
        # exist_ok=True: doesn't raise error if dir exists.
        # Without these: would crash if dir already exists
        # or if parent dirs don't exist.

        logger.info(
            f"VectorStore initialized | "
            f"persist_dir={self.persist_dir}"
        )

    def build_index(self, chunks: List[Document]) -> None:
        """
        Build a FAISS index from document chunks.

        This is called during INGESTION — when a new
        document is uploaded and processed. It:
          1. Embeds all chunks (calls OpenAI API)
          2. Builds FAISS index
          3. Auto-saves to disk

        Args:
            chunks: List of Document objects from chunker
        """

        if not chunks:
            raise ValueError("Cannot build index from empty chunks.")

        logger.info(
            f"Building FAISS index from {len(chunks)} chunks..."
        )

        # ── Build index using LangChain's FAISS wrapper ──────
        self.vectorstore = LangChainFAISS.from_documents(
            documents=chunks,
            # The chunks from our Phase 3 chunker.
            # LangChain extracts page_content for embedding
            # and stores full Document (with metadata) for retrieval.

            embedding=self.embedding_manager.embeddings,
            # The OpenAIEmbeddings object from Phase 4.
            # LangChain uses this to embed all chunks automatically.
            # It calls embed_documents() internally — same as
            # what we built in Phase 4, now used here.
        )
        # What happens inside from_documents():
        #   1. Extracts text from each chunk's page_content
        #   2. Calls OpenAI embeddings API in batches
        #   3. Creates FAISS IndexFlatL2 internally
        #   4. Adds all vectors to the index
        #   5. Stores Document objects in an internal dict
        #      keyed by a UUID for each chunk
        #
        # After this line: self.vectorstore is a fully
        # searchable FAISS index with 1 vector per chunk.

        logger.info("✅ FAISS index built successfully.")

        # Auto-save after building
        self.save_index()
        # Always persist immediately after building.
        # If the server crashes after building but before
        # saving, all the embedding API calls are wasted.

    def save_index(self) -> None:
        """
        Save the FAISS index to disk.

        FAISS saves two files:
          - index.faiss: the binary vector index
          - index.pkl:   the document store (text + metadata)

        Without saving, the index lives only in RAM.
        Server restart = all embeddings lost = must re-embed.
        Re-embedding 10,000 chunks costs ~$0.10 per restart.
        Saving prevents this completely.
        """

        if self.vectorstore is None:
            raise RuntimeError(
                "No index to save. Call build_index() first."
            )

        save_path = str(self.persist_dir)

        self.vectorstore.save_local(save_path)
        # save_local() creates two files in the directory:
        #   {save_path}/index.faiss  ← binary FAISS index
        #   {save_path}/index.pkl    ← pickled document store
        #
        # The .faiss file is FAISS's own binary format.
        # The .pkl file is Python's pickle of the
        # InMemoryDocstore object that maps IDs → Documents.

        logger.info(f"💾 Index saved to '{save_path}'")

        # Log file sizes for monitoring
        faiss_file = self.persist_dir / "index.faiss"
        pkl_file = self.persist_dir / "index.pkl"

        if faiss_file.exists():
            size_mb = faiss_file.stat().st_size / (1024 * 1024)
            # stat().st_size → file size in bytes
            # ÷ 1024 → KB
            # ÷ 1024 again → MB
            logger.info(f"   index.faiss: {size_mb:.2f} MB")

        if pkl_file.exists():
            size_kb = pkl_file.stat().st_size / 1024
            logger.info(f"   index.pkl:   {size_kb:.2f} KB")

    def load_index(self) -> bool:
        """
        Load a previously saved FAISS index from disk.

        Called at SERVER STARTUP — avoids re-embedding
        all documents every time the API restarts.

        Returns:
            True if loaded successfully, False if no saved
            index exists (first-time run).
        """

        faiss_file = self.persist_dir / "index.faiss"

        if not faiss_file.exists():
            logger.info(
                "No saved index found. "
                "Upload a document to build one."
            )
            return False
            # Return False so the caller knows to expect
            # no vectorstore yet — not an error, just
            # means no documents have been ingested yet.

        logger.info(f"Loading index from '{self.persist_dir}'...")

        self.vectorstore = LangChainFAISS.load_local(
            folder_path=str(self.persist_dir),
            embeddings=self.embedding_manager.embeddings,
            # Must pass same embedding model used during build!
            # LangChain needs it to embed queries at search time.
            # Using a DIFFERENT model here = garbage results.
            # The vectors in the index were created with model A.
            # If you embed queries with model B, you're comparing
            # apples to oranges in vector space.

            allow_dangerous_deserialization=True,
            # Required since LangChain 0.2.x as a safety flag.
            # pickle deserialization CAN be dangerous if the
            # .pkl file is from an untrusted source.
            # In our system, WE create the .pkl, so it's safe.
        )

        # Count how many vectors are in the loaded index
        num_vectors = self.vectorstore.index.ntotal
        # .index: the raw FAISS index object inside LangChain's wrapper
        # .ntotal: property of FAISS index → total vectors stored
        logger.info(
            f"✅ Loaded index with {num_vectors} vectors."
        )

        return True

    def add_documents(self, chunks: List[Document]) -> None:
        """
        Add NEW chunks to an EXISTING index.

        Used when a user uploads a SECOND document —
        we don't want to rebuild the whole index,
        just add new chunks to it.

        This is called INCREMENTAL INDEXING.
        """

        if not chunks:
            return

        if self.vectorstore is None:
            # No existing index — build fresh instead
            logger.info(
                "No existing index. Building from scratch."
            )
            self.build_index(chunks)
            return

        logger.info(
            f"Adding {len(chunks)} new chunks to existing index..."
        )

        self.vectorstore.add_documents(chunks)
        # add_documents() on an existing FAISS vectorstore:
        #   1. Embeds the new chunks
        #   2. Adds their vectors to the existing index
        #   3. Updates the internal document store
        # Existing vectors are untouched — only appended to.

        # Save after adding
        self.save_index()

        logger.info(
            f"✅ Index now has "
            f"{self.vectorstore.index.ntotal} total vectors."
        )

    def similarity_search(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> List[Document]:
        """
        Find the top-k most semantically similar chunks
        to a given query string.

        This is the RETRIEVAL step — called every time
        a user asks a question.

        Args:
            query: User's natural language question
            top_k: Number of chunks to return

        Returns:
            List of Document objects, sorted by similarity
            (most similar first)
        """

        if self.vectorstore is None:
            raise RuntimeError(
                "Index not built yet. "
                "Please upload a document first."
            )

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(
            f"Searching for top-{top_k} chunks | "
            f"query='{query[:50]}...'"
        )

        # ── The actual similarity search ─────────────────────
        results = self.vectorstore.similarity_search(
            query=query,
            # Raw query string — LangChain embeds it internally
            # using the embedding model we passed at load time.
            # It calls embed_query() internally, same as Phase 4.

            k=top_k,
            # Return the k most similar chunks.
            # Internally, FAISS computes L2 distance between
            # the query vector and ALL stored vectors,
            # then returns the k smallest distances.
        )
        # Returns List[Document] sorted by similarity.
        # result[0] = most relevant chunk
        # result[-1] = least relevant of the top-k

        logger.info(
            f"✅ Retrieved {len(results)} chunks."
        )

        return results

    def similarity_search_with_scores(
        self,
        query: str,
        top_k: int = TOP_K_RESULTS,
    ) -> List[Tuple[Document, float]]:
        """
        Same as similarity_search but also returns
        the similarity score for each result.

        Used for:
          - Debugging retrieval quality
          - Implementing score thresholds
            (e.g., reject chunks with score < 0.7)
          - Showing confidence to users in the UI

        Returns:
            List of (Document, score) tuples.
            Score is cosine similarity: 0.0 to 1.0
            Higher = more similar.
        """

        if self.vectorstore is None:
            raise RuntimeError("Index not built yet.")

        results_with_scores = (
            self.vectorstore.similarity_search_with_relevance_scores(
                query=query,
                k=top_k,
            )
        )
        # Returns List[Tuple[Document, float]]
        # The float is a normalized relevance score
        # between 0 and 1 (LangChain converts raw
        # FAISS L2 distances to similarity scores for you).

        # Log the scores for observability
        for i, (doc, score) in enumerate(results_with_scores):
            logger.info(
                f"  Rank {i+1}: score={score:.4f} | "
                f"source={doc.metadata.get('file_name', 'unknown')} | "
                f"preview='{doc.page_content[:60]}...'"
            )

        return results_with_scores

    def get_index_stats(self) -> dict:
        """
        Return statistics about the current index.
        Useful for the API health endpoint and UI dashboard.
        """

        if self.vectorstore is None:
            return {
                "status": "empty",
                "total_vectors": 0,
                "index_type": None,
                "persist_dir": str(self.persist_dir),
            }

        return {
            "status": "ready",
            "total_vectors": self.vectorstore.index.ntotal,
            # ntotal: raw FAISS property, total vectors in index

            "index_type": type(
                self.vectorstore.index
            ).__name__,
            # e.g., "IndexFlatL2" — tells you which FAISS
            # algorithm is being used

            "dimensions": self.vectorstore.index.d,
            # .d: dimensionality of vectors in the index
            # Should be 1536 for text-embedding-3-small

            "persist_dir": str(self.persist_dir),
        }