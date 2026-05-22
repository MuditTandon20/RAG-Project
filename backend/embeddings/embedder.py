# backend/embeddings/embedder.py
# ─────────────────────────────────────────────────────
# PURPOSE: Convert text chunks into numerical vectors
# (embeddings) using OpenAI's embedding API.
# These vectors capture semantic meaning and will be
# stored in FAISS for similarity search in Phase 5.
# ─────────────────────────────────────────────────────

import logging
import time
# time: built-in module to measure elapsed time.
# We use it to track how long embedding takes —
# critical for understanding API latency in production.

from typing import List, Tuple
import numpy as np
# numpy: THE numerical computing library for Python.
# Used everywhere in ML — arrays, matrix operations, etc.
# We use it here to work with raw embedding vectors.

from langchain_openai import OpenAIEmbeddings
# OpenAIEmbeddings: LangChain's wrapper around OpenAI's
# embedding API. It handles:
#   - API authentication
#   - Batching (sending multiple texts in one API call)
#   - Retry logic on rate limits
#   - Returning numpy-compatible float arrays
#
# Alternative: HuggingFaceEmbeddings for local models:
#   from langchain_community.embeddings import HuggingFaceEmbeddings
#   embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#   Everything else stays the same! This is LangChain's power.

from langchain.schema import Document

from backend.config import OPENAI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """
    Manages all embedding operations for the RAG system.

    WHY A CLASS instead of functions?
    The embedding model is expensive to initialize
    (loads config, sets up API connection). By wrapping
    it in a class, we initialize it ONCE and reuse it
    for every embedding call. This is called the
    "singleton-like" pattern — one instance, shared.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding model.

        Args:
            model_name: OpenAI embedding model to use.
                        Default from config.
        """

        self.model_name = model_name

        logger.info(f"Initializing embedding model: {model_name}")

        # ── Create the LangChain embeddings object ───────
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            # Which OpenAI embedding model to call.

            openai_api_key=OPENAI_API_KEY,
            # Pass our API key from config.
            # Never hardcode: OpenAIEmbeddings(api_key="sk-...")
            # Always pull from environment.

            chunk_size=500,
            # How many texts to send per API batch request.
            # OpenAI allows up to 2048 inputs per request.
            # 500 is conservative and avoids rate limit errors.
            # Higher = fewer API calls = faster for large batches.
            # Lower = more reliable under rate limits.
        )

        logger.info("✅ Embedding model ready.")

    def embed_documents(
        self, chunks: List[Document]
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        Generate embeddings for a list of Document chunks.

        This is called ONCE during ingestion to embed
        all your document chunks before storing in FAISS.

        Args:
            chunks: List of Document objects from chunker

        Returns:
            Tuple of:
              - Original chunks (unchanged, for reference)
              - List of embedding vectors (one per chunk)

        Why return BOTH?
        FAISS needs vectors. But we also need to store the
        original text to show users WHAT was retrieved.
        We keep them paired so index i in vectors always
        corresponds to index i in chunks.
        """

        if not chunks:
            logger.warning("embed_documents() received empty list.")
            return [], []

        # Extract just the text content from each Document
        texts = [chunk.page_content for chunk in chunks]
        # List comprehension: builds ["text of chunk 1",
        #                             "text of chunk 2", ...]
        # We embed the TEXT, not the whole Document object.

        logger.info(
            f"Embedding {len(texts)} chunks using "
            f"'{self.model_name}'..."
        )

        start_time = time.time()
        # Record start time BEFORE the API call.
        # We'll subtract this after to get elapsed time.

        # ── The actual embedding call ─────────────────────
        vectors = self.embeddings.embed_documents(texts)
        # This makes an HTTP POST to:
        #   https://api.openai.com/v1/embeddings
        # Sends all texts in batches of chunk_size.
        # Returns: List[List[float]]
        #   Outer list: one per input text
        #   Inner list: 1536 floats (the embedding vector)
        #
        # Example return for 3 chunks:
        # [
        #   [0.023, -0.811, 0.442, ...],  # chunk 1 vector
        #   [0.118, -0.224, 0.891, ...],  # chunk 2 vector
        #   [-0.334, 0.671, 0.023, ...],  # chunk 3 vector
        # ]

        elapsed = time.time() - start_time
        # Subtract start from current time → elapsed seconds.

        logger.info(
            f"✅ Embedded {len(vectors)} chunks in "
            f"{elapsed:.2f}s | "
            f"Avg: {elapsed/len(texts)*1000:.0f}ms per chunk"
        )

        # ── Validation ────────────────────────────────────
        assert len(vectors) == len(chunks), (
            f"Mismatch! {len(chunks)} chunks but "
            f"{len(vectors)} vectors returned."
        )
        # assert: crashes immediately if condition is False.
        # This is a "sanity check" — if OpenAI somehow
        # returns wrong number of vectors, we catch it HERE
        # rather than getting a confusing error in FAISS later.

        return chunks, vectors

    def embed_query(self, query: str) -> List[float]:
        """
        Generate an embedding for a single user query.

        Called EVERY TIME a user asks a question.
        This query vector is then compared against all
        stored chunk vectors using cosine similarity.

        Args:
            query: The user's natural language question

        Returns:
            A single embedding vector (List of 1536 floats)

        WHY a separate method from embed_documents?
        OpenAI actually uses DIFFERENT internal prompting
        for document embeddings vs query embeddings in
        some newer models. LangChain handles this for you
        by calling the right API mode automatically.
        Always use embed_query() for questions and
        embed_documents() for your corpus chunks.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
            # strip() removes leading/trailing whitespace.
            # Catches cases where user sends "   " (spaces only).

        logger.info(f"Embedding query: '{query[:60]}...'")
        # [:60] truncates long queries in the log.
        # You don't want full 500-char queries in every log line.

        vector = self.embeddings.embed_query(query)
        # Returns a single List[float] — not a list of lists,
        # because there's only one query.

        logger.info(
            f"✅ Query embedded | "
            f"Vector dimensions: {len(vector)}"
        )

        return vector

    def compute_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Included as a utility to help you understand and
        debug retrieval. In production, FAISS does this
        internally — you won't call this directly in the
        main pipeline. But it's invaluable for:
          - Debugging why certain chunks ARE/aren't retrieved
          - Building intuition for your embedding space
          - Writing unit tests for retrieval quality

        Args:
            vec1, vec2: Two embedding vectors

        Returns:
            Cosine similarity score (float between -1 and 1)
        """

        # Convert Python lists to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        # numpy arrays support vectorized math operations.
        # Without numpy, you'd need a for loop to multiply
        # each element — numpy does it in C, massively faster.

        # ── Cosine Similarity Formula ─────────────────────
        dot_product = np.dot(a, b)
        # np.dot(): computes dot product.
        # For 1D arrays: sum of element-wise products.
        # For 2D arrays: matrix multiplication.
        # Here we use 1D: a[0]*b[0] + a[1]*b[1] + ...

        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        # np.linalg.norm(): Euclidean length of vector.
        # = √(a[0]² + a[1]² + ... + a[n]²)
        # "linalg" = linear algebra submodule of numpy

        if magnitude_a == 0 or magnitude_b == 0:
            # Guard against zero vectors (empty text edge case).
            # Division by zero would give NaN — catch it early.
            return 0.0

        similarity = dot_product / (magnitude_a * magnitude_b)
        # This IS the cosine similarity formula:
        # cos(θ) = (A · B) / (|A| × |B|)

        return float(similarity)
        # Convert numpy float64 to Python float.
        # Avoids type issues when serializing to JSON later.

    def get_model_info(self) -> dict:
        """
        Return metadata about the current embedding model.
        Useful for logging and API responses.
        """
        return {
            "model_name": self.model_name,
            "dimensions": 1536 if "small" in self.model_name
                          else 3072,
            "provider": "OpenAI",
        }