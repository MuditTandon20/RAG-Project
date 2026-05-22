# backend/retrieval/retriever.py
# ─────────────────────────────────────────────────────
# PURPOSE: Provide a clean, configurable retrieval
# interface on top of our VectorStore.
# This is the component the RAG chain (Phase 8) calls
# to get relevant context for a user query.
# ─────────────────────────────────────────────────────

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
# dataclass: decorator that auto-generates __init__,
# __repr__, __eq__ for classes that mainly hold data.
# Cleaner than writing boilerplate constructors manually.

from langchain.schema import Document

from backend.vectorstore.store import VectorStore
from backend.config import TOP_K_RESULTS

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Structured container for retrieval results.

    WHY a dataclass instead of returning raw List[Document]?
    - Adds metadata about the retrieval itself (strategy used,
      scores, whether results were filtered)
    - Makes the API response richer and more debuggable
    - Type-safe: callers know exactly what fields to expect
    - Easy to serialize to JSON for the API layer (Phase 9)

    @dataclass automatically generates:
      __init__(self, documents, scores, ...)
      __repr__ for clean printing
    """

    documents: List[Document]
    # The retrieved chunks, sorted by relevance.

    scores: List[float] = field(default_factory=list)
    # Similarity score for each document.
    # Parallel list: scores[i] corresponds to documents[i].
    # default_factory=list: creates a new empty list for
    # each instance (never share mutable defaults!).

    query: str = ""
    # The original query — useful for logging and debugging.

    strategy: str = "similarity"
    # Which retrieval strategy was used.
    # One of: "similarity", "mmr", "threshold"

    total_chunks_searched: int = 0
    # How many chunks were in the index when we searched.
    # Useful for understanding retrieval coverage.

    filtered_count: int = 0
    # How many chunks were filtered out by score threshold.
    # If this is high, your threshold may be too strict
    # OR the document doesn't contain relevant info.

    @property
    def is_empty(self) -> bool:
        """True if no documents were retrieved."""
        return len(self.documents) == 0
        # Property: accessed like an attribute (result.is_empty)
        # not a method call (result.is_empty()).
        # Use for computed values that don't change state.

    @property
    def top_score(self) -> Optional[float]:
        """Similarity score of the best retrieved chunk."""
        return self.scores[0] if self.scores else None

    def __repr__(self) -> str:
        return (
            f"RetrievalResult("
            f"docs={len(self.documents)}, "
            f"top_score={self.top_score:.4f if self.top_score else 'N/A'}, "
            f"strategy='{self.strategy}')"
        )


class Retriever:
    """
    High-level retrieval interface for the RAG pipeline.

    Sits between the VectorStore (raw FAISS operations)
    and the RAG chain (which just wants relevant Documents).

    Implements three retrieval strategies:
      1. similarity()   — basic top-k
      2. with_threshold()— top-k with minimum score filter
      3. mmr()          — diverse top-k using MMR algorithm

    This separation of concerns means:
      - VectorStore handles HOW vectors are stored/searched
      - Retriever handles WHAT strategy to apply
      - RAG chain handles WHAT to do with the results
    Each class has ONE clear responsibility.
    """

    def __init__(
        self,
        vectorstore: VectorStore,
        top_k: int = TOP_K_RESULTS,
        score_threshold: float = 0.5,
        mmr_lambda: float = 0.5,
    ):
        """
        Args:
            vectorstore:      Our VectorStore from Phase 5.
            top_k:            Default number of chunks to return.
            score_threshold:  Minimum similarity score (0-1).
                              Chunks below this are rejected.
                              0.5 = reject if less than 50% similar.
                              Tune based on your use case:
                                strict domain (legal/medical): 0.7
                                general knowledge: 0.4-0.5
            mmr_lambda:       MMR diversity-relevance balance.
                              0.5 = equal weight to both.
                              Increase toward 1.0 for more relevance.
                              Decrease toward 0.0 for more diversity.
        """

        self.vectorstore = vectorstore
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.mmr_lambda = mmr_lambda

        logger.info(
            f"Retriever initialized | "
            f"top_k={top_k} | "
            f"threshold={score_threshold} | "
            f"mmr_lambda={mmr_lambda}"
        )

    # ══════════════════════════════════════════════════════
    # Strategy 1: Basic Similarity Search
    # ══════════════════════════════════════════════════════

    def similarity(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Retrieve top-k chunks by cosine similarity.

        Fastest strategy. Good starting point.
        No diversity guarantee — may return redundant chunks.

        Args:
            query:  User's natural language question
            top_k:  Override default top_k if provided

        Returns:
            RetrievalResult with ranked documents and scores
        """

        k = top_k or self.top_k
        # Use provided top_k, fall back to instance default.
        # "or" works here because we never want k=0 (falsy).

        logger.info(f"[similarity] Query: '{query[:60]}'")

        results_with_scores = (
            self.vectorstore.similarity_search_with_scores(
                query=query,
                top_k=k,
            )
        )
        # Returns List[Tuple[Document, float]]
        # from our Phase 5 VectorStore implementation.

        # Unzip the list of tuples into two parallel lists
        if results_with_scores:
            documents, scores = zip(*results_with_scores)
            # zip(*list_of_tuples) transposes:
            # [(doc1, 0.9), (doc2, 0.8)] →
            # (doc1, doc2), (0.9, 0.8)
            documents = list(documents)
            scores = list(scores)
        else:
            documents, scores = [], []

        total = self.vectorstore.vectorstore.index.ntotal
        # Raw FAISS property: total vectors in the index.
        # Tells us how many chunks we searched through.

        result = RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
            strategy="similarity",
            total_chunks_searched=total,
        )

        logger.info(f"[similarity] Retrieved: {result}")
        return result

    # ══════════════════════════════════════════════════════
    # Strategy 2: Similarity with Score Threshold
    # ══════════════════════════════════════════════════════

    def with_threshold(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve top-k chunks, then filter by minimum score.

        WHY this matters:
          Without threshold: LLM gets low-quality context
          and may hallucinate or give irrelevant answers.

          With threshold: LLM gets "I found nothing relevant"
          signal and responds "I don't have information about
          this in the provided documents." — honest and correct.

        Args:
            query:      User query
            top_k:      Max chunks to return (before filtering)
            threshold:  Min score. Override instance default.
        """

        k = top_k or self.top_k
        min_score = threshold if threshold is not None \
                    else self.score_threshold
        # 'if threshold is not None' instead of 'or':
        # because threshold=0.0 is a valid value but
        # 0.0 is falsy, so 'threshold or default' would
        # incorrectly use the default when threshold=0.0.

        logger.info(
            f"[threshold] Query: '{query[:60]}' | "
            f"min_score={min_score}"
        )

        # First get more candidates than top_k
        # so we have headroom after filtering.
        candidate_k = min(k * 3, 20)
        # Fetch 3× more than needed, up to 20 max.
        # Example: top_k=4 → fetch 12, filter to best 4.
        # Without this, filtering might leave you with
        # fewer results than top_k even though more
        # relevant chunks exist further in the ranking.

        results_with_scores = (
            self.vectorstore.similarity_search_with_scores(
                query=query,
                top_k=candidate_k,
            )
        )

        # ── Apply score threshold filter ──────────────────
        filtered = [
            (doc, score)
            for doc, score in results_with_scores
            if score >= min_score
            # Keep only chunks with similarity >= threshold.
            # score=0.9 → very relevant ✅
            # score=0.3 → barely related ❌
        ]
        # List comprehension with condition:
        # builds a new list including only items
        # where the condition (score >= min_score) is True.

        filtered_count = len(results_with_scores) - len(filtered)
        # How many chunks were removed by the filter.
        # Log this — high filtered_count = query is
        # not covered by the document corpus.

        # Take only top_k from the filtered results
        final = filtered[:k]
        # Slicing: takes first k items from filtered list.
        # filtered is already sorted by score (FAISS returns
        # results in order of similarity), so [:k] gives
        # the best k that passed the threshold.

        if final:
            documents, scores = zip(*final)
            documents, scores = list(documents), list(scores)
        else:
            documents, scores = [], []
            logger.warning(
                f"[threshold] No chunks passed threshold "
                f"{min_score} for query: '{query[:60]}'"
            )
            # This is a WARNING not an ERROR —
            # it's expected behavior, not a system failure.

        total = self.vectorstore.vectorstore.index.ntotal

        result = RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
            strategy="threshold",
            total_chunks_searched=total,
            filtered_count=filtered_count,
        )

        logger.info(f"[threshold] Retrieved: {result}")
        return result

    # ══════════════════════════════════════════════════════
    # Strategy 3: MMR — Maximal Marginal Relevance
    # ══════════════════════════════════════════════════════

    def mmr(
        self,
        query: str,
        top_k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
    ) -> RetrievalResult:
        """
        Retrieve diverse, relevant chunks using MMR.

        Best strategy for production RAG — prevents the
        echo chamber problem where all retrieved chunks
        say the same thing.

        Args:
            query:       User query
            top_k:       Final number of chunks to return
            fetch_k:     Candidate pool size before MMR.
                         Larger pool = better MMR selection.
                         Typical: fetch_k = top_k × 4-5
            lambda_mult: MMR lambda. Higher = more relevance.
                         Lower = more diversity.
        """

        k = top_k or self.top_k
        fk = fetch_k or k * 4
        # fetch_k is the candidate pool MMR picks from.
        # Fetch 4× more candidates, let MMR pick the best k.
        # Example: top_k=4 → fetch 16 candidates,
        # MMR selects 4 that are relevant AND diverse.

        lm = lambda_mult if lambda_mult is not None \
             else self.mmr_lambda

        logger.info(
            f"[MMR] Query: '{query[:60]}' | "
            f"top_k={k} | fetch_k={fk} | lambda={lm}"
        )

        # ── MMR search via LangChain ──────────────────────
        documents = self.vectorstore.vectorstore.max_marginal_relevance_search(
            query=query,
            # Query string — LangChain embeds it internally.

            k=k,
            # Final number of chunks to return after MMR.

            fetch_k=fk,
            # How many candidates to retrieve BEFORE MMR.
            # MMR then selects k diverse chunks from these fk.

            lambda_mult=lm,
            # The λ parameter from the MMR formula.
            # 1.0 = pure similarity (no diversity penalty)
            # 0.0 = pure diversity (ignores query relevance)
            # 0.5 = balanced (our default)
        )
        # max_marginal_relevance_search returns List[Document]
        # WITHOUT scores — MMR scores are internal and not
        # directly meaningful to expose (they mix similarity
        # and diversity into a single composite score).

        # Generate approximate scores for the MMR results
        # by doing a quick similarity search for reference.
        scores = self._approximate_scores(query, documents)

        total = self.vectorstore.vectorstore.index.ntotal

        result = RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
            strategy="mmr",
            total_chunks_searched=total,
        )

        logger.info(f"[MMR] Retrieved: {result}")
        return result

    # ══════════════════════════════════════════════════════
    # Unified Interface — smart strategy selection
    # ══════════════════════════════════════════════════════

    def retrieve(
        self,
        query: str,
        strategy: str = "mmr",
        top_k: Optional[int] = None,
    ) -> RetrievalResult:
        """
        Unified retrieval entry point.

        The RAG chain (Phase 8) calls THIS method —
        it doesn't need to know which strategy is used.
        Strategy can be configured globally or per-request.

        Args:
            query:    User question
            strategy: "similarity" | "threshold" | "mmr"
            top_k:    Number of chunks to return

        Returns:
            RetrievalResult
        """

        # Strategy routing — dispatch pattern
        strategy_map = {
            "similarity": self.similarity,
            "threshold":  self.with_threshold,
            "mmr":        self.mmr,
        }
        # Dictionary as a dispatch table — cleaner than
        # if/elif/else chains. Adding a new strategy
        # means adding ONE entry here, not touching the logic.

        if strategy not in strategy_map:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {list(strategy_map.keys())}"
            )

        retrieval_fn = strategy_map[strategy]
        # Get the function object (not calling it yet).
        # retrieval_fn is now a bound method of self.

        return retrieval_fn(query=query, top_k=top_k)
        # NOW call it with the query.
        # This is called "late binding" — we decide WHICH
        # function to call at runtime, not compile time.

    # ══════════════════════════════════════════════════════
    # Helper Methods
    # ══════════════════════════════════════════════════════

    def _approximate_scores(
        self,
        query: str,
        documents: List[Document],
    ) -> List[float]:
        """
        Get approximate similarity scores for a list of documents.

        Used after MMR to attach meaningful scores to results,
        since MMR doesn't return scores directly.

        Convention: methods starting with _ are "private" —
        internal helpers not meant for external callers.
        Python doesn't enforce this — it's a convention.
        """

        if not documents:
            return []

        # Get similarity search results to find scores
        all_results = self.vectorstore.similarity_search_with_scores(
            query=query,
            top_k=20,
        )
        # Fetch more results than we need as a lookup table.

        # Build a lookup: page_content → score
        score_lookup = {
            doc.page_content: score
            for doc, score in all_results
        }
        # Dictionary comprehension: builds {text: score, ...}
        # We key by page_content because Document objects
        # don't have a reliable unique ID to key by.

        # Look up score for each MMR-selected document
        scores = [
            score_lookup.get(doc.page_content, 0.0)
            for doc in documents
        ]
        # .get(key, default): returns default (0.0) if
        # the document isn't in our lookup — shouldn't
        # happen but defensive programming is good practice.

        return scores

    def format_context(
        self,
        result: RetrievalResult,
    ) -> str:
        """
        Format retrieved documents into a single context
        string ready to be injected into the LLM prompt.

        This is the bridge between retrieval (Phase 6)
        and generation (Phase 7).

        Args:
            result: RetrievalResult from any strategy

        Returns:
            Formatted string like:

            [Source: manual.pdf | Page: 3 | Score: 0.91]
            The Model X has a range of 348 miles...

            [Source: manual.pdf | Page: 5 | Score: 0.87]
            Range varies based on temperature...
        """

        if result.is_empty:
            return "No relevant context found in the documents."

        context_parts = []

        for i, (doc, score) in enumerate(
            zip(result.documents, result.scores)
        ):
            # zip() pairs each document with its score.
            # Since they're parallel lists, zip gives us
            # (doc1, score1), (doc2, score2), ...

            source = doc.metadata.get("file_name", "unknown")
            page   = doc.metadata.get("page", "?")
            # .get() with default avoids KeyError if
            # metadata field is missing.

            header = (
                f"[Source: {source} | "
                f"Page: {page} | "
                f"Score: {score:.2f}]"
            )
            content = doc.page_content.strip()
            # .strip() removes leading/trailing whitespace
            # and newlines — keeps context clean.

            context_parts.append(f"{header}\n{content}")

        return "\n\n".join(context_parts)
        # Join all chunks with double newline between them.
        # Double newline creates clear visual separation
        # in the prompt, helping the LLM distinguish chunks.