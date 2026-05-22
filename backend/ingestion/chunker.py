# backend/ingestion/chunker.py
# ─────────────────────────────────────────────────────
# PURPOSE: Take a list of LangChain Document objects
# (raw pages from the loader) and split them into
# smaller, overlapping chunks ready for embedding.
# ─────────────────────────────────────────────────────

import logging
from typing import List

from langchain.schema import Document
# Document: the standard LangChain container we built
# intuition for in Phase 2. We receive List[Document]
# and return a new, larger List[Document] of chunks.

from langchain.text_splitter import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter: LangChain's most
# intelligent general-purpose splitter.
#
# HOW IT WORKS (the "recursive" part):
# It tries separators in ORDER, falling back to the next
# if a split would exceed chunk_size:
#
#   1st try: split on "\n\n"  (paragraph breaks)
#   2nd try: split on "\n"    (line breaks)
#   3rd try: split on ". "    (sentence ends)
#   4th try: split on " "     (word boundaries)
#   5th try: split on ""      (individual characters)
#
# It STOPS at the first separator that keeps chunks
# within chunk_size. This is why it respects natural
# text structure better than CharacterTextSplitter.

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP
# Import constants from config — never hardcode these.
# Centralizing config means you can tune the whole
# system by changing ONE file.

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Args:
        documents:     List of raw Document objects from loader
        chunk_size:    Max characters per chunk (default from config)
        chunk_overlap: Characters to repeat between adjacent chunks

    Returns:
        List of chunked Document objects, each with inherited
        metadata PLUS chunk-specific fields added.
    """

    # ── 1. Guard against empty input ─────────────────────
    if not documents:
        logger.warning("chunk_documents() received empty list.")
        return []
        # Return early — nothing to process.
        # This prevents confusing errors downstream.

    # ── 2. Build the splitter ────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        # Maximum number of CHARACTERS per chunk.
        # Note: this is characters, not tokens!
        # 1 token ≈ 4 characters (rough rule of thumb).
        # So chunk_size=500 chars ≈ 125 tokens.

        chunk_overlap=chunk_overlap,
        # Characters to REPEAT from the end of one chunk
        # at the START of the next chunk.
        # Creates a "sliding window" effect.

        length_function=len,
        # The function used to MEASURE chunk size.
        # len() counts characters by default.
        # Alternative: use tiktoken to count tokens instead:
        #   import tiktoken
        #   enc = tiktoken.get_encoding("cl100k_base")
        #   length_function = lambda x: len(enc.encode(x))
        # Use token counting for production with OpenAI models.

        separators=["\n\n", "\n", ". ", " ", ""],
        # The ordered list of separators to try.
        # This IS the "recursive" part.
        # "\n\n" = paragraph break (most preferred)
        # "\n"   = line break
        # ". "   = sentence end (note the space after period)
        # " "    = word boundary
        # ""     = character level (last resort)

        add_start_index=True,
        # Adds "start_index" to chunk metadata.
        # Records WHERE in the original document this chunk
        # came from (character offset).
        # Useful for: highlighting source text in the UI.
    )

    logger.info(
        f"Chunking {len(documents)} document(s) | "
        f"chunk_size={chunk_size} | overlap={chunk_overlap}"
    )

    # ── 3. Split the documents ───────────────────────────
    chunks = splitter.split_documents(documents)
    # split_documents() does two things:
    #   a) Calls split_text() on each document's page_content
    #   b) INHERITS the original document's metadata into
    #      every chunk it produces
    #
    # So if the original doc had:
    #   metadata = {"source": "manual.pdf", "page": 3}
    # Every chunk from it will have:
    #   metadata = {"source": "manual.pdf", "page": 3,
    #               "start_index": 240}
    #
    # This is crucial — it means when we retrieve a chunk
    # later, we KNOW which page and file it came from.
    # We can show users: "Answer found on page 3 of manual.pdf"

    # ── 4. Add chunk index metadata ─────────────────────
    for i, chunk in enumerate(chunks):
        # enumerate() gives us both index i and the chunk object.
        # i starts at 0 by default.

        chunk.metadata["chunk_index"] = i
        # Track the global position of this chunk.
        # Useful for debugging ("which chunk is retrieved most?")

        chunk.metadata["chunk_total"] = len(chunks)
        # Total chunks in this batch.
        # Helps the UI show "chunk 4 of 47" style citations.

    # ── 5. Log result statistics ─────────────────────────
    _log_chunk_stats(chunks)
    # Separate function to keep this function clean.
    # Single Responsibility Principle: each function
    # does ONE thing well.

    return chunks


def _log_chunk_stats(chunks: List[Document]) -> None:
    """
    Log useful statistics about the chunks produced.
    Helps with tuning chunk_size and chunk_overlap.
    """

    if not chunks:
        return

    # Calculate character lengths of all chunks
    lengths = [len(chunk.page_content) for chunk in chunks]
    # List comprehension: concise way to build a list
    # by applying an expression to each item.
    # Equivalent to:
    #   lengths = []
    #   for chunk in chunks:
    #       lengths.append(len(chunk.page_content))

    avg_length = sum(lengths) / len(lengths)
    min_length = min(lengths)
    max_length = max(lengths)

    logger.info(
        f"✅ Created {len(chunks)} chunks | "
        f"avg={avg_length:.0f} chars | "
        f"min={min_length} | max={max_length}"
        # :.0f formats float with 0 decimal places
    )


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Simpler utility: chunk a raw string (not a Document).
    
    Used for: chunking text that comes from APIs,
    web scraping, or other non-file sources.
    Returns List[str] instead of List[Document].
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    return splitter.split_text(text)
    # split_text() works on raw strings.
    # split_documents() works on List[Document].
    # Know which to use based on your input type.
