# backend/ingestion/loader.py
# ─────────────────────────────────────────────────────
# PURPOSE: Accept a file path, detect its type,
# load it using the appropriate parser, and return
# a standardized list of LangChain Document objects.
# ─────────────────────────────────────────────────────

import os
# os: used to check file existence and extract extensions

import logging
# logging: Python's built-in logging module.
# NEVER use print() in production code — it has no
# timestamps, no severity levels, no easy filtering.
# logging.info(), logging.warning(), logging.error()
# are the professional alternatives.

from pathlib import Path
# Path: modern, object-oriented way to handle file paths.
# Works on Windows, Mac, Linux without path separator issues.
# Path("data/uploads/file.pdf").suffix → ".pdf"

from typing import List
# List: type hint. Makes code self-documenting.
# def foo() -> List[str] tells readers AND tools
# (like VSCode) that this function returns a list of strings.

from langchain_community.document_loaders import (
    PyMuPDFLoader,      # Primary PDF loader
    TextLoader,         # For .txt files
    DirectoryLoader,    # Load all files in a folder
)
# langchain_community: Community-maintained loaders.
# Separated from core langchain to keep the core lean.
# Alternative loaders:
#   - PyPDFLoader: simpler but loses formatting info
#   - UnstructuredFileLoader: handles 20+ formats but
#     heavier dependency, needs system-level installs

from langchain.schema import Document
# Document: The universal LangChain data container.
# Every loader returns List[Document].
# Every downstream component (chunker, embedder) accepts List[Document].
# This is the "lingua franca" of the LangChain pipeline.

# Configure logging for this module
logger = logging.getLogger(__name__)
# __name__ evaluates to "backend.ingestion.loader"
# This means log messages from THIS file will be labeled
# with the module path — makes debugging much easier
# in large systems with many files.


def load_document(file_path: str) -> List[Document]:
    """
    Load a single document from a file path.
    
    Supports: PDF (.pdf), Plain Text (.txt)
    
    Args:
        file_path: Absolute or relative path to the file
        
    Returns:
        List of Document objects, one per page (PDF) or
        one per file (TXT)
        
    Raises:
        FileNotFoundError: if file doesn't exist
        ValueError: if file format is unsupported
    """
    
    # ── 1. Validate the file exists ──────────────────────
    path = Path(file_path)
    # Convert string to Path object for easier manipulation.
    # Path objects have useful properties: .suffix, .stem, .name
    
    if not path.exists():
        # Always validate inputs at the boundary of your system.
        # "Defensive programming" — don't assume callers are correct.
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # ── 2. Determine file type ───────────────────────────
    extension = path.suffix.lower()
    # .suffix returns ".pdf", ".txt", ".docx" etc.
    # .lower() normalizes "FILE.PDF" → ".pdf"
    # This prevents bugs where Windows users upload ".PDF"
    
    logger.info(f"Loading document: {path.name} (type: {extension})")
    # Structured log message — tells us WHAT file and WHAT type.
    # In production with 1000 concurrent users, these logs are
    # invaluable for debugging.
    
    # ── 3. Route to the correct loader ──────────────────
    if extension == ".pdf":
        documents = _load_pdf(file_path)
        
    elif extension == ".txt":
        documents = _load_text(file_path)
        
    else:
        # Fail with a descriptive error, not a cryptic exception
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported types: .pdf, .txt"
        )
    
    # ── 4. Enrich metadata ───────────────────────────────
    for doc in documents:
        # Add consistent metadata fields to EVERY document,
        # regardless of source type. This lets downstream
        # components always find these fields without
        # checking if they exist.
        doc.metadata["file_name"] = path.name
        # path.name → "tesla_manual.pdf" (just filename, no path)
        
        doc.metadata["file_type"] = extension
        # ".pdf" or ".txt" — useful for display in the UI
        
        doc.metadata["file_path"] = str(file_path)
        # Full path — useful for re-loading if needed
    
    logger.info(
        f"✅ Loaded {len(documents)} page(s) from '{path.name}'"
    )
    
    return documents
    # Returns List[Document] — the standardized format
    # that every other phase of the pipeline expects.


def _load_pdf(file_path: str) -> List[Document]:
    """
    Internal function to load a PDF using PyMuPDF.
    
    Why PyMuPDF (fitz) over alternatives?
    
    PyMuPDF advantages:
      - Extracts text WITH positional data (bounding boxes)
      - Handles multi-column layouts correctly  
      - Faster than pypdf (C library under the hood)
      - Can extract images, tables (future feature)
      - Handles corrupt PDFs more gracefully
      
    Alternative — PyPDFLoader:
      - Simpler but flattens multi-column text incorrectly
      - Loses page structure information
      
    Alternative — UnstructuredPDFLoader:
      - Best quality for complex PDFs
      - But requires Tesseract OCR install for scanned PDFs
      - Heavier setup, not ideal for learning projects
    """
    
    loader = PyMuPDFLoader(file_path)
    # Creates a loader object but does NOT read the file yet.
    # LangChain uses lazy loading — actual file I/O happens
    # only when you call .load()
    
    documents = loader.load()
    # Actually reads the PDF.
    # Returns one Document per PAGE.
    # A 50-page PDF → 50 Document objects.
    # Each Document's metadata contains:
    #   { "source": "path", "page": 0, "total_pages": 50, ... }
    
    return documents


def _load_text(file_path: str) -> List[Document]:
    """
    Internal function to load a plain text file.
    
    TextLoader returns ONE Document for the entire file.
    This is fine for short texts but for long ones,
    the chunker (Phase 3) will split it anyway.
    """
    
    loader = TextLoader(
        file_path,
        encoding="utf-8"
        # Always specify encoding explicitly.
        # Default varies by OS — Windows often uses cp1252,
        # which breaks on special characters (ë, ñ, etc.)
        # UTF-8 is the universal standard for text in 2024.
    )
    
    documents = loader.load()
    # Returns a List with ONE Document containing all text.
    
    return documents


def load_documents_from_directory(directory_path: str) -> List[Document]:
    """
    Load ALL supported documents from a directory.
    
    Useful for bulk ingestion — e.g., loading an entire
    folder of research papers at once.
    
    This uses DirectoryLoader which internally creates
    the right loader for each file type automatically.
    """
    
    all_documents = []
    # Start with empty list, accumulate documents
    
    directory = Path(directory_path)
    
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Not a directory: {directory_path}"
        )
    
    # Find all supported files in the directory
    supported_extensions = [".pdf", ".txt"]
    
    for ext in supported_extensions:
        # glob() searches for files matching a pattern.
        # "**/*.pdf" means: in this folder AND all subfolders
        # find every file ending in .pdf
        files = list(directory.glob(f"**/*{ext}"))
        
        for file_path in files:
            try:
                docs = load_document(str(file_path))
                all_documents.extend(docs)
                # extend() adds all items from docs into
                # all_documents (unlike append which would
                # add the list itself as a single item)
                
            except Exception as e:
                # Log the error but CONTINUE with other files.
                # This is "fault tolerance" — one bad file
                # shouldn't kill the entire ingestion job.
                logger.error(
                    f"❌ Failed to load {file_path.name}: {e}"
                )
                continue
                # Skip to the next file
    
    logger.info(
        f"📚 Total documents loaded from directory: "
        f"{len(all_documents)}"
    )
    
    return all_documents
