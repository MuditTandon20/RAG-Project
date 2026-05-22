# backend/ingestion/test_chunker.py

import sys
sys.path.append(".")

from backend.ingestion.loader import load_document
from backend.ingestion.chunker import chunk_documents

# Load a document first
docs = load_document("data/uploads/sample.pdf")
print(f"Pages loaded: {len(docs)}")

# Chunk it
chunks = chunk_documents(docs)
print(f"Chunks created: {len(chunks)}")

# Inspect the first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n{'='*50}")
    print(f"CHUNK {i+1}")
    print(f"{'='*50}")
    print(f"Content ({len(chunk.page_content)} chars):")
    print(chunk.page_content)
    print(f"\nMetadata: {chunk.metadata}")

# Check overlap is working
print(f"\n{'='*50}")
print("OVERLAP CHECK — end of chunk 1 vs start of chunk 2:")
print(f"End of chunk 1:   ...{chunks[0].page_content[-60:]!r}")
print(f"Start of chunk 2: {chunks[1].page_content[:60]!r}...")
# !r adds repr() formatting — shows \n as literal \n
# Helps you SEE the actual whitespace characters
