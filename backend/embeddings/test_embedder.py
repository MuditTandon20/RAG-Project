# backend/embeddings/test_embedder.py

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.embeddings.embedder import EmbeddingManager

em = EmbeddingManager()

# ── Test 1: Embed a query ─────────────────────────────
query_vector = em.embed_query("What is the battery range of Model X?")
print(f"Query vector dimensions: {len(query_vector)}")
print(f"First 5 values: {query_vector[:5]}")

# ── Test 2: Semantic similarity intuition ─────────────
sentences = [
    "The EV has a range of 350 miles on full charge.",  # Very relevant
    "Electric vehicles use lithium ion batteries.",      # Somewhat relevant
    "The recipe calls for 2 cups of flour.",            # Irrelevant
]

query = "How far can an electric car travel?"
q_vec = em.embed_query(query)

print(f"\nQuery: '{query}'\n")

for sentence in sentences:
    s_vec = em.embed_query(sentence)
    score = em.compute_similarity(q_vec, s_vec)
    bar = "█" * int(score * 30)  # Visual bar
    print(f"Score: {score:.4f} {bar}")
    print(f"Text:  {sentence}\n")
