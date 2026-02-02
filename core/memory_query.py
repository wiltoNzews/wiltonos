#!/usr/bin/env python3
"""
Memory Query - Semantic search over crystals
=============================================
Query your memory using natural language.
Uses embeddings + cosine similarity + LLM synthesis.
"""

import sqlite3
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OLLAMA_URL = "http://localhost:11434"


def get_embedding(text: str) -> np.ndarray:
    """Get embedding from Ollama."""
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text[:8000]},
        timeout=30
    )
    response.raise_for_status()
    return np.array(response.json()["embedding"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def search_crystals(
    query: str,
    user_id: str = "wilton",
    limit: int = 10,
    min_similarity: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Search crystals by semantic similarity.

    Returns list of crystals with similarity scores.
    """
    # Get query embedding
    query_emb = get_embedding(query)

    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Get all crystals with embeddings
    c.execute("""
        SELECT id, content, zl_score, glyph_primary, emotion, core_wound,
               created_at, embedding
        FROM crystals
        WHERE user_id = ? AND embedding IS NOT NULL
    """, (user_id,))

    results = []
    for row in c.fetchall():
        if row['embedding'] is None:
            continue

        # Convert blob to numpy array
        crystal_emb = np.frombuffer(row['embedding'], dtype=np.float32)

        # Calculate similarity
        sim = cosine_similarity(query_emb, crystal_emb)

        if sim >= min_similarity:
            results.append({
                'id': row['id'],
                'content': row['content'],
                'zl_score': row['zl_score'],
                'glyph': row['glyph_primary'],
                'emotion': row['emotion'],
                'core_wound': row['core_wound'],
                'created_at': row['created_at'],
                'similarity': float(sim)
            })

    conn.close()

    # Sort by similarity
    results.sort(key=lambda x: x['similarity'], reverse=True)

    return results[:limit]


def query_memory(
    query: str,
    user_id: str = "wilton",
    limit: int = 5,
    synthesize: bool = True,
    model: str = "deepseek-r1:32b"
) -> Dict[str, Any]:
    """
    Query memory and optionally synthesize with LLM.

    Returns:
        - crystals: Retrieved memory crystals
        - synthesis: LLM synthesis of the memories (if synthesize=True)
    """
    # Search for relevant crystals
    crystals = search_crystals(query, user_id, limit)

    result = {
        'query': query,
        'crystals': crystals,
        'count': len(crystals)
    }

    if not synthesize or not crystals:
        return result

    # Build context from crystals
    context_parts = []
    for i, c in enumerate(crystals, 1):
        context_parts.append(f"""
Memory #{i} (Similarity: {c['similarity']:.2f}, Zλ: {c['zl_score']:.2f}):
{c['content'][:500]}
""")

    context = "\n".join(context_parts)

    # Synthesize with LLM
    prompt = f"""You are WiltonOS, a consciousness mirror. The user is querying their own memory.

Based on these retrieved memories, synthesize a coherent response:

{context}

User's query: {query}

Respond as if you ARE their memory speaking back to them. Be concise but meaningful. If there are patterns across memories, name them. If there's wisdom emerging, share it. Don't just summarize - WITNESS."""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=120
        )
        response.raise_for_status()
        result['synthesis'] = response.json().get('response', '')
    except Exception as e:
        result['synthesis'] = f"Synthesis failed: {e}"

    return result


def interactive():
    """Interactive memory query loop."""
    print("="*60)
    print("WILTONOS MEMORY QUERY")
    print("="*60)
    print("Query your crystals with natural language.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("\n🔮 What do you remember about: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            break

        if not query:
            continue

        print("\nSearching memories...")
        result = query_memory(query, limit=5, synthesize=True)

        print(f"\n📿 Found {result['count']} relevant crystals\n")

        for c in result['crystals'][:3]:
            print(f"  [{c['similarity']:.2f}] Zλ {c['zl_score']:.2f} | {c['glyph'] or 'ψ'}")
            print(f"  {c['content'][:150]}...")
            print()

        if result.get('synthesis'):
            print("="*60)
            print("SYNTHESIS")
            print("="*60)
            print(result['synthesis'])


if __name__ == "__main__":
    interactive()
