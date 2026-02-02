#!/usr/bin/env python3
"""
WiltonOS Crystal Embeddings
Embed all crystals for semantic search.

Usage:
    python wiltonos_embed.py embed          # Embed all crystals
    python wiltonos_embed.py search "query" # Semantic search
    python wiltonos_embed.py status         # Check progress
"""
import os
import sys
import json
import sqlite3
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

DB_PATH = Path.home() / "crystals_unified.db"
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE = 10  # Embed 10 at a time


def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text[:8000]},  # Limit text length
            timeout=30
        )
        if resp.ok:
            return resp.json().get("embedding", [])
    except Exception as e:
        print(f"Embedding error: {e}")
    return []


def ensure_embedding_table():
    """Create embeddings table if needed."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS crystal_embeddings (
            crystal_id INTEGER PRIMARY KEY,
            embedding BLOB,
            embedded_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_unembedded_crystals(limit: int = None) -> List[Tuple[int, str]]:
    """Get crystals that need embedding."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    query = """
        SELECT c.id, c.content
        FROM crystals c
        LEFT JOIN crystal_embeddings e ON c.id = e.crystal_id
        WHERE e.crystal_id IS NULL
        AND c.content IS NOT NULL
        AND LENGTH(c.content) > 50
    """
    if limit:
        query += f" LIMIT {limit}"

    c.execute(query)
    results = c.fetchall()
    conn.close()
    return results


def save_embedding(crystal_id: int, embedding: List[float]):
    """Save embedding to database."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Convert to bytes for storage
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

    c.execute("""
        INSERT OR REPLACE INTO crystal_embeddings (crystal_id, embedding, embedded_at)
        VALUES (?, ?, ?)
    """, (crystal_id, embedding_bytes, datetime.now().isoformat()))

    conn.commit()
    conn.close()


def embed_all():
    """Embed all crystals."""
    ensure_embedding_table()

    crystals = get_unembedded_crystals()
    total = len(crystals)

    if total == 0:
        print("All crystals already embedded!")
        return

    print(f"Embedding {total:,} crystals...")
    print(f"Estimated time: {total * 0.5 / 60:.1f} minutes")
    print()

    start = datetime.now()
    done = 0
    errors = 0

    for crystal_id, content in crystals:
        embedding = get_embedding(content)

        if embedding:
            save_embedding(crystal_id, embedding)
            done += 1
        else:
            errors += 1

        # Progress every 100
        if (done + errors) % 100 == 0:
            elapsed = (datetime.now() - start).total_seconds()
            rate = (done + errors) / elapsed * 60
            remaining = (total - done - errors) / rate if rate > 0 else 0
            print(f"  {done + errors}/{total} ({done} done, {errors} errors) - {rate:.1f}/min - ~{remaining:.0f} min left")

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\nComplete: {done:,} embedded, {errors} errors in {elapsed:.1f} minutes")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def semantic_search(query: str, limit: int = 20) -> List[Tuple[int, str, float]]:
    """Search crystals by semantic similarity."""
    # Get query embedding
    query_embedding = get_embedding(query)
    if not query_embedding:
        print("Failed to embed query")
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)

    # Get all embeddings
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT e.crystal_id, c.content, e.embedding
        FROM crystal_embeddings e
        JOIN crystals c ON e.crystal_id = c.id
    """)

    results = []
    for crystal_id, content, embedding_bytes in c.fetchall():
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        similarity = cosine_similarity(query_vec, embedding)
        results.append((crystal_id, content, similarity))

    conn.close()

    # Sort by similarity
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:limit]


def show_status():
    """Show embedding status."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM crystals")
    total = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM crystal_embeddings")
    embedded = c.fetchone()[0]

    conn.close()

    pct = embedded * 100 / total if total > 0 else 0
    print(f"Crystals: {total:,}")
    print(f"Embedded: {embedded:,} ({pct:.1f}%)")
    print(f"Remaining: {total - embedded:,}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "embed":
        embed_all()
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: python wiltonos_embed.py search 'your query'")
            return
        query = " ".join(sys.argv[2:])
        print(f"Searching for: {query}\n")
        results = semantic_search(query, limit=10)
        for crystal_id, content, score in results:
            print(f"[{score:.3f}] Crystal #{crystal_id}")
            print(f"  {content[:200]}...")
            print()
    elif cmd == "status":
        show_status()
    else:
        print(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
