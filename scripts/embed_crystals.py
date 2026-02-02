#!/usr/bin/env python3
"""
Embed all crystals using Ollama's nomic-embed-text
==================================================
Stores embeddings in SQLite for vector search.
Checkpoints progress - can resume if interrupted.
"""

import sqlite3
import json
import requests
import numpy as np
from pathlib import Path
import time
import sys

DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"
BATCH_SIZE = 10  # Process 10 at a time for progress visibility

def get_embedding(text: str) -> bytes:
    """Get embedding from Ollama and return as bytes."""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": text[:8000]  # Limit text length
        }, timeout=30)
        response.raise_for_status()
        embedding = response.json()["embedding"]
        # Convert to numpy and then to bytes for storage
        arr = np.array(embedding, dtype=np.float32)
        return arr.tobytes()
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Get crystals without embeddings
    c.execute("""
        SELECT id, content FROM crystals
        WHERE embedding IS NULL AND content IS NOT NULL
        ORDER BY id
    """)
    pending = c.fetchall()

    total = len(pending)
    print(f"="*60)
    print(f"CRYSTAL EMBEDDING")
    print(f"="*60)
    print(f"Pending: {total} crystals")
    print(f"Model: {MODEL}")
    print(f"="*60)
    print()

    if total == 0:
        print("✓ All crystals already embedded!")
        return

    start_time = time.time()
    embedded = 0
    errors = 0

    for i, (crystal_id, content) in enumerate(pending):
        if not content or len(content.strip()) < 10:
            continue

        embedding = get_embedding(content)

        if embedding:
            c.execute("UPDATE crystals SET embedding = ? WHERE id = ?",
                     (embedding, crystal_id))
            embedded += 1
        else:
            errors += 1

        # Progress every 50
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = embedded / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate / 60 if rate > 0 else 0
            print(f"[{i+1}/{total}] Embedded: {embedded} | Errors: {errors} | {rate:.1f}/sec | ETA: {eta:.1f}min")
            conn.commit()  # Checkpoint

        # Commit every 100
        if (i + 1) % 100 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    print()
    print(f"="*60)
    print(f"COMPLETE")
    print(f"="*60)
    print(f"Embedded: {embedded}")
    print(f"Errors: {errors}")
    print(f"Time: {elapsed/60:.1f} minutes")
    print(f"Rate: {embedded/elapsed:.1f} crystals/sec")

if __name__ == "__main__":
    main()
