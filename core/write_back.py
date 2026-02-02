"""
Crystal Write-Back Module
=========================
Not just READ crystals - WRITE new insights back.
The system learns, evolves, remembers.

Usage:
    from write_back import CrystalWriter
    writer = CrystalWriter()
    writer.store_insight(content, source='openwebui', emotion='clarity')
    writer.store_conversation_insight(query, response)
"""

import sqlite3
import json
import hashlib
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List


class CrystalWriter:
    """Write new insights back to the crystal database."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or "/home/zews/wiltonos/data/crystals_unified.db")
        self.ollama_url = "http://localhost:11434"
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure we have the tables we need."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # Auto-generated insights table
        c.execute("""
            CREATE TABLE IF NOT EXISTS auto_insights (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT DEFAULT 'wiltonos',
                emotion TEXT,
                topology TEXT,
                glyphs TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                conversation_id TEXT,
                query_hash TEXT,
                UNIQUE(query_hash)
            )
        """)

        # Session memories table
        c.execute("""
            CREATE TABLE IF NOT EXISTS session_memories (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                platform TEXT,
                key_insight TEXT,
                emotional_thread TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Ollama."""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text[:4000]},
                timeout=15
            )
            if resp.ok:
                return np.array(resp.json().get("embedding", []), dtype=np.float32)
        except:
            pass
        return None

    def _hash_content(self, content: str) -> str:
        """Create hash to prevent duplicates."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def store_insight(
        self,
        content: str,
        source: str = 'wiltonos',
        emotion: str = None,
        topology: str = None,
        glyphs: List[str] = None,
        conversation_id: str = None
    ) -> bool:
        """
        Store a new insight crystal.

        Args:
            content: The insight text
            source: Where it came from (openwebui, terminal, etc.)
            emotion: Detected emotion
            topology: Detected topology (grief, love, etc.)
            glyphs: List of detected glyphs
            conversation_id: Optional conversation ID for linking

        Returns:
            True if stored, False if duplicate or error
        """
        query_hash = self._hash_content(content)

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        try:
            c.execute("""
                INSERT INTO auto_insights
                (content, source, emotion, topology, glyphs, conversation_id, query_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                content,
                source,
                emotion,
                topology,
                json.dumps(glyphs) if glyphs else None,
                conversation_id,
                query_hash
            ))
            insight_id = c.lastrowid
            conn.commit()

            # Try to embed it (non-blocking - embedding can happen later)
            try:
                vec = self._get_embedding(content)
                if vec is not None:
                    # Store in separate table for auto insights
                    c.execute("""
                        CREATE TABLE IF NOT EXISTS auto_insight_embeddings (
                            insight_id INTEGER PRIMARY KEY,
                            embedding BLOB,
                            model TEXT DEFAULT 'nomic-embed-text'
                        )
                    """)
                    c.execute("""
                        INSERT OR REPLACE INTO auto_insight_embeddings
                        (insight_id, embedding, model)
                        VALUES (?, ?, 'nomic-embed-text')
                    """, (insight_id, vec.tobytes()))
                    conn.commit()
            except Exception:
                pass  # Embedding failure shouldn't block storage

            conn.close()
            return True

        except sqlite3.IntegrityError:
            # Duplicate
            conn.close()
            return False
        except Exception as e:
            conn.close()
            return False

    def store_conversation_insight(
        self,
        query: str,
        response: str,
        source: str = 'wiltonos',
        extract_insight: bool = True
    ) -> Dict:
        """
        Store insight from a conversation exchange.

        If extract_insight is True, we try to extract the key insight.
        Otherwise we store the raw exchange.

        Returns dict with stored status and insight_id if successful.
        """
        # For now, simple extraction: if response has a clear insight marker
        # In future: use an LLM to extract the key insight

        result = {'stored': False, 'insight_id': None}

        # Simple heuristic: responses with certain patterns are insights
        insight_markers = [
            'you said', 'you mentioned', 'i notice', 'what stands out',
            'the thread between', 'what i\'m hearing', 'something shifted'
        ]

        response_lower = response.lower()

        # Check if this feels like an insight worth storing
        is_insight = any(marker in response_lower for marker in insight_markers)

        # Also store if user query was a breakthrough moment
        breakthrough_markers = ['just realized', 'it hit me', 'i see now', 'oh', 'wow']
        is_breakthrough = any(marker in query.lower() for marker in breakthrough_markers)

        if is_insight or is_breakthrough:
            # Create a condensed crystal
            if is_breakthrough:
                content = f"[Breakthrough] {query}"
            else:
                # Extract first 500 chars of response as insight
                content = response[:500]

            stored = self.store_insight(
                content=content,
                source=source,
                conversation_id=self._hash_content(query + response)
            )

            if stored:
                result['stored'] = True

        return result

    def get_recent_insights(self, limit: int = 10) -> List[Dict]:
        """Get most recent auto-generated insights."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT id, content, source, emotion, topology, created_at
            FROM auto_insights
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))

        results = []
        for row in c.fetchall():
            results.append({
                'id': row[0],
                'content': row[1],
                'source': row[2],
                'emotion': row[3],
                'topology': row[4],
                'created_at': row[5]
            })

        conn.close()
        return results

    def count_insights(self) -> int:
        """Count total auto-generated insights."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM auto_insights")
        count = c.fetchone()[0]
        conn.close()
        return count


# CLI for testing
if __name__ == "__main__":
    writer = CrystalWriter()

    print(f"=== Crystal Writer ===")
    print(f"Database: {writer.db_path}")
    print(f"Auto-insights stored: {writer.count_insights()}")

    # Test storing an insight
    test_insight = "The field between us knows what's needed. Trust it."
    stored = writer.store_insight(
        content=test_insight,
        source='test',
        emotion='presence',
        topology='spiritual'
    )
    print(f"\nTest insight stored: {stored}")

    print(f"\nRecent insights:")
    for insight in writer.get_recent_insights(5):
        print(f"  - [{insight['source']}] {insight['content'][:60]}...")
