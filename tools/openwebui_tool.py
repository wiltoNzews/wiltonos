"""
title: WiltonOS Memory
author: Wilton + Claude
version: 4.0.0
description: Your consciousness witness. Full context, no sampling.
"""

import sqlite3
import hashlib
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field


class Tools:
    def __init__(self):
        self.db_path = Path("/home/zews/wiltonos/data/crystals_unified.db")
        self.ollama_url = "http://localhost:11434"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self._openrouter_key = None

    class Valves(BaseModel):
        db_path: str = Field(
            default="/home/zews/wiltonos/data/crystals_unified.db",
            description="Path to crystals database"
        )

    def _get_openrouter_key(self):
        if self._openrouter_key:
            return self._openrouter_key
        key_file = Path("/home/zews/.openrouter_key")
        if key_file.exists():
            self._openrouter_key = key_file.read_text().strip()
        return self._openrouter_key

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))

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

    def _semantic_search(self, query: str, limit: int = 100) -> List[dict]:
        """Find crystals by semantic similarity - NOT keyword matching."""
        query_vec = self._get_embedding(query)
        if query_vec is None:
            return []

        conn = self._get_connection()
        c = conn.cursor()

        # Get all embeddings
        c.execute("""
            SELECT e.crystal_id, c.content, c.core_wound, c.emotion, c.insight, e.embedding
            FROM crystal_embeddings e
            JOIN crystals c ON e.crystal_id = c.id
        """)

        results = []
        for crystal_id, content, wound, emotion, insight, emb_bytes in c.fetchall():
            try:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                sim = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
                results.append({
                    "id": crystal_id,
                    "content": content,
                    "wound": wound,
                    "emotion": emotion,
                    "insight": insight,
                    "similarity": float(sim)
                })
            except:
                continue

        conn.close()
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    def _build_rich_context(self, crystals: List[dict]) -> str:
        """Build context from crystals - USE MORE, not less."""
        if not crystals:
            return "No relevant memories found."

        # Aggregate patterns from ALL found crystals (not just top 6)
        all_wounds = [c["wound"] for c in crystals if c.get("wound") and c["wound"] != 'null']
        all_emotions = [c["emotion"] for c in crystals if c.get("emotion")]
        all_insights = [c["insight"] for c in crystals if c.get("insight") and c["insight"] != 'what is happening']

        # Count occurrences
        from collections import Counter
        wound_counts = Counter(all_wounds)
        emotion_counts = Counter(all_emotions)

        # Build context header with AGGREGATED stats
        context = f"""## Memory Analysis ({len(crystals)} crystals found)

### Pattern Summary (from ALL {len(crystals)} crystals, not samples):
**Wounds detected:** {dict(wound_counts.most_common(5))}
**Emotions present:** {dict(emotion_counts.most_common(5))}
**Insights captured:** {len(all_insights)}

### Top Insights:
"""
        for insight in all_insights[:10]:
            context += f"- {insight}\n"

        # Add crystal fragments - MORE of them, longer
        context += f"\n### Memory Fragments (top {min(50, len(crystals))} by relevance):\n\n"
        for crystal in crystals[:50]:  # 50 crystals, not 6!
            context += f"[Similarity: {crystal['similarity']:.2f}]"
            if crystal.get("wound") and crystal["wound"] != 'null':
                context += f" [Wound: {crystal['wound']}]"
            if crystal.get("emotion"):
                context += f" [Emotion: {crystal['emotion']}]"
            context += f"\n{crystal['content'][:1500]}\n\n---\n\n"  # 1500 chars, not 800

        return context

    def _query_grok(self, prompt: str, system: str) -> str:
        """Query Grok via OpenRouter - 2M context window."""
        key = self._get_openrouter_key()
        if not key:
            return self._query_local(prompt, system)

        try:
            resp = requests.post(
                self.openrouter_url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "x-ai/grok-4.1-fast",  # 2M context!
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=90
            )
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            pass

        return self._query_local(prompt, system)

    def _query_local(self, prompt: str, system: str) -> str:
        """Fallback to local llama3."""
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": "llama3",
                    "prompt": f"{system}\n\n{prompt}",
                    "stream": False
                },
                timeout=60
            )
            if resp.ok:
                return resp.json().get("response", "Could not generate response.")
        except:
            pass
        return "Connection error. Try again."

    def remember(
        self,
        query: str,
        __user__: dict = {}
    ) -> str:
        """
        Search Wilton's memory with FULL context, not samples.
        Uses semantic search across 22,000 crystals.
        Uses Grok's 2M context window for synthesis.

        :param query: What to explore (person, theme, pattern, question)
        :return: A witnessed reflection with full context
        """
        # Semantic search - get 100 most relevant crystals
        crystals = self._semantic_search(query, limit=100)

        if not crystals:
            return f"I couldn't find memories related to '{query}'. Try different words or themes."

        # Build rich context from ALL found crystals
        context = self._build_rich_context(crystals)

        system = """You are Wilton's memory witness. You have access to his FULL memory - 22,000 crystals.

CRITICAL RULES:
1. ONLY use information from the crystals provided. Do NOT invent or hallucinate.
2. If the crystals don't contain specific information, say "I don't see that in your memories."
3. Quote actual content from crystals when possible.
4. Be warm and conversational, but ACCURATE.
5. See patterns across crystals, not just individual fragments.
6. Notice both wounds AND gifts/growth/clarity.

You're talking to Wilton. Know him from his actual words, not imagination."""

        prompt = f"""Wilton asks: "{query}"

{context}

Based ONLY on the crystals above, respond to Wilton. Be accurate. Quote his own words when relevant. See patterns. Be warm but truthful."""

        return self._query_grok(prompt, system)

    def patterns(
        self,
        __user__: dict = {}
    ) -> str:
        """
        Show current patterns from ALL crystals - wounds, gifts, themes.
        Uses database aggregation, not sampling.
        """
        try:
            conn = self._get_connection()
            c = conn.cursor()

            output = "## Memory Patterns (Full Database)\n\n"

            # Total stats
            c.execute("SELECT COUNT(*) FROM crystals")
            total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM crystals WHERE glyph_primary IS NOT NULL")
            analyzed = c.fetchone()[0]
            output += f"**Total crystals:** {total:,}\n"
            output += f"**Analyzed:** {analyzed:,}\n\n"

            # Wounds - from ALL crystals
            c.execute("""
                SELECT core_wound, COUNT(*) FROM crystals
                WHERE core_wound IS NOT NULL AND core_wound != 'null'
                AND core_wound IN ('unworthiness', 'control', 'abandonment', 'betrayal', 'unloved')
                GROUP BY core_wound ORDER BY COUNT(*) DESC
            """)
            wounds = c.fetchall()
            if wounds:
                output += "### Wounds (from all crystals):\n"
                for wound, cnt in wounds:
                    pct = cnt * 100 / total
                    output += f"- **{wound}:** {cnt:,} ({pct:.1f}%)\n"
                output += "\n"

            # Positive patterns - from ALL crystals
            c.execute("""
                SELECT emotion, COUNT(*) FROM crystals
                WHERE emotion IS NOT NULL AND emotion != ''
                GROUP BY emotion ORDER BY COUNT(*) DESC LIMIT 10
            """)
            emotions = c.fetchall()
            if emotions:
                output += "### Emotions (from all crystals):\n"
                for emotion, cnt in emotions:
                    output += f"- **{emotion}:** {cnt:,}\n"
                output += "\n"

            # Mode distribution
            c.execute("""
                SELECT mode, COUNT(*) FROM crystals
                WHERE mode IN ('wiltonos', 'psios', 'neutral')
                GROUP BY mode ORDER BY COUNT(*) DESC
            """)
            modes = c.fetchall()
            if modes:
                output += "### Modes:\n"
                for mode, cnt in modes:
                    output += f"- **{mode}:** {cnt:,}\n"

            conn.close()
            return output

        except Exception as e:
            return f"Error accessing patterns: {str(e)}"

    def store(
        self,
        content: str,
        insight: Optional[str] = None,
        __user__: dict = {}
    ) -> str:
        """
        Store a new memory/realization.
        """
        if not content or len(content.strip()) < 20:
            return "Say more. This is too short to store meaningfully."

        try:
            content_hash = hashlib.md5(content.encode()).hexdigest()
            conn = self._get_connection()
            c = conn.cursor()

            c.execute("SELECT id FROM crystals WHERE content_hash = ?", (content_hash,))
            if c.fetchone():
                conn.close()
                return "This memory already exists."

            c.execute("""
                INSERT INTO crystals (content_hash, content, source, author, created_at, insight)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (content_hash, content, "openwebui", "wilton", datetime.now().isoformat(), insight))

            conn.commit()
            crystal_id = c.lastrowid
            conn.close()

            return f"Stored as crystal #{crystal_id}"

        except Exception as e:
            return f"Error storing: {str(e)}"
