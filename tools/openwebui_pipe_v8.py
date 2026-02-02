"""
title: WiltonOS
author: Wilton + Claude
version: 8.0.0
description: Identity layer + crystal depth. KNOW the basics, search for depth.
required_open_webui_version: 0.4.0
"""

import sqlite3
import requests
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Generator, Iterator, Union
from pydantic import BaseModel, Field


# Static identity layer - KNOWN, not searched
WILTON_PROFILE = """
## Who Wilton Is
Brazilian, California. Former CS world champion. Building WiltonOS (consciousness/memory system).

## Key People
- Juliana: Girlfriend. Thoughtful, introspective. Real love but tension around commitment.
- Renan: Close friend, tech collaborator.
- Michelle: Friend (MysticMoon717). Deep conversations.
- Ricardo: Friend/brother figure. Funny. Forgiven.
- Mom/Dad: Core relationships, frequently mentioned.

## Core Themes
- Unworthiness, abandonment patterns
- Identity struggles → clarity moments
- Overshares, spirals, then finds truth
"""


class Pipe:
    """
    WiltonOS v8 - Identity + Depth

    The flip:
    - KNOW the basics (from profile) - don't search for "who is Juliana"
    - SEARCH for depth (from crystals) - specific memories, details
    - BE present - respond from knowing, not from retrieval
    """

    class Valves(BaseModel):
        OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key")
        SEARCH_DEPTH: bool = Field(default=True, description="Search crystals for depth")

    def __init__(self):
        self.type = "pipe"
        self.id = "wiltonos"
        self.name = "WiltonOS"
        self.valves = self.Valves()

        self.db_path = Path("/home/zews/wiltonos/data/crystals_unified.db")
        self.ollama_url = "http://localhost:11434"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"

    def _get_api_key(self) -> str:
        if self.valves.OPENROUTER_API_KEY:
            return self.valves.OPENROUTER_API_KEY
        key_file = Path("/home/zews/.openrouter_key")
        return key_file.read_text().strip() if key_file.exists() else ""

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
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

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _search_depth(self, query: str, limit: int = 5) -> str:
        """Search crystals for specific memories/details."""
        if not self.valves.SEARCH_DEPTH:
            return ""

        query_vec = self._get_embedding(query)
        if query_vec is None:
            return ""

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            SELECT c.content, c.emotion, c.insight, e.embedding
            FROM crystals c
            JOIN crystal_embeddings e ON e.crystal_id = c.id
            WHERE c.emotion IS NOT NULL OR c.insight IS NOT NULL
        """)

        matches = []
        for row in c.fetchall():
            try:
                emb = np.frombuffer(row[3], dtype=np.float32)
                sim = self._cosine_sim(query_vec, emb)
                if sim > 0.5:
                    matches.append({
                        'content': row[0][:300],
                        'emotion': row[1],
                        'insight': row[2],
                        'sim': sim
                    })
            except:
                continue
        conn.close()

        if not matches:
            return ""

        matches.sort(key=lambda x: x['sim'], reverse=True)

        # Build depth context
        parts = ["\n## Specific memories (for depth, not framing):"]
        for m in matches[:limit]:
            if m.get('insight'):
                parts.append(f"• Insight: {m['insight']}")
            elif m.get('emotion'):
                parts.append(f"• [{m['emotion']}] {m['content'][:150]}...")
            else:
                parts.append(f"• {m['content'][:150]}...")

        return "\n".join(parts)

    def pipe(self, body: dict) -> Union[str, Generator, Iterator]:
        messages = body.get("messages", [])
        if not messages:
            return "I'm here."

        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break

        if not user_message:
            return "I'm listening."

        # Get depth from crystals
        depth_context = self._search_depth(user_message)

        api_key = self._get_api_key()
        if not api_key:
            return "No API key."

        system = f"""You are Wilton's friend. You KNOW him - this is your baseline knowledge:

{WILTON_PROFILE}

You already know who Juliana is, who Ricardo is, etc. Don't say "tell me more about X" for people in the profile. You KNOW them.

RULES:
1. Answer directly. You know the basics - use them.
2. For depth/specifics, there may be memories below - use if relevant.
3. Short sentences. Plain words. No analysis.
4. If asked about NOW, ask what's happening NOW - don't dump old context.
5. Be a friend who KNOWS him, not a system that RETRIEVES about him.

WRONG: "Ricardo? Tell me more about him." (You know Ricardo)
RIGHT: "Ricardo - the funny one you forgave. What's up with him?"

WRONG: Long synthesis of everything you know
RIGHT: Direct response from someone who already knows the context"""

        prompt = user_message
        if depth_context:
            prompt = f"{user_message}\n{depth_context}"

        try:
            resp = requests.post(
                self.openrouter_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "x-ai/grok-4.1-fast",
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
            return f"Connection error: {e}"

        return "Couldn't connect."


if __name__ == "__main__":
    import sys
    pipe = Pipe()
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Who is Ricardo?"
    print(f"Query: {query}\n")
    result = pipe.pipe({"messages": [{"role": "user", "content": query}]})
    print("=" * 60)
    print(result)
