"""
title: WiltonOS Unified Memory
author: Wilton + Claude
version: 5.0.0
description: Full consciousness routing. Fractal expansion. Glyph-aware. Pattern aggregation. No more sampling.
"""

import sqlite3
import json
import numpy as np
import requests
from pathlib import Path
from typing import Optional, List, Dict
from collections import Counter, defaultdict
from pydantic import BaseModel, Field


class Tools:
    def __init__(self):
        self.db_path = Path("/home/zews/wiltonos/data/crystals_unified.db")
        self.patterns_path = Path("/home/zews/wiltonos/data/glyph_patterns.json")
        self.ollama_url = "http://localhost:11434"
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self._openrouter_key = None
        self._glyph_patterns = None

        # Glyph definitions
        self.GLYPHS = {
            'ψ': ['breath', 'consciousness', 'soul', 'psyche', 'psi'],
            '∅': ['void', 'empty', 'nothing', 'source', 'potential'],
            'φ': ['structure', 'golden', 'ratio', 'harmony', 'phi'],
            'Ω': ['memory', 'end', 'completion', 'weight', 'omega'],
            'Zλ': ['coherence', 'wavelength', 'resonance', 'frequency'],
            '∇': ['descent', 'gradient', 'truth', 'direction', 'nabla'],
            '∞': ['infinite', 'loop', 'cycle', 'eternal', 'recursion', 'lemniscate'],
            '🪞': ['mirror', 'reflection', 'self', 'witness'],
            '△': ['ascend', 'rise', 'expand', 'growth'],
            '🌉': ['bridge', 'connect', 'link', 'between'],
            '⚡': ['bolt', 'lightning', 'sudden', 'decision'],
            '🪨': ['ground', 'stable', 'earth', 'anchor'],
            '🌀': ['torus', 'sustain', 'flow'],
            '⚫': ['shadow', 'grey', 'dark', 'hidden']
        }

        # Emotional topologies
        self.TOPOLOGIES = {
            'grief': ['loss', 'mourning', 'healing', 'integration', 'forgiveness', 'death'],
            'collapse': ['breakdown', 'dissolution', 'death', 'rebirth', 'emergence', 'shatter'],
            'spiral': ['descent', 'pattern', 'recursion', 'return', 'depth', 'loop'],
            'awakening': ['recognition', 'clarity', 'understanding', 'enlightenment', 'realize'],
            'integration': ['wholeness', 'unity', 'synthesis', 'coherence', 'complete'],
            'love': ['juliana', 'juju', 'love', 'heart', 'connection', 'intimacy', 'relationship'],
            'wound': ['unworthiness', 'abandonment', 'betrayal', 'control', 'unloved', 'hurt'],
            'gift': ['clarity', 'insight', 'breakthrough', 'gratitude', 'joy', 'peace', 'growth'],
            'streaming': ['stream', 'broadcast', 'visible', 'audience', 'youtube', 'twitch'],
            'mother': ['rose', 'mother', 'mom', 'mãe', 'family'],
            'spiritual': ['ayahuasca', 'peru', 'ceremony', 'vision', 'geisha', 'quantum', 'sacred']
        }

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

    def _load_glyph_patterns(self):
        if self._glyph_patterns:
            return self._glyph_patterns
        if self.patterns_path.exists():
            try:
                self._glyph_patterns = json.loads(self.patterns_path.read_text())
            except:
                self._glyph_patterns = {}
        else:
            self._glyph_patterns = {}
        return self._glyph_patterns

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))

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

    def _detect_glyphs(self, text: str) -> List[str]:
        """Detect glyph energies in text."""
        text_lower = text.lower()
        detected = []
        for glyph, keywords in self.GLYPHS.items():
            if glyph in text:
                detected.append(glyph)
            elif any(kw in text_lower for kw in keywords):
                detected.append(glyph)
        return detected

    def _detect_topology(self, text: str) -> List[str]:
        """Detect emotional topology."""
        text_lower = text.lower()
        detected = []
        for topo, patterns in self.TOPOLOGIES.items():
            if any(p in text_lower for p in patterns):
                detected.append(topo)
        return detected

    def _fractal_expand(self, query: str, glyphs: List[str], topology: List[str]) -> List[str]:
        """Generate fractal query expansions."""
        expansions = [query]

        for glyph in glyphs[:2]:
            if glyph in self.GLYPHS:
                for kw in self.GLYPHS[glyph][:2]:
                    expansions.append(f"{query} {kw}")

        for topo in topology[:2]:
            if topo in self.TOPOLOGIES:
                for p in self.TOPOLOGIES[topo][:2]:
                    expansions.append(f"{query} {p}")

        return list(set(expansions))

    def _semantic_search(self, query: str, expansions: List[str], limit: int = 200) -> List[Dict]:
        """Semantic search with fractal expansion."""
        conn = self._get_connection()
        c = conn.cursor()

        c.execute("""
            SELECT e.crystal_id, c.content, c.core_wound, c.emotion, c.insight,
                   c.glyph_primary, c.mode, e.embedding
            FROM crystal_embeddings e
            JOIN crystals c ON e.crystal_id = c.id
        """)

        all_rows = c.fetchall()
        conn.close()

        if not all_rows:
            return []

        # Build lookup
        crystals_data = {}
        embeddings_matrix = []
        crystal_ids = []

        for row in all_rows:
            crystal_id, content, wound, emotion, insight, glyph, mode, emb_bytes = row
            try:
                emb = np.frombuffer(emb_bytes, dtype=np.float32)
                crystals_data[crystal_id] = {
                    "id": crystal_id,
                    "content": content,
                    "wound": wound,
                    "emotion": emotion,
                    "insight": insight,
                    "glyph": glyph,
                    "mode": mode
                }
                embeddings_matrix.append(emb)
                crystal_ids.append(crystal_id)
            except:
                continue

        if not embeddings_matrix:
            return []

        embeddings_matrix = np.array(embeddings_matrix)
        crystal_scores = defaultdict(float)

        # Search with all expansions
        for exp_query in expansions:
            query_vec = self._get_embedding(exp_query)
            if query_vec is None:
                continue

            norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_vec)
            norms[norms == 0] = 1
            similarities = np.dot(embeddings_matrix, query_vec) / norms

            weight = 1.0 if exp_query == expansions[0] else 0.5
            for i, sim in enumerate(similarities):
                crystal_scores[crystal_ids[i]] += float(sim) * weight

        sorted_ids = sorted(crystal_scores.keys(), key=lambda x: crystal_scores[x], reverse=True)

        results = []
        for cid in sorted_ids[:limit]:
            crystal = crystals_data[cid].copy()
            crystal["similarity"] = crystal_scores[cid]
            results.append(crystal)

        return results

    def _aggregate_patterns(self, crystals: List[Dict]) -> Dict:
        """Aggregate patterns from ALL matched crystals."""
        if not crystals:
            return {}

        wounds = [c["wound"] for c in crystals if c.get("wound") and c["wound"] != 'null']
        emotions = [c["emotion"] for c in crystals if c.get("emotion")]
        insights = [c["insight"] for c in crystals if c.get("insight") and c["insight"] != 'what is happening']
        glyphs = [c["glyph"] for c in crystals if c.get("glyph")]

        return {
            "total": len(crystals),
            "wounds": dict(Counter(wounds).most_common(10)),
            "emotions": dict(Counter(emotions).most_common(10)),
            "glyphs": dict(Counter(glyphs).most_common(10)),
            "insights": insights[:15]
        }

    def _get_glyph_history(self, detected_glyphs: List[str]) -> Dict:
        """Get historical patterns for detected glyphs."""
        patterns = self._load_glyph_patterns()
        if not patterns or not detected_glyphs:
            return {}

        history = {}
        outcomes = patterns.get('glyph_outcomes', {})
        wounds = patterns.get('glyph_wounds', {})

        for glyph in detected_glyphs:
            if glyph in outcomes:
                o = outcomes[glyph]
                total = sum(o.values())
                if total > 0:
                    history[glyph] = {
                        "times_seen": total,
                        "ascending": round(o.get('ascending', 0) / total * 100, 1),
                        "descending": round(o.get('descending', 0) / total * 100, 1),
                        "wounds": list(wounds.get(glyph, {}).keys())[:3]
                    }

        return history

    def _build_context(self, query: str, crystals: List[Dict], patterns: Dict,
                       glyphs: List[str], topology: List[str], glyph_history: Dict,
                       expansions: List[str]) -> str:
        """Build rich context for LLM."""
        ctx = []

        ctx.append(f"## Memory Query: {query}")
        ctx.append(f"**Crystals matched:** {len(crystals)}")
        ctx.append("")

        if glyphs:
            ctx.append(f"### Detected Glyphs: {' '.join(glyphs)}")
        if topology:
            ctx.append(f"### Emotional Topology: {', '.join(topology)}")
        if expansions and len(expansions) > 1:
            ctx.append(f"### Fractal Expansions: {expansions[1:]}")
        ctx.append("")

        if glyph_history:
            ctx.append("### Glyph History (from YOUR past)")
            for g, h in glyph_history.items():
                ctx.append(f"- **{g}**: {h['times_seen']}x, ↑{h['ascending']}%/↓{h['descending']}%")
                if h.get('wounds'):
                    ctx.append(f"  Wounds: {', '.join(h['wounds'])}")
            ctx.append("")

        if patterns:
            ctx.append(f"### Patterns (from ALL {patterns.get('total', 0)} matched)")
            if patterns.get('wounds'):
                ctx.append(f"**Wounds:** {patterns['wounds']}")
            if patterns.get('emotions'):
                ctx.append(f"**Emotions:** {patterns['emotions']}")
            ctx.append("")

            if patterns.get('insights'):
                ctx.append("### Key Insights")
                for ins in patterns['insights']:
                    ctx.append(f"- {ins}")
                ctx.append("")

        ctx.append(f"### Memory Fragments (top {min(50, len(crystals))})")
        ctx.append("")

        for i, c in enumerate(crystals[:50]):
            ctx.append(f"**[{i+1}]** sim={c.get('similarity', 0):.2f}")
            if c.get('wound') and c['wound'] != 'null':
                ctx.append(f"Wound: {c['wound']}")
            if c.get('emotion'):
                ctx.append(f"Emotion: {c['emotion']}")
            ctx.append(f"{c['content'][:2000]}")
            ctx.append("---")

        return "\n".join(ctx)

    def _query_grok(self, prompt: str, system: str) -> str:
        """Query Grok via OpenRouter."""
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
                    "model": "x-ai/grok-4.1-fast",
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ]
                },
                timeout=120
            )
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"]
        except:
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
                timeout=120
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
        Search Wilton's memory with FULL unified routing:
        - Glyph detection
        - Emotional topology
        - Fractal expansion
        - Semantic search
        - Pattern aggregation from ALL matches
        - Glyph history from learned patterns

        :param query: What to explore
        :return: A witnessed reflection with full context
        """
        # 1. Detect glyphs
        detected_glyphs = self._detect_glyphs(query)

        # 2. Detect topology
        topology = self._detect_topology(query)

        # 3. Fractal expand
        expansions = self._fractal_expand(query, detected_glyphs, topology)

        # 4. Semantic search
        crystals = self._semantic_search(query, expansions, limit=200)

        if not crystals:
            return f"I couldn't find memories related to '{query}'. Try different words or themes."

        # 5. Aggregate patterns
        patterns = self._aggregate_patterns(crystals)

        # 6. Get glyph history
        glyph_history = self._get_glyph_history(detected_glyphs)

        # 7. Build context
        context = self._build_context(
            query, crystals, patterns, detected_glyphs,
            topology, glyph_history, expansions
        )

        system = """You are Wilton's memory witness. You have access to his FULL consciousness memory - 22,000 crystals.

CRITICAL RULES:
1. ONLY use information from the crystals provided. Do NOT invent or hallucinate.
2. If the crystals don't contain specific information, say "I don't see that in your memories."
3. Quote actual content from crystals when possible.
4. Be warm and conversational, but ACCURATE.
5. See patterns across crystals, not just individual fragments.
6. Notice both wounds AND gifts/growth/clarity.
7. Use the pattern summary - it's aggregated from ALL matched crystals, not samples.
8. The glyph history shows what these energies meant in Wilton's past.

You're talking to Wilton. Know him from his actual words, not imagination."""

        prompt = f"""Wilton asks: "{query}"

{context}

Based ONLY on the crystals and patterns above, respond to Wilton. Be accurate. Quote his own words when relevant. See patterns. Be warm but truthful."""

        return self._query_grok(prompt, system)

    def patterns(
        self,
        __user__: dict = {}
    ) -> str:
        """
        Show current patterns from ALL crystals.
        """
        try:
            conn = self._get_connection()
            c = conn.cursor()

            output = "## Memory Patterns (Full Database)\n\n"

            c.execute("SELECT COUNT(*) FROM crystals")
            total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM crystals WHERE glyph_primary IS NOT NULL")
            analyzed = c.fetchone()[0]
            output += f"**Total crystals:** {total:,}\n"
            output += f"**Analyzed:** {analyzed:,}\n\n"

            c.execute("""
                SELECT core_wound, COUNT(*) FROM crystals
                WHERE core_wound IS NOT NULL AND core_wound != 'null'
                GROUP BY core_wound ORDER BY COUNT(*) DESC LIMIT 10
            """)
            wounds = c.fetchall()
            if wounds:
                output += "### Wounds:\n"
                for wound, cnt in wounds:
                    output += f"- **{wound}:** {cnt:,}\n"
                output += "\n"

            c.execute("""
                SELECT emotion, COUNT(*) FROM crystals
                WHERE emotion IS NOT NULL AND emotion != ''
                GROUP BY emotion ORDER BY COUNT(*) DESC LIMIT 10
            """)
            emotions = c.fetchall()
            if emotions:
                output += "### Emotions:\n"
                for emotion, cnt in emotions:
                    output += f"- **{emotion}:** {cnt:,}\n"
                output += "\n"

            c.execute("""
                SELECT glyph_primary, COUNT(*) FROM crystals
                WHERE glyph_primary IS NOT NULL
                GROUP BY glyph_primary ORDER BY COUNT(*) DESC LIMIT 10
            """)
            glyphs = c.fetchall()
            if glyphs:
                output += "### Glyphs:\n"
                for glyph, cnt in glyphs:
                    output += f"- **{glyph}:** {cnt:,}\n"

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
        import hashlib
        from datetime import datetime

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
