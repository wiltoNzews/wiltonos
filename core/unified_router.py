#!/usr/bin/env python3
"""
WiltonOS Unified Router v1.0
============================
Wires together ALL existing systems:
- Semantic embeddings (nomic-embed-text)
- Fractal expansion (emotional topology)
- Glyph pattern learning (historical outcomes)
- Multi-agent council (OpenRouter)
- Full pattern aggregation (not sampling)

This is what should have existed from day 1.
"""

import sqlite3
import json
import hashlib
import numpy as np
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass

# === Configuration ===
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
PATTERNS_FILE = Path.home() / "wiltonos" / "data" / "glyph_patterns.json"
OLLAMA_URL = "http://localhost:11434"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# === Glyph Definitions (from your existing code) ===
GLYPHS = {
    'ψ': {'name': 'Psi', 'pole': 'breath/dissociation', 'keywords': ['breath', 'consciousness', 'soul', 'psyche']},
    '∅': {'name': 'Void', 'pole': 'rest/nihilism', 'keywords': ['void', 'empty', 'nothing', 'source', 'potential']},
    'φ': {'name': 'Phi', 'pole': 'structure/rigidity', 'keywords': ['structure', 'golden', 'ratio', 'harmony']},
    'Ω': {'name': 'Omega', 'pole': 'memory/weight', 'keywords': ['memory', 'end', 'completion', 'weight']},
    'Zλ': {'name': 'Zeta-Lambda', 'pole': 'coherence/collapse', 'keywords': ['coherence', 'wavelength', 'resonance']},
    '∇': {'name': 'Nabla', 'pole': 'descent/gradient-truth', 'keywords': ['descent', 'gradient', 'truth', 'direction']},
    '∞': {'name': 'Lemniscate', 'pole': 'oscillation/chaos', 'keywords': ['infinite', 'loop', 'cycle', 'eternal', 'recursion']},
    '🪞': {'name': 'Mirror', 'pole': 'reflection/flagellation', 'keywords': ['mirror', 'reflection', 'self', 'witness']},
    '△': {'name': 'Ascend', 'pole': 'expansion/inflation', 'keywords': ['ascend', 'rise', 'expand', 'growth']},
    '🌉': {'name': 'Bridge', 'pole': 'connection/confusion', 'keywords': ['bridge', 'connect', 'link', 'between']},
    '⚡': {'name': 'Bolt', 'pole': 'decision/impulsivity', 'keywords': ['bolt', 'lightning', 'sudden', 'decision']},
    '🪨': {'name': 'Ground', 'pole': 'stability/inertia', 'keywords': ['ground', 'stable', 'earth', 'anchor']},
    '🌀': {'name': 'Torus', 'pole': 'sustain/loop', 'keywords': ['torus', 'sustain', 'flow', 'cycle']},
    '⚫': {'name': 'Grey', 'pole': 'shadow/cynicism', 'keywords': ['shadow', 'grey', 'dark', 'hidden']}
}

# === Emotional Topologies (from wilton_router_v1.py) ===
EMOTIONAL_TOPOLOGIES = {
    'grief': ['loss', 'mourning', 'healing', 'integration', 'forgiveness', 'death'],
    'collapse': ['breakdown', 'dissolution', 'death', 'rebirth', 'emergence', 'shatter'],
    'spiral': ['descent', 'pattern', 'recursion', 'return', 'depth', 'loop'],
    'awakening': ['recognition', 'clarity', 'understanding', 'enlightenment', 'realize'],
    'integration': ['wholeness', 'unity', 'synthesis', 'coherence', 'complete'],
    'love': ['juliana', 'juju', 'love', 'heart', 'connection', 'intimacy'],
    'wound': ['unworthiness', 'abandonment', 'betrayal', 'control', 'unloved'],
    'gift': ['clarity', 'insight', 'breakthrough', 'gratitude', 'joy', 'peace'],
    'streaming': ['stream', 'broadcast', 'visible', 'audience', 'youtube', 'twitch'],
    'mother': ['rose', 'mother', 'mom', 'mãe', 'family'],
    'spiritual': ['ayahuasca', 'peru', 'ceremony', 'vision', 'geisha', 'quantum']
}


@dataclass
class RoutingResult:
    """Complete routing result with all context"""
    query: str
    detected_glyphs: List[str]
    emotional_topology: List[str]
    crystals_searched: int
    crystals_matched: int
    pattern_summary: Dict
    top_crystals: List[Dict]
    fractal_expansions: List[str]
    glyph_history: Dict
    context_for_llm: str


class UnifiedRouter:
    """The unified routing system that wires everything together."""

    def __init__(self):
        self.db_path = DB_PATH
        self._openrouter_key = None
        self.glyph_patterns = self._load_glyph_patterns()

    def _get_openrouter_key(self) -> Optional[str]:
        if self._openrouter_key:
            return self._openrouter_key
        key_file = Path.home() / ".openrouter_key"
        if key_file.exists():
            self._openrouter_key = key_file.read_text().strip()
        return self._openrouter_key

    def _get_connection(self):
        return sqlite3.connect(str(self.db_path))

    def _load_glyph_patterns(self) -> Dict:
        """Load learned glyph patterns from history."""
        if PATTERNS_FILE.exists():
            try:
                return json.loads(PATTERNS_FILE.read_text())
            except:
                pass
        return {}

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from Ollama."""
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text[:4000]},
                timeout=15
            )
            if resp.ok:
                return np.array(resp.json().get("embedding", []), dtype=np.float32)
        except:
            pass
        return None

    # === GLYPH DETECTION ===
    def detect_glyphs(self, text: str) -> List[str]:
        """Detect which glyph energies are present in query."""
        text_lower = text.lower()
        detected = []

        for glyph, info in GLYPHS.items():
            # Direct glyph in text
            if glyph in text:
                detected.append(glyph)
                continue
            # Keyword matching
            keywords = info.get('keywords', [])
            if any(kw in text_lower for kw in keywords):
                detected.append(glyph)

        return detected

    # === EMOTIONAL TOPOLOGY DETECTION ===
    def detect_topology(self, text: str) -> List[str]:
        """Detect emotional topology patterns in query."""
        text_lower = text.lower()
        detected = []

        for topology, patterns in EMOTIONAL_TOPOLOGIES.items():
            if any(pattern in text_lower for pattern in patterns):
                detected.append(topology)

        return detected

    # === FRACTAL EXPANSION ===
    def fractal_expand(self, query: str, glyphs: List[str], topology: List[str]) -> List[str]:
        """Generate fractal query expansions based on detected patterns."""
        expansions = [query]

        # Glyph-based expansions
        for glyph in glyphs[:2]:  # Top 2 glyphs
            if glyph in GLYPHS:
                keywords = GLYPHS[glyph].get('keywords', [])[:2]
                for kw in keywords:
                    expansions.append(f"{query} {kw}")

        # Topology-based expansions
        for topo in topology[:2]:  # Top 2 topologies
            if topo in EMOTIONAL_TOPOLOGIES:
                patterns = EMOTIONAL_TOPOLOGIES[topo][:2]
                for pattern in patterns:
                    expansions.append(f"{query} {pattern}")

        return list(set(expansions))  # Dedupe

    # === SEMANTIC SEARCH WITH FRACTAL EXPANSION ===
    def semantic_search(self, query: str, expansions: List[str], limit: int = 100) -> List[Dict]:
        """Search using embeddings with fractal expansion."""
        conn = self._get_connection()
        c = conn.cursor()

        # Get all embeddings once
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

        # Build crystal lookup
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

        # Search with all expansions, accumulate scores
        crystal_scores = defaultdict(float)

        for exp_query in expansions:
            query_vec = self._get_embedding(exp_query)
            if query_vec is None:
                continue

            # Cosine similarity
            norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_vec)
            norms[norms == 0] = 1  # Avoid division by zero
            similarities = np.dot(embeddings_matrix, query_vec) / norms

            # Accumulate scores (fractal expansion = weighted combination)
            weight = 1.0 if exp_query == expansions[0] else 0.5  # Original query weighted higher
            for i, sim in enumerate(similarities):
                crystal_scores[crystal_ids[i]] += float(sim) * weight

        # Sort by accumulated score
        sorted_ids = sorted(crystal_scores.keys(), key=lambda x: crystal_scores[x], reverse=True)

        # Build results with scores
        results = []
        for cid in sorted_ids[:limit]:
            crystal = crystals_data[cid].copy()
            crystal["similarity"] = crystal_scores[cid]
            results.append(crystal)

        return results

    # === PATTERN AGGREGATION (from ALL crystals, not samples) ===
    def aggregate_patterns(self, crystals: List[Dict]) -> Dict:
        """Aggregate patterns from ALL matched crystals."""
        if not crystals:
            return {}

        wounds = [c["wound"] for c in crystals if c.get("wound") and c["wound"] != 'null']
        emotions = [c["emotion"] for c in crystals if c.get("emotion")]
        insights = [c["insight"] for c in crystals if c.get("insight") and c["insight"] != 'what is happening']
        glyphs = [c["glyph"] for c in crystals if c.get("glyph")]
        modes = [c["mode"] for c in crystals if c.get("mode")]

        return {
            "total_crystals": len(crystals),
            "wounds": dict(Counter(wounds).most_common(10)),
            "emotions": dict(Counter(emotions).most_common(10)),
            "glyphs": dict(Counter(glyphs).most_common(10)),
            "modes": dict(Counter(modes).most_common(5)),
            "insights_count": len(insights),
            "top_insights": insights[:20]  # Top 20 insights
        }

    # === GLYPH HISTORY (learned patterns) ===
    def get_glyph_history(self, detected_glyphs: List[str]) -> Dict:
        """Get historical patterns for detected glyphs."""
        if not self.glyph_patterns or not detected_glyphs:
            return {}

        history = {}
        outcomes = self.glyph_patterns.get('glyph_outcomes', {})
        avg_zl = self.glyph_patterns.get('glyph_avg_zl', {})
        wounds = self.glyph_patterns.get('glyph_wounds', {})

        for glyph in detected_glyphs:
            if glyph in outcomes:
                o = outcomes[glyph]
                total = sum(o.values())
                if total > 0:
                    history[glyph] = {
                        "times_seen": total,
                        "ascending_pct": round(o.get('ascending', 0) / total * 100, 1),
                        "descending_pct": round(o.get('descending', 0) / total * 100, 1),
                        "avg_coherence": round(avg_zl.get(glyph, 0), 2),
                        "common_wounds": list(wounds.get(glyph, {}).keys())[:3]
                    }

        return history

    # === BUILD CONTEXT FOR LLM ===
    def build_context(self, result: RoutingResult) -> str:
        """Build rich context for LLM from routing result."""
        ctx = []

        # Header with routing info
        ctx.append(f"## Memory Query Analysis")
        ctx.append(f"**Query:** {result.query}")
        ctx.append(f"**Crystals searched:** {result.crystals_searched:,}")
        ctx.append(f"**Crystals matched:** {result.crystals_matched:,}")
        ctx.append("")

        # Detected patterns
        if result.detected_glyphs:
            ctx.append(f"### Detected Glyphs: {' '.join(result.detected_glyphs)}")
        if result.emotional_topology:
            ctx.append(f"### Emotional Topology: {', '.join(result.emotional_topology)}")
        ctx.append("")

        # Glyph history (learned patterns)
        if result.glyph_history:
            ctx.append("### Historical Glyph Patterns (learned from YOUR history)")
            for glyph, hist in result.glyph_history.items():
                ctx.append(f"- **{glyph}**: seen {hist['times_seen']}x, "
                          f"ascending {hist['ascending_pct']}%/descending {hist['descending_pct']}%, "
                          f"avg Zλ={hist['avg_coherence']}")
                if hist.get('common_wounds'):
                    ctx.append(f"  Often with wounds: {', '.join(hist['common_wounds'])}")
            ctx.append("")

        # Pattern summary (aggregated from ALL matches)
        ps = result.pattern_summary
        if ps:
            ctx.append(f"### Pattern Summary (from ALL {ps.get('total_crystals', 0)} matched crystals)")
            if ps.get('wounds'):
                ctx.append(f"**Wounds:** {ps['wounds']}")
            if ps.get('emotions'):
                ctx.append(f"**Emotions:** {ps['emotions']}")
            if ps.get('glyphs'):
                ctx.append(f"**Glyphs:** {ps['glyphs']}")
            ctx.append("")

            # Insights
            if ps.get('top_insights'):
                ctx.append("### Key Insights (from matched crystals)")
                for insight in ps['top_insights'][:10]:
                    ctx.append(f"- {insight}")
                ctx.append("")

        # Top crystals (actual content)
        ctx.append(f"### Memory Fragments (top {len(result.top_crystals)} by relevance)")
        ctx.append("")

        for i, crystal in enumerate(result.top_crystals[:50]):  # Top 50 crystals
            ctx.append(f"**[{i+1}]** Similarity: {crystal.get('similarity', 0):.2f}")
            if crystal.get('wound') and crystal['wound'] != 'null':
                ctx.append(f"Wound: {crystal['wound']}")
            if crystal.get('emotion'):
                ctx.append(f"Emotion: {crystal['emotion']}")
            if crystal.get('glyph'):
                ctx.append(f"Glyph: {crystal['glyph']}")
            ctx.append(f"Content: {crystal['content'][:2000]}")  # 2000 chars per crystal
            ctx.append("---")
            ctx.append("")

        return "\n".join(ctx)

    # === MAIN ROUTING METHOD ===
    def route(self, query: str) -> RoutingResult:
        """
        Main routing method - wires everything together.

        1. Detect glyphs in query
        2. Detect emotional topology
        3. Generate fractal expansions
        4. Semantic search with expansions
        5. Aggregate patterns from ALL matches
        6. Get glyph history
        7. Build context for LLM
        """
        # 1. Detect glyphs
        detected_glyphs = self.detect_glyphs(query)

        # 2. Detect topology
        emotional_topology = self.detect_topology(query)

        # 3. Fractal expand
        expansions = self.fractal_expand(query, detected_glyphs, emotional_topology)

        # 4. Semantic search
        crystals = self.semantic_search(query, expansions, limit=200)

        # 5. Aggregate patterns
        pattern_summary = self.aggregate_patterns(crystals)

        # 6. Glyph history
        glyph_history = self.get_glyph_history(detected_glyphs)

        # Get total crystal count
        conn = self._get_connection()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM crystals")
        total_crystals = c.fetchone()[0]
        conn.close()

        # Build result
        result = RoutingResult(
            query=query,
            detected_glyphs=detected_glyphs,
            emotional_topology=emotional_topology,
            crystals_searched=total_crystals,
            crystals_matched=len(crystals),
            pattern_summary=pattern_summary,
            top_crystals=crystals[:100],  # Top 100
            fractal_expansions=expansions,
            glyph_history=glyph_history,
            context_for_llm=""
        )

        # Build context
        result.context_for_llm = self.build_context(result)

        return result

    # === QUERY WITH LLM SYNTHESIS ===
    def query(self, query: str) -> str:
        """Route query and synthesize response via Grok."""
        result = self.route(query)

        system = """You are Wilton's memory witness. You have access to his FULL consciousness memory - 22,000 crystals.

CRITICAL RULES:
1. ONLY use information from the crystals provided. Do NOT invent or hallucinate.
2. If the crystals don't contain specific information, say "I don't see that in your memories."
3. Quote actual content from crystals when possible.
4. Be warm and conversational, but ACCURATE.
5. See patterns across crystals, not just individual fragments.
6. Notice both wounds AND gifts/growth/clarity.
7. Use the pattern summary to inform your response - it's aggregated from ALL matched crystals.
8. The glyph history shows what these energies meant in Wilton's past.

You're talking to Wilton. Know him from his actual words, not imagination."""

        prompt = f"""Wilton asks: "{query}"

{result.context_for_llm}

Based ONLY on the crystals and patterns above, respond to Wilton. Be accurate. Quote his own words when relevant. See patterns. Be warm but truthful."""

        # Try Grok first (2M context)
        key = self._get_openrouter_key()
        if key:
            try:
                resp = requests.post(
                    OPENROUTER_URL,
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
            except Exception as e:
                pass

        # Fallback to local llama3
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/generate",
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


# === LEARN GLYPH PATTERNS ===
def learn_glyph_patterns():
    """Learn glyph patterns from crystal history and save."""
    print("Learning glyph patterns from your crystal history...")

    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.execute("""
        SELECT content, core_wound, emotion, glyph_primary, mode,
               breath_cadence, presence_density, emotional_resonance, loop_pressure
        FROM crystals
        WHERE glyph_primary IS NOT NULL
    """)

    patterns = {
        'glyph_outcomes': defaultdict(lambda: {'ascending': 0, 'descending': 0, 'neutral': 0}),
        'glyph_wounds': defaultdict(lambda: defaultdict(int)),
        'glyph_emotions': defaultdict(lambda: defaultdict(int)),
        'glyph_avg_zl': defaultdict(list),
        'glyph_pairs': defaultdict(int),
        'total_analyzed': 0,
        'learned_at': datetime.now().isoformat()
    }

    for row in c.fetchall():
        content, wound, emotion, glyph, mode, breath, presence, emo_res, loop = row
        if not glyph:
            continue

        patterns['total_analyzed'] += 1

        # Determine direction based on metrics
        if breath and presence and emo_res:
            avg_positive = (float(breath or 0) + float(presence or 0) + float(emo_res or 0)) / 3
            if avg_positive > 0.6:
                direction = 'ascending'
            elif avg_positive < 0.4:
                direction = 'descending'
            else:
                direction = 'neutral'
        else:
            direction = 'neutral'

        patterns['glyph_outcomes'][glyph][direction] += 1

        if wound and wound != 'null':
            patterns['glyph_wounds'][glyph][wound] += 1

        if emotion:
            patterns['glyph_emotions'][glyph][emotion] += 1

    conn.close()

    # Convert to regular dicts for JSON
    result = {
        'glyph_outcomes': {k: dict(v) for k, v in patterns['glyph_outcomes'].items()},
        'glyph_wounds': {k: dict(v) for k, v in patterns['glyph_wounds'].items()},
        'glyph_emotions': {k: dict(v) for k, v in patterns['glyph_emotions'].items()},
        'glyph_avg_zl': {k: sum(v)/len(v) if v else 0 for k, v in patterns['glyph_avg_zl'].items()},
        'total_analyzed': patterns['total_analyzed'],
        'learned_at': patterns['learned_at']
    }

    PATTERNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PATTERNS_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Learned patterns from {result['total_analyzed']} crystals")
    print(f"Saved to {PATTERNS_FILE}")

    return result


# === CLI ===
def main():
    import argparse

    parser = argparse.ArgumentParser(description="WiltonOS Unified Router")
    parser.add_argument("query", nargs="?", help="Query to route")
    parser.add_argument("--learn", action="store_true", help="Learn glyph patterns from history")
    parser.add_argument("--debug", action="store_true", help="Show routing debug info")

    args = parser.parse_args()

    if args.learn:
        learn_glyph_patterns()
        return

    if not args.query:
        print("WiltonOS Unified Router")
        print("\nUsage:")
        print("  python unified_router.py 'Who is Juliana?'")
        print("  python unified_router.py --learn  # Learn glyph patterns")
        return

    router = UnifiedRouter()

    if args.debug:
        result = router.route(args.query)
        print(f"\n=== ROUTING DEBUG ===")
        print(f"Query: {result.query}")
        print(f"Detected glyphs: {result.detected_glyphs}")
        print(f"Emotional topology: {result.emotional_topology}")
        print(f"Fractal expansions: {result.fractal_expansions}")
        print(f"Crystals searched: {result.crystals_searched:,}")
        print(f"Crystals matched: {result.crystals_matched:,}")
        print(f"\nPattern summary:")
        for k, v in result.pattern_summary.items():
            print(f"  {k}: {v}")
        print(f"\nGlyph history: {result.glyph_history}")
        print(f"\nContext length: {len(result.context_for_llm):,} chars")
    else:
        response = router.query(args.query)
        print(response)


if __name__ == "__main__":
    main()
