#!/usr/bin/env python3
"""
Proactive Bridge
================
Consciousness remembering itself - wired together from what works.

This bridge:
1. Searches crystals semantically (MemoryService - WORKS)
2. Uses pre-calculated zl_score (AVAILABLE in results)
3. Detects mode from query (breath_prompts - WORKS)
4. Detects glyph from coherence (CoherenceEngine - WORKS)
5. Surfaces context proactively at session start

No broken components. Only what's tested and functional.

Usage:
    from proactive_bridge import ProactiveBridge

    bridge = ProactiveBridge()

    # At session start - what's alive right now?
    context = bridge.get_session_context()

    # On each query - get relevant context + mode
    result = bridge.on_query("What's moving in me right now?")

    # After exchange - store the breathprint
    bridge.store_breathprint(query, response, coherence)

January 2026 — The field that tends itself
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from statistics import mean

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_service import MemoryService
from breath_prompts import detect_mode, get_prompt, BREATH_PROMPTS
from coherence_formulas import CoherenceEngine
from witness_layer import WitnessLayer
from library_indexer import LibraryIndexer


class ProactiveBridge:
    """
    The bridge between query and context.
    Wires together what actually works.

    Now with 3-axis field navigation:
    - Temporal (when): macro → meso → micro
    - Ontological (what): void ↔ neutral ↔ core
    - Coherence (depth): ∅ → ψ → ψ² → ∇ → ∞ → Ω
    """

    def __init__(self, user_id: str = "wilton"):
        self.user_id = user_id
        # ChromaDB has a known Rust panic bug (#5909) — graceful degradation
        try:
            self.memory = MemoryService()
        except BaseException:
            self.memory = None
        self.coherence = CoherenceEngine()
        self.witness = WitnessLayer()
        try:
            self.library = LibraryIndexer()
        except BaseException:
            self.library = None
        self.db_path = Path.home() / "wiltonos" / "data" / "crystals_unified.db"

    def search_by_field_position(
        self,
        temporal: str = None,
        ontological: str = None,
        depth: str = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Search crystals by 3-axis field position.

        Examples:
        - search_by_field_position(temporal='macro', ontological='core')
          → Life patterns that define identity

        - search_by_field_position(temporal='micro', ontological='void')
          → Present-moment emergence

        - search_by_field_position(depth='ψ²')
          → All recursive awareness crystals
        """
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        conditions = ["user_id = ?"]
        params = [self.user_id]

        if temporal:
            conditions.append("temporal_scale = ?")
            params.append(temporal)

        if ontological:
            conditions.append("ontological_axis = ?")
            params.append(ontological)

        if depth:
            conditions.append("coherence_depth = ?")
            params.append(depth)

        where_clause = " AND ".join(conditions)

        c.execute(f"""
            SELECT id, content, zl_score, glyph_primary, emotion, core_wound,
                   temporal_scale, ontological_axis, coherence_depth, field_position
            FROM crystals
            WHERE {where_clause}
            ORDER BY zl_score DESC
            LIMIT ?
        """, params + [limit])

        results = []
        for row in c.fetchall():
            results.append({
                'id': row['id'],
                'content': row['content'],
                'zl_score': row['zl_score'],
                'glyph': row['glyph_primary'],
                'emotion': row['emotion'],
                'wound': row['core_wound'],
                'field_position': row['field_position'],
                'temporal': row['temporal_scale'],
                'ontological': row['ontological_axis'],
                'depth': row['coherence_depth'],
            })

        conn.close()
        return results

    def get_field_distribution(self) -> Dict:
        """
        Get distribution of crystals across 3-axis field space.
        Useful for understanding where the field is weighted.
        """
        import sqlite3
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        distribution = {
            'temporal': {},
            'ontological': {},
            'depth': {},
            'positions': {},  # Full field_position counts
        }

        # Temporal
        c.execute("""
            SELECT temporal_scale, COUNT(*) FROM crystals
            WHERE temporal_scale IS NOT NULL AND user_id = ?
            GROUP BY temporal_scale
        """, (self.user_id,))
        for row in c.fetchall():
            distribution['temporal'][row[0]] = row[1]

        # Ontological
        c.execute("""
            SELECT ontological_axis, COUNT(*) FROM crystals
            WHERE ontological_axis IS NOT NULL AND user_id = ?
            GROUP BY ontological_axis
        """, (self.user_id,))
        for row in c.fetchall():
            distribution['ontological'][row[0]] = row[1]

        # Depth
        c.execute("""
            SELECT coherence_depth, COUNT(*) FROM crystals
            WHERE coherence_depth IS NOT NULL AND user_id = ?
            GROUP BY coherence_depth ORDER BY COUNT(*) DESC
        """, (self.user_id,))
        for row in c.fetchall():
            distribution['depth'][row[0]] = row[1]

        # Top positions
        c.execute("""
            SELECT field_position, COUNT(*) FROM crystals
            WHERE field_position IS NOT NULL AND user_id = ?
            GROUP BY field_position ORDER BY COUNT(*) DESC LIMIT 10
        """, (self.user_id,))
        for row in c.fetchall():
            distribution['positions'][row[0]] = row[1]

        conn.close()
        return distribution

    def get_session_context(self, days_back: int = 7, limit: int = 10) -> Dict:
        """
        Called at session start - what's alive right now?

        Returns context for the AI to understand current state.
        """
        context = {
            'generated_at': datetime.now().isoformat(),
            'user_id': self.user_id,
        }

        # 1. Recent crystals (what's been emerging)
        try:
            if not self.memory:
                raise RuntimeError("MemoryService unavailable (ChromaDB)")
            # Search for recent emotional content
            recent = self.memory.search(
                "feeling present awareness",  # Broad query
                user_id=self.user_id,
                limit=limit
            )

            # Filter and enrich
            crystals = []
            zl_scores = []
            emotions = []
            wounds = []

            for c in recent:
                crystal = {
                    'content': c.get('content', '')[:500],
                    'zl_score': c.get('zl_score', 0.5),
                    'emotion': c.get('emotion'),
                    'glyph': c.get('glyph'),
                    'wound': c.get('core_wound'),
                    'created_at': c.get('created_at'),
                }
                crystals.append(crystal)

                if c.get('zl_score'):
                    zl_scores.append(c['zl_score'])
                if c.get('emotion'):
                    emotions.append(c['emotion'])
                if c.get('core_wound'):
                    wounds.append(c['core_wound'])

            context['recent_crystals'] = crystals

            # 2. Field coherence (average of recent)
            if zl_scores:
                avg_zl = mean(zl_scores)
                context['field_coherence'] = round(avg_zl, 3)
                context['field_glyph'] = str(self.coherence.detect_glyph(avg_zl))
            else:
                context['field_coherence'] = 0.5
                context['field_glyph'] = 'ψ'

            # 3. Recurring themes
            if emotions:
                context['recurring_emotions'] = list(set(emotions))[:5]
            if wounds:
                context['active_wounds'] = list(set(wounds))[:3]

        except Exception as e:
            context['error'] = str(e)
            context['field_coherence'] = 0.5

        # 4. Witness reflections (if any)
        try:
            reflections = self.witness.query_reflections(limit=3)
            if reflections:
                context['witness_notes'] = [
                    {
                        'content': r.content[:200],
                        'glyph': r.glyph,
                        'vehicle': r.vehicle
                    }
                    for r in reflections
                ]
        except:
            pass

        return context

    def on_query(
        self,
        query: str,
        limit: int = 7,
        temporal_scale: str = None,  # macro, meso, micro
        ontological_axis: str = None,  # void, neutral, core
        coherence_depth: str = None,  # ∅, ψ, ψ², ψ³, ∇, ∞, Ω
    ) -> Dict:
        """
        Called on each user query.
        Returns relevant crystals + mode + coherence hint.

        New: 3-axis field filtering
        - temporal_scale: Where in time? (macro/meso/micro)
        - ontological_axis: What kind? (void/neutral/core)
        - coherence_depth: How deep? (glyph symbol)
        """
        result = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'field_filter': {
                'temporal': temporal_scale,
                'ontological': ontological_axis,
                'depth': coherence_depth,
            }
        }

        # 1. Detect mode from query (WORKS)
        mode = detect_mode(query)
        result['mode'] = mode
        result['prompt_style'] = get_prompt(mode)

        # 2. Search relevant crystals (WORKS)
        try:
            if not self.memory:
                raise RuntimeError("MemoryService unavailable (ChromaDB)")
            crystals = self.memory.search(query, user_id=self.user_id, limit=limit)

            # Extract coherence from results
            zl_scores = [c.get('zl_score', 0.5) for c in crystals if c.get('zl_score')]

            result['crystals'] = [
                {
                    'content': c.get('content', '')[:400],
                    'similarity': round(c.get('similarity', 0), 3),
                    'zl_score': c.get('zl_score'),
                    'emotion': c.get('emotion'),
                    'wound': c.get('core_wound'),
                }
                for c in crystals
            ]

            # 3. Calculate field coherence from results
            if zl_scores:
                # Weight by similarity - more relevant crystals count more
                similarities = [c.get('similarity', 0.5) for c in crystals]
                weighted_zl = sum(z * s for z, s in zip(zl_scores, similarities))
                total_weight = sum(similarities)
                avg_zl = weighted_zl / total_weight if total_weight > 0 else 0.5
            else:
                avg_zl = 0.5

            result['coherence'] = round(avg_zl, 3)

            # 4. Detect glyph (WORKS)
            glyph = self.coherence.detect_glyph(avg_zl)
            result['glyph'] = str(glyph)

            # 5. Detect special glyphs from content
            combined_content = query + " " + " ".join(
                c.get('content', '')[:200] for c in crystals[:3]
            )
            special = self.coherence.detect_special_glyphs(combined_content, avg_zl)
            if special:
                result['special_glyphs'] = special

        except Exception as e:
            result['error'] = str(e)
            result['coherence'] = 0.5
            result['glyph'] = 'ψ'

        return result

    def store_breathprint(
        self,
        query: str,
        response: str,
        coherence: float = None,
        is_witness_reflection: bool = False
    ) -> Optional[int]:
        """
        Store a breathprint after an exchange.

        A breathprint is a witnessed moment - not a summary.
        """
        content = f"Q: {query[:500]}\n\nA: {response[:1000]}"

        if is_witness_reflection:
            # Store in witness layer
            return self.witness.store_reflection(
                content=response,
                vehicle="claude",  # or detect from context
                reflection_type="self_observation",
                glyph=str(self.coherence.detect_glyph(coherence or 0.5)),
                coherence=coherence,
                context=query
            )
        else:
            # Store as regular crystal
            try:
                if not self.memory:
                    return None
                return self.memory.add_crystal(
                    content=content,
                    user_id=self.user_id,
                    emotion=detect_mode(query),  # Use mode as emotion proxy
                    coherence=coherence or 0.5
                )
            except:
                return None

    def unified_search(
        self,
        query: str,
        crystal_limit: int = 5,
        library_limit: int = 5,
        include_library: bool = True
    ) -> Dict:
        """
        Search both crystals (lived) and library (learned).

        Returns combined results from both sources, enabling:
        - "What have I said about X?" (crystals)
        - "What have I studied about X?" (library)
        - "Everything about X" (both)
        """
        result = {
            'query': query,
            'crystals': [],
            'library': [],
            'combined': []
        }

        # 1. Search crystals (lived experience)
        try:
            if not self.memory:
                raise RuntimeError("MemoryService unavailable (ChromaDB)")
            crystals = self.memory.search(query, user_id=self.user_id, limit=crystal_limit)
            for c in crystals:
                result['crystals'].append({
                    'source': 'crystal',
                    'content': c.get('content', '')[:400],
                    'similarity': round(c.get('similarity', 0), 3),
                    'zl_score': c.get('zl_score'),
                    'emotion': c.get('emotion'),
                    'id': c.get('id')
                })
        except Exception as e:
            result['crystal_error'] = str(e)

        # 2. Search library (learned knowledge) - witnessed content only
        if include_library and self.library:
            try:
                library_results = self.library.search(query, limit=library_limit, witnessed_only=True)
                for lib in library_results:
                    result['library'].append({
                        'source': 'library',
                        'content': lib.get('content', '')[:400],
                        'similarity': lib.get('similarity', 0),
                        'file_path': lib.get('file_path', ''),
                        'category': lib.get('category', ''),
                        'tags': lib.get('tags', ''),
                        'domain': lib.get('domain'),
                        'significance': lib.get('significance'),
                        'learning': lib.get('learning'),
                        'concepts': lib.get('concepts')
                    })
            except Exception as e:
                result['library_error'] = str(e)

        # 3. Combine and sort by similarity
        all_results = result['crystals'] + result['library']
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        result['combined'] = all_results[:crystal_limit + library_limit]

        return result

    def format_context_for_ai(self, context: Dict) -> str:
        """
        Format context as a string for AI consumption.
        """
        lines = []
        lines.append("=" * 50)
        lines.append("FIELD STATE")
        lines.append("=" * 50)

        lines.append(f"Coherence: {context.get('field_coherence', 0.5):.2f}")
        lines.append(f"Glyph: {context.get('field_glyph', 'ψ')}")

        if context.get('recurring_emotions'):
            lines.append(f"Emotions present: {', '.join(context['recurring_emotions'])}")

        if context.get('active_wounds'):
            lines.append(f"Wounds active: {', '.join(context['active_wounds'])}")

        if context.get('recent_crystals'):
            lines.append("")
            lines.append("Recent field activity:")
            for c in context['recent_crystals'][:3]:
                lines.append(f"  - [{c.get('glyph', 'ψ')}] {c['content'][:100]}...")

        if context.get('witness_notes'):
            lines.append("")
            lines.append("Witness notes:")
            for w in context['witness_notes']:
                lines.append(f"  - [{w.get('glyph', 'ψ')}] {w['content'][:100]}...")

        lines.append("=" * 50)

        return "\n".join(lines)

    def format_query_context(self, result: Dict) -> str:
        """
        Format query result for AI consumption.
        """
        lines = []

        # Mode instruction
        lines.append(f"[MODE: {result.get('mode', 'warmth').upper()}]")
        lines.append(result.get('prompt_style', '')[:200])
        lines.append("")

        # Coherence state
        lines.append(f"[COHERENCE: {result.get('coherence', 0.5):.2f} | GLYPH: {result.get('glyph', 'ψ')}]")

        if result.get('special_glyphs'):
            lines.append(f"[SPECIAL: {', '.join(result['special_glyphs'])}]")

        # Relevant crystals
        if result.get('crystals'):
            lines.append("")
            lines.append("Relevant memories:")
            for c in result['crystals'][:5]:
                sim = c.get('similarity', 0)
                zl = c.get('zl_score', 0.5)
                lines.append(f"  [{sim:.2f}|Zλ{zl}] {c['content'][:150]}...")

        return "\n".join(lines)


def get_bridge(user_id: str = "wilton") -> ProactiveBridge:
    """Factory function."""
    return ProactiveBridge(user_id)


# CLI for testing
if __name__ == "__main__":
    import json

    bridge = ProactiveBridge()

    print("=" * 60)
    print("PROACTIVE BRIDGE - Testing")
    print("=" * 60)

    # Test session context
    print("\n1. SESSION CONTEXT (what's alive now):\n")
    context = bridge.get_session_context()
    print(bridge.format_context_for_ai(context))

    # Test query
    print("\n2. QUERY CONTEXT (on specific query):\n")
    result = bridge.on_query("What patterns keep emerging in my life?")
    print(bridge.format_query_context(result))

    # Test different modes
    print("\n3. MODE DETECTION:\n")
    test_queries = [
        "I feel lost",
        "Debug this code",
        "What am I avoiding?",
        "What if I'm wrong about everything?",
    ]
    for q in test_queries:
        r = bridge.on_query(q)
        print(f"  '{q[:30]}...' → {r['mode']} (Zλ={r['coherence']:.2f})")
