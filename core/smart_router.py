"""
Smart Router - Multi-Scale Coherent Sampling + Formulas
========================================================
Not flat similarity search. STRUCTURED sampling across scales.
Now with actual coherence formulas from the crystals.

Formulas integrated:
- ψ oscillator: ψ = clamp(ψ + sine(breath) - return_force(ψ), floor, ceiling)
- Zλ coherence with glyph detection (∅ → ψ → ψ² → ∇ → ∞)
- Lemniscate sampling (walk the ∞ curve)
- 3:1 coherence ratio (aligned + challengers)
- Attractor detection (truth, silence, mirror, etc.)

Scales:
- Macro  → Big themes, clusters
- Meso   → Related topics
- Micro  → Specific moments

Sources: Roy Herbert (Chronoflux), Roo (RHE), Brandon (OmniLens), JNT (Hans Lattice)
"""

import sqlite3
import numpy as np
import requests
import math
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict, Counter
import json

# Import coherence formulas
try:
    from coherence_formulas import CoherenceEngine, GlyphState, FieldMode
except ImportError:
    CoherenceEngine = None


class SmartRouter:
    """Multi-scale coherent sampling router with formula integration."""

    def __init__(self, db_path: str = None, user_id: str = "wilton"):
        self.db_path = Path(db_path or "/home/zews/wiltonos/data/crystals_unified.db")
        self.user_id = user_id
        self.ollama_url = "http://localhost:11434"

        # Coherence ratio: 3 aligned : 1 challenging
        self.COHERENCE_RATIO = 3

        # Scale distribution
        self.SCALE_WEIGHTS = {
            'macro': 0.2,   # 20% broad themes
            'meso': 0.5,    # 50% related topics
            'micro': 0.3    # 30% specific moments
        }

        # Initialize coherence engine if available
        self.coherence_engine = CoherenceEngine() if CoherenceEngine else None

        # Glyph-driven sampling parameters: theta_steps and scale weights
        self.GLYPH_SAMPLING = {
            GlyphState.VOID:        {'theta_steps': 4,  'weights': {'macro': 0.50, 'meso': 0.30, 'micro': 0.20}},
            GlyphState.PSI:         {'theta_steps': 6,  'weights': {'macro': 0.30, 'meso': 0.50, 'micro': 0.20}},
            GlyphState.PSI_SQUARED: {'theta_steps': 8,  'weights': {'macro': 0.20, 'meso': 0.50, 'micro': 0.30}},
            GlyphState.PSI_CUBED:   {'theta_steps': 10, 'weights': {'macro': 0.20, 'meso': 0.40, 'micro': 0.40}},
            GlyphState.NABLA:       {'theta_steps': 10, 'weights': {'macro': 0.15, 'meso': 0.35, 'micro': 0.50}},
            GlyphState.INFINITY:    {'theta_steps': 12, 'weights': {'macro': 0.15, 'meso': 0.35, 'micro': 0.50}},
            GlyphState.OMEGA:       {'theta_steps': 16, 'weights': {'macro': 0.10, 'meso': 0.30, 'micro': 0.60}},
            GlyphState.CROSSBLADE:  {'theta_steps': 4,  'weights': {'macro': 0.50, 'meso': 0.30, 'micro': 0.20}},
            GlyphState.LAYER_MERGE: {'theta_steps': 12, 'weights': {'macro': 0.33, 'meso': 0.34, 'micro': 0.33}},
        }

        # Variance seed for non-repetitive sampling
        self._variance_seed = random.random()

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

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _load_all_crystals(self) -> List[Dict]:
        """Load all crystals with embeddings."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT c.id, c.content, c.core_wound, c.emotion, c.insight,
                   c.created_at, e.embedding
            FROM crystals c
            JOIN crystal_embeddings e ON e.crystal_id = c.id
            WHERE c.user_id = ?
        """, (self.user_id,))

        crystals = []
        for row in c.fetchall():
            try:
                emb = np.frombuffer(row[6], dtype=np.float32)
                crystals.append({
                    'id': row[0],
                    'content': row[1],
                    'wound': row[2],
                    'emotion': row[3],
                    'insight': row[4],
                    'created_at': row[5],
                    'embedding': emb
                })
            except:
                continue

        conn.close()
        return crystals

    def _compute_similarities(self, query_vec: np.ndarray, crystals: List[Dict]) -> List[Dict]:
        """Compute similarities and add to crystals."""
        for c in crystals:
            c['similarity'] = self._cosine_sim(query_vec, c['embedding'])
        return crystals

    def _scale_sample(
        self,
        crystals: List[Dict],
        total: int = 50,
        scale_weights: Dict[str, float] = None
    ) -> Dict[str, List[Dict]]:
        """
        Sample across scales: macro, meso, micro.

        Macro (top 20%): Highest similarity - the core resonance
        Meso (middle 50%): Medium similarity - related but not exact
        Micro (bottom 30%): Lower similarity - specific tangents, surprises

        scale_weights: Optional glyph-driven weights override (e.g. {'macro': 0.5, 'meso': 0.3, 'micro': 0.2})
        """
        weights = scale_weights or self.SCALE_WEIGHTS

        sorted_crystals = sorted(crystals, key=lambda x: x['similarity'], reverse=True)
        n = len(sorted_crystals)

        # Define scale boundaries
        macro_end = int(n * 0.1)      # Top 10% of matches
        meso_end = int(n * 0.4)       # 10-40% range

        macro_pool = sorted_crystals[:macro_end] if macro_end > 0 else sorted_crystals[:5]
        meso_pool = sorted_crystals[macro_end:meso_end] if meso_end > macro_end else []
        micro_pool = sorted_crystals[meso_end:] if meso_end < n else []

        # Sample according to weights
        n_macro = int(total * weights['macro'])
        n_meso = int(total * weights['meso'])
        n_micro = total - n_macro - n_meso

        result = {
            'macro': macro_pool[:n_macro],
            'meso': meso_pool[:n_meso] if meso_pool else macro_pool[n_macro:n_macro+n_meso],
            'micro': micro_pool[:n_micro] if micro_pool else []
        }

        return result

    def _lemniscate_sample(
        self,
        crystals: List[Dict],
        n_samples: int = 50,
        theta_steps: int = 8
    ) -> List[Dict]:
        """
        Sample crystals along a lemniscate (∞) path.

        Instead of just top-N by similarity, walk the ∞ curve
        to get crystals from different "phases" of the loop.

        The lemniscate: r² = a² cos 2θ
        θ from 0 to 2π traces the full figure-8
        """
        if not crystals or len(crystals) < n_samples:
            return crystals[:n_samples]

        sorted_crystals = sorted(crystals, key=lambda x: x.get('similarity', 0), reverse=True)
        n = len(sorted_crystals)

        samples = []
        samples_per_step = max(1, n_samples // theta_steps)

        # Add variance so same query doesn't always hit same crystals
        phase_offset = self._variance_seed * 0.5

        for i in range(theta_steps):
            # θ position on lemniscate (0 to 2π) with variance
            theta = ((i / theta_steps) + phase_offset) * 2 * math.pi

            # r² = cos(2θ) gives position on curve
            r_squared = math.cos(2 * theta)

            # Convert to index (0 to n-1)
            normalized = (r_squared + 1) / 2  # 0 to 1
            idx_center = int(normalized * (n - 1))

            # Add small random jitter
            jitter = random.randint(-2, 2)
            idx_center = max(0, min(n - 1, idx_center + jitter))

            # Sample around that index
            start = max(0, idx_center - samples_per_step // 2)
            end = min(n, start + samples_per_step)

            samples.extend(sorted_crystals[start:end])

        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for c in samples:
            cid = c.get('id')
            if cid not in seen:
                seen.add(cid)
                unique.append(c)

        # Rotate variance for next query
        self._variance_seed = (self._variance_seed + 0.1) % 1.0

        return unique[:n_samples]

    def _apply_coherence_ratio(
        self,
        aligned: List[Dict],
        all_crystals: List[Dict],
        query_vec: np.ndarray
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Apply 3:1 coherence ratio.

        For every 3 aligned crystals, find 1 challenging crystal.
        Challenging = low similarity to query but high similarity to aligned crystals
        (things that connect to what we found but from a different angle)
        """
        n_aligned = len(aligned)
        n_challengers = max(1, n_aligned // self.COHERENCE_RATIO)

        # Get average embedding of aligned crystals
        if not aligned:
            return aligned, []

        aligned_embeddings = [c['embedding'] for c in aligned if 'embedding' in c]
        if not aligned_embeddings:
            return aligned, []

        avg_aligned = np.mean(aligned_embeddings, axis=0)

        # Find challengers: low query similarity but connected to aligned
        aligned_ids = {c['id'] for c in aligned}
        candidates = []

        for c in all_crystals:
            if c['id'] in aligned_ids:
                continue

            query_sim = c.get('similarity', 0)
            aligned_sim = self._cosine_sim(c['embedding'], avg_aligned)

            # Challenger score: high aligned_sim, lower query_sim
            # This finds things related to what we found but from different angle
            if query_sim < 0.5 and aligned_sim > 0.3:
                c['challenger_score'] = aligned_sim - query_sim
                candidates.append(c)

        # Sort by challenger score and take top N
        candidates.sort(key=lambda x: x.get('challenger_score', 0), reverse=True)
        challengers = candidates[:n_challengers]

        return aligned, challengers

    def _temporal_sample(
        self,
        crystals: List[Dict],
        n: int = 10
    ) -> Dict[str, List[Dict]]:
        """
        Temporal sampling: earlier vs later.

        Returns crystals from different time periods.
        """
        # Sort by created_at
        with_dates = [c for c in crystals if c.get('created_at')]
        if not with_dates:
            return {'early': [], 'recent': []}

        sorted_temporal = sorted(with_dates, key=lambda x: x['created_at'])

        half = len(sorted_temporal) // 2
        early = sorted_temporal[:half]
        recent = sorted_temporal[half:]

        return {
            'early': early[:n//2],
            'recent': recent[-(n//2):]
        }

    def _find_synapses(
        self,
        crystals: List[Dict],
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Find synapses: connections BETWEEN crystals.

        A synapse is a pair of crystals that are highly similar to each other
        but represent different aspects/times/emotions.
        """
        synapses = []
        n = len(crystals)

        for i in range(min(n, 20)):  # Limit computation
            for j in range(i + 1, min(n, 20)):
                c1, c2 = crystals[i], crystals[j]

                sim = self._cosine_sim(c1['embedding'], c2['embedding'])

                if sim > threshold:
                    # Check if they're actually different (not duplicates)
                    if c1.get('emotion') != c2.get('emotion') or c1.get('wound') != c2.get('wound'):
                        synapses.append({
                            'crystal_a': c1,
                            'crystal_b': c2,
                            'similarity': sim,
                            'connection': f"{c1.get('emotion', '?')} ↔ {c2.get('emotion', '?')}"
                        })

        return sorted(synapses, key=lambda x: x['similarity'], reverse=True)[:5]

    def route(
        self,
        query: str,
        total_crystals: int = 50,
        include_challengers: bool = True,
        include_temporal: bool = True,
        include_synapses: bool = True,
        use_lemniscate: bool = True
    ) -> Dict:
        """
        Main routing function - smart multi-scale sampling with formulas.

        Returns:
            Dict with:
            - aligned: crystals that resonate with query
            - challengers: crystals that challenge/contrast
            - temporal: early vs recent crystals
            - synapses: connections between crystals
            - scales: macro/meso/micro breakdown
            - coherence: Zλ, glyph, mode, attractor (if engine available)
            - context: formatted context string
        """
        query_vec = self._get_embedding(query)
        if query_vec is None:
            return {'error': 'Could not embed query', 'context': ''}

        # Load and score all crystals
        all_crystals = self._load_all_crystals()
        if not all_crystals:
            return {'error': 'No crystals found', 'context': ''}

        crystals = self._compute_similarities(query_vec, all_crystals)

        # Compute preliminary glyph from top-30 scored crystals
        # This drives sampling parameters BEFORE we sample
        prelim_glyph = GlyphState.PSI  # default
        prelim_zeta = 0.0
        if self.coherence_engine:
            top_scored = sorted(crystals, key=lambda x: x.get('similarity', 0), reverse=True)[:30]
            if top_scored:
                prelim_zeta = self.coherence_engine.calculate_zeta_lambda(top_scored, query_vec)
                special = self.coherence_engine.detect_special_glyphs(query, top_scored)
                prelim_glyph = special if special else self.coherence_engine.detect_glyph(prelim_zeta)

        # Look up glyph-specific sampling params
        glyph_cfg = self.GLYPH_SAMPLING.get(prelim_glyph, self.GLYPH_SAMPLING[GlyphState.PSI])
        theta_steps = glyph_cfg['theta_steps']
        scale_weights = glyph_cfg['weights']

        # Choose sampling method
        if use_lemniscate:
            # Lemniscate sampling (walk the ∞ curve with variance)
            sampled = self._lemniscate_sample(crystals, total_crystals, theta_steps=theta_steps)
            # Still organize by scale for context building
            scales = self._scale_sample(sampled, total_crystals, scale_weights=scale_weights)
        else:
            # Traditional multi-scale sampling
            scales = self._scale_sample(crystals, total_crystals, scale_weights=scale_weights)

        # Combine scales into aligned set
        aligned = scales['macro'] + scales['meso'] + scales['micro']

        # Apply coherence ratio
        challengers = []
        if include_challengers:
            aligned, challengers = self._apply_coherence_ratio(aligned, all_crystals, query_vec)

        # Temporal sampling
        temporal = {}
        if include_temporal:
            temporal = self._temporal_sample(aligned, n=6)

        # Find synapses
        synapses = []
        if include_synapses:
            synapses = self._find_synapses(aligned)

        # Calculate coherence state if engine available
        coherence_state = None
        if self.coherence_engine and aligned:
            zeta = self.coherence_engine.calculate_zeta_lambda(aligned, query_vec)
            glyph = self.coherence_engine.detect_glyph(zeta)
            mode = self.coherence_engine.detect_mode(zeta)
            attractor = self.coherence_engine.detect_attractor(aligned, query)

            coherence_state = {
                'zeta_lambda': round(zeta, 3),
                'glyph': glyph.value,
                'mode': mode.value,
                'attractor': attractor
            }

        # Build context
        context = self._build_context(aligned, challengers, temporal, synapses, scales, coherence_state)

        return {
            'aligned': aligned,
            'challengers': challengers,
            'temporal': temporal,
            'synapses': synapses,
            'scales': {k: len(v) for k, v in scales.items()},
            'coherence': coherence_state,
            'context': context,
            'stats': {
                'total_searched': len(all_crystals),
                'aligned_count': len(aligned),
                'challenger_count': len(challengers),
                'synapse_count': len(synapses),
                'sampling': 'lemniscate' if use_lemniscate else 'scale',
                'prelim_glyph': prelim_glyph.value,
                'prelim_zeta': round(prelim_zeta, 3),
                'theta_steps': theta_steps,
                'scale_weights': scale_weights
            }
        }

    def _build_context(
        self,
        aligned: List[Dict],
        challengers: List[Dict],
        temporal: Dict[str, List[Dict]],
        synapses: List[Dict],
        scales: Dict[str, List[Dict]],
        coherence_state: Dict = None
    ) -> str:
        """Build structured context for LLM with coherence state."""
        parts = []

        # Coherence state header (if available)
        if coherence_state:
            parts.append(f"## Field State: {coherence_state['glyph']} | Zλ={coherence_state['zeta_lambda']} | {coherence_state['mode']} | → {coherence_state['attractor']}\n")

        # Scale summary
        parts.append("## Multi-Scale View\n")

        # Macro: the big themes
        if scales.get('macro'):
            parts.append("### MACRO (Core Resonance)")
            wounds = Counter(c.get('wound') for c in scales['macro'] if c.get('wound'))
            emotions = Counter(c.get('emotion') for c in scales['macro'] if c.get('emotion'))
            if wounds:
                parts.append(f"Dominant wound: {wounds.most_common(1)[0][0]}")
            if emotions:
                top_emotions = [e for e, _ in emotions.most_common(3)]
                parts.append(f"Emotions: {', '.join(top_emotions)}")
            parts.append("")

        # Meso: related topics
        if scales.get('meso'):
            parts.append("### MESO (Related Threads)")
            for c in scales['meso'][:3]:
                parts.append(f"- {c['content'][:200]}...")
            parts.append("")

        # Micro: specific moments
        if scales.get('micro'):
            parts.append("### MICRO (Specific Moments)")
            for c in scales['micro'][:2]:
                parts.append(f"- {c['content'][:200]}...")
            parts.append("")

        # Challengers (the 1 in 3:1)
        if challengers:
            parts.append("### CHALLENGER (Tension/Contrast)")
            parts.append("*What challenges or inverts the above:*")
            for c in challengers[:2]:
                parts.append(f"- {c['content'][:200]}...")
            parts.append("")

        # Synapses
        if synapses:
            parts.append("### SYNAPSES (Connections)")
            for s in synapses[:3]:
                parts.append(f"- {s['connection']}: {s['similarity']:.2f} resonance")
            parts.append("")

        # Temporal
        if temporal.get('early') or temporal.get('recent'):
            parts.append("### TEMPORAL (Time Thread)")
            if temporal.get('early'):
                parts.append(f"Earlier: {temporal['early'][0]['content'][:100]}...")
            if temporal.get('recent'):
                parts.append(f"Recent: {temporal['recent'][-1]['content'][:100]}...")
            parts.append("")

        # Raw crystals for weaving (not quoting)
        parts.append("## Memory Fragments (read, don't quote)\n")
        for c in aligned[:20]:
            parts.append(f"---\n{c['content'][:1000]}\n")

        return "\n".join(parts)

    def outside_in(self, query: str, depth: int = 3) -> Dict:
        """
        Outside-in search: start broad, narrow down.

        1. Find broad themes
        2. Zoom into most relevant theme
        3. Find specific moments within that theme
        """
        # First pass: broad
        result = self.route(query, total_crystals=100,
                          include_challengers=False,
                          include_temporal=False,
                          include_synapses=False)

        if not result.get('aligned'):
            return result

        # Find dominant theme
        wounds = Counter(c.get('wound') for c in result['aligned'] if c.get('wound'))
        if not wounds:
            return result

        dominant_wound = wounds.most_common(1)[0][0]

        # Second pass: zoom into that theme
        refined_query = f"{query} {dominant_wound}"
        refined = self.route(refined_query, total_crystals=50,
                            include_challengers=True,
                            include_temporal=True,
                            include_synapses=True)

        refined['search_path'] = f"broad → {dominant_wound} → specific"
        return refined

    def inside_out(self, query: str) -> Dict:
        """
        Inside-out search: start specific, expand out.

        1. Find exact matches
        2. Find what connects to those
        3. Expand to broader themes
        """
        # Narrow search first
        result = self.route(query, total_crystals=10,
                          include_challengers=False,
                          include_temporal=False,
                          include_synapses=False)

        if not result.get('aligned'):
            return result

        # Get the core crystal
        core = result['aligned'][0] if result['aligned'] else None
        if not core:
            return result

        # Use core's wound/emotion to expand
        expansion_terms = []
        if core.get('wound'):
            expansion_terms.append(core['wound'])
        if core.get('emotion'):
            expansion_terms.append(core['emotion'])

        if not expansion_terms:
            return result

        # Expand search
        expanded_query = f"{query} {' '.join(expansion_terms)}"
        expanded = self.route(expanded_query, total_crystals=50,
                             include_challengers=True,
                             include_temporal=True,
                             include_synapses=True)

        expanded['search_path'] = f"specific → {' + '.join(expansion_terms)} → broad"
        return expanded


# CLI for testing
if __name__ == "__main__":
    import sys

    router = SmartRouter()

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What am I avoiding?"

    print(f"Query: {query}\n")
    print("=" * 60)

    result = router.route(query)

    print(f"Stats: {result.get('stats', {})}")
    print(f"Scales: {result.get('scales', {})}")
    print(f"\nSynapses found: {len(result.get('synapses', []))}")

    for s in result.get('synapses', [])[:3]:
        print(f"  - {s['connection']}")

    print("\n" + "=" * 60)
    print("CONTEXT (first 2000 chars):")
    print("=" * 60)
    print(result.get('context', '')[:2000])
