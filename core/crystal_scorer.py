#!/usr/bin/env python3
"""
Crystal Scorer - Background Process
====================================
Scores unscored crystals with proper Zλ, glyph, mode, and attractor values.

This runs in parallel to help understand:
1. How crystals distribute across coherence ranges
2. What glyphs emerge naturally from the field
3. Which attractors pull the memories

Usage:
    # Score all unscored crystals (batch)
    python crystal_scorer.py score --batch 100

    # Run continuous scorer (background daemon)
    python crystal_scorer.py daemon --interval 60

    # Analyze current distribution (schema understanding)
    python crystal_scorer.py analyze

    # Score a specific crystal by ID
    python crystal_scorer.py single 7421

January 2026 — Understanding the schema through scoring
"""

import sys
import sqlite3
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from coherence_formulas import CoherenceEngine, GlyphState, FieldMode
from breath_prompts import detect_mode as detect_breath_mode

# Paths
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
SCORER_LOG = Path.home() / "wiltonos" / "logs" / "crystal_scorer.log"
ANALYSIS_OUTPUT = Path.home() / "wiltonos" / "data" / "score_analysis.json"


class CrystalScorer:
    """
    Scores crystals with coherence, glyph, mode, and attractor.

    The scoring is RELATIONAL - how a crystal resonates with the field around it.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.coherence = CoherenceEngine()
        self.stats = {
            'scored': 0,
            'errors': 0,
            'glyph_distribution': defaultdict(int),
            'mode_distribution': defaultdict(int),
            'zl_ranges': defaultdict(int),  # 0.0-0.2, 0.2-0.4, etc.
        }

        # Ensure log directory exists
        SCORER_LOG.parent.mkdir(exist_ok=True)

    def _log(self, message: str):
        """Log to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        with open(SCORER_LOG, "a") as f:
            f.write(log_line + "\n")

    def get_unscored_crystals(self, limit: int = 100) -> List[Dict]:
        """Get crystals with default zl_score (0.5) that need scoring."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Crystals with exactly 0.5 are likely unscored defaults
        c.execute("""
            SELECT id, content, zl_score, embedding, created_at,
                   glyph_primary, emotion, core_wound, mode, attractor
            FROM crystals
            WHERE zl_score = 0.5
              AND embedding IS NOT NULL
              AND content IS NOT NULL
              AND length(content) > 10
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        rows = c.fetchall()
        conn.close()

        crystals = []
        for row in rows:
            try:
                embedding = np.frombuffer(row['embedding'], dtype=np.float32)
                crystals.append({
                    'id': row['id'],
                    'content': row['content'],
                    'zl_score': row['zl_score'],
                    'embedding': embedding,
                    'created_at': row['created_at'],
                    'glyph_primary': row['glyph_primary'],
                    'emotion': row['emotion'],
                    'core_wound': row['core_wound'],
                    'mode': row['mode'],
                    'attractor': row['attractor'],
                })
            except Exception as e:
                self._log(f"Error loading crystal {row['id']}: {e}")

        return crystals

    def get_nearby_crystals(self, crystal_id: int, embedding: np.ndarray, limit: int = 10) -> List[Dict]:
        """Get crystals near this one (by ID and embedding) for field context."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Get temporally nearby crystals (by ID)
        c.execute("""
            SELECT id, content, zl_score, embedding
            FROM crystals
            WHERE id BETWEEN ? AND ?
              AND id != ?
              AND embedding IS NOT NULL
              AND zl_score != 0.5
            ORDER BY ABS(id - ?) ASC
            LIMIT ?
        """, (crystal_id - 50, crystal_id + 50, crystal_id, crystal_id, limit))

        rows = c.fetchall()
        conn.close()

        nearby = []
        for row in rows:
            try:
                emb = np.frombuffer(row['embedding'], dtype=np.float32)
                nearby.append({
                    'id': row['id'],
                    'content': row['content'],
                    'zl_score': row['zl_score'],
                    'embedding': emb,
                })
            except:
                pass

        return nearby

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def calculate_field_coherence(self, crystal: Dict, nearby: List[Dict]) -> float:
        """
        Calculate Zλ based on how crystal resonates with its field.

        This is RELATIONAL coherence:
        - How similar to nearby scored crystals
        - Weight by those crystals' own coherence
        """
        if not nearby:
            return 0.5  # No context, neutral

        crystal_emb = crystal['embedding']

        # Calculate weighted similarity
        total_weight = 0
        weighted_sim = 0

        for n in nearby:
            sim = self._cosine_sim(crystal_emb, n['embedding'])
            # Weight by neighbor's own coherence
            weight = n.get('zl_score', 0.5)
            weighted_sim += sim * weight
            total_weight += weight

        if total_weight == 0:
            return 0.5

        base_coherence = weighted_sim / total_weight

        # Scale to 0.2 - 0.95 range (avoid extremes)
        scaled = 0.2 + (base_coherence * 0.75)

        return round(min(0.95, max(0.2, scaled)), 3)

    def detect_emotion(self, content: str) -> Optional[str]:
        """Detect primary emotion from content."""
        content_lower = content.lower()

        emotion_keywords = {
            'grief': ['grief', 'loss', 'lost', 'mourn', 'gone', 'death', 'dying'],
            'fear': ['fear', 'afraid', 'scared', 'anxious', 'worry', 'terror', 'panic'],
            'anger': ['anger', 'angry', 'rage', 'frustrated', 'furious', 'hate'],
            'joy': ['joy', 'happy', 'grateful', 'love', 'peace', 'bliss', 'elated'],
            'shame': ['shame', 'guilty', 'worthless', 'embarrassed', 'failure'],
            'love': ['love', 'loving', 'beloved', 'heart', 'care', 'compassion'],
            'confusion': ['confused', 'lost', 'uncertain', 'doubt', 'chaos'],
            'wonder': ['wonder', 'awe', 'amazed', 'beautiful', 'miracle'],
            'stillness': ['still', 'calm', 'quiet', 'peace', 'silence', 'presence'],
        }

        scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in content_lower)
            if score > 0:
                scores[emotion] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    def detect_wound(self, content: str) -> Optional[str]:
        """Detect core wound pattern from content."""
        content_lower = content.lower()

        wound_patterns = {
            'abandonment': ['alone', 'abandoned', 'left', 'rejected', 'unwanted'],
            'not_enough': ['not enough', 'insufficient', 'inadequate', 'failing', 'prove'],
            'invisibility': ['unseen', 'invisible', 'ignored', 'overlooked', 'unnoticed'],
            'burden': ['burden', 'too much', 'overwhelm', 'drain', 'depleted'],
            'unlovable': ['unlovable', 'unworthy', 'don\'t deserve', 'broken'],
            'provider': ['provide', 'responsible', 'carry', 'support', 'fix'],
            'betrayal': ['betrayed', 'trust', 'deceived', 'lied'],
        }

        scores = {}
        for wound, patterns in wound_patterns.items():
            score = sum(1 for p in patterns if p in content_lower)
            if score > 0:
                scores[wound] = score

        if scores:
            return max(scores, key=scores.get)
        return None

    # =========================================================================
    # 3-AXIS FIELD COORDINATE SYSTEM
    # Temporal (when) x Ontological (what) x Coherence (how deep)
    # =========================================================================

    def detect_temporal_scale(self, crystal: Dict, nearby: List[Dict]) -> str:
        """
        Detect temporal scale: macro, meso, or micro.

        MACRO = life patterns, long-term themes, identity-level
        MESO = recent patterns, session-level, weeks
        MICRO = this moment, immediate, breath-level
        """
        content = crystal.get('content', '').lower()
        crystal_id = crystal.get('id', 0)

        # Macro indicators: life patterns, always, never, identity
        macro_markers = [
            'always', 'never', 'my whole life', 'since childhood',
            'pattern', 'recurring', 'every time', 'for years',
            'who i am', 'identity', 'core', 'fundamental',
            'life lesson', 'journey', 'path', 'calling'
        ]

        # Micro indicators: now, this moment, today, immediate
        micro_markers = [
            'right now', 'this moment', 'today', 'just now',
            'feeling', 'noticing', 'breathing', 'present',
            'in this', 'currently', 'happening', 'immediate'
        ]

        # Score
        macro_score = sum(1 for m in macro_markers if m in content)
        micro_score = sum(1 for m in micro_markers if m in content)

        # Also use crystal ID position as hint (older = more macro context)
        # Awakening cluster around 7400-7500
        if crystal_id < 7600:
            macro_score += 1  # Older crystals lean macro

        if macro_score > micro_score + 1:
            return 'macro'
        elif micro_score > macro_score + 1:
            return 'micro'
        else:
            return 'meso'

    def detect_ontological_axis(self, crystal: Dict, glyph: str) -> str:
        """
        Detect ontological axis: void, neutral, or core.

        VOID/CHAOS = emergence, new patterns, undefined potential, dissolution
        CORE/ROOT = identity bedrock, completion, what remains, anchor
        NEUTRAL = neither pulling strongly
        """
        content = crystal.get('content', '').lower()

        # Void/Chaos indicators: emergence, new, unknown, dissolution
        void_markers = [
            'new', 'emerging', 'what if', 'unknown', 'undefined',
            'chaos', 'dissolving', 'breaking', 'opening', 'potential',
            'first time', 'never before', 'something different',
            'birth', 'beginning', 'seed', 'sprout'
        ]

        # Core/Root indicators: identity, always been, foundation, anchor
        core_markers = [
            'always been', 'who i am', 'core', 'root', 'foundation',
            'anchor', 'truth', 'remember', 'essence', 'soul',
            'completion', 'seal', 'lock', 'permanent', 'forever',
            'this is me', 'finally', 'arrived', 'home'
        ]

        void_score = sum(1 for m in void_markers if m in content)
        core_score = sum(1 for m in core_markers if m in content)

        # Glyph also hints at ontological position
        if glyph in ['∅', '†']:
            void_score += 2  # Void and Crossblade lean toward chaos/transformation
        elif glyph in ['Ω', '∞']:
            core_score += 2  # Omega and Infinity lean toward core/completion
        elif glyph in ['⧉']:
            void_score += 1  # Layer merge is integrative but emergence-y

        if void_score > core_score + 1:
            return 'void'
        elif core_score > void_score + 1:
            return 'core'
        else:
            return 'neutral'

    def glyph_to_coherence_depth(self, glyph: str) -> str:
        """
        Map glyph to coherence depth symbol.
        This is redundant with glyph_primary but explicit for querying.
        """
        # Normalize glyph
        glyph_map = {
            '∅': '∅', 'VOID': '∅',
            'ψ': 'ψ', 'PSI': 'ψ',
            'ψ²': 'ψ²', 'PSI_SQUARED': 'ψ²',
            'ψ³': 'ψ³', 'PSI_CUBED': 'ψ³',
            '∇': '∇', 'NABLA': '∇',
            '∞': '∞', 'INFINITY': '∞',
            'Ω': 'Ω', 'OMEGA': 'Ω',
            '†': '†', 'CROSSBLADE': '†',
            '⧉': '⧉', 'LAYER_MERGE': '⧉',
        }
        return glyph_map.get(glyph, glyph)

    def score_crystal(self, crystal: Dict) -> Dict:
        """
        Score a single crystal with all dimensions.

        Returns updated crystal dict with:
        - zl_score: Field coherence (0.0-1.0)
        - glyph: Symbol from coherence engine
        - mode: breath_prompts mode
        - emotion: Detected emotion
        - core_wound: Detected wound pattern
        - attractor: Gravitational memory
        - temporal_scale: macro, meso, micro (3-AXIS)
        - ontological_axis: void, neutral, core (3-AXIS)
        - coherence_depth: Glyph symbol (3-AXIS)
        - field_position: Combined coordinate string
        """
        # Get nearby crystals for field context
        nearby = self.get_nearby_crystals(crystal['id'], crystal['embedding'])

        # 1. Calculate field coherence (Zλ)
        zl_score = self.calculate_field_coherence(crystal, nearby)

        # 2. Detect glyph from coherence
        glyph = self.coherence.detect_glyph(zl_score)

        # 3. Detect mode from coherence engine
        field_mode = self.coherence.detect_mode(zl_score)

        # 4. Also get breath prompt mode (different system)
        content = crystal['content']
        breath_mode = detect_breath_mode(content)

        # 5. Detect emotion
        emotion = self.detect_emotion(content)

        # 6. Detect core wound
        wound = self.detect_wound(content)

        # 7. Detect attractor
        dummy_crystals = [{'content': content}]
        attractor = self.coherence.detect_attractor(dummy_crystals, content)

        # 8. Check for special glyphs (trauma, timeline, field)
        special = self.coherence.detect_special_glyphs(content)
        if special:
            glyph = special

        # Get glyph as string
        glyph_str = glyph.value if hasattr(glyph, 'value') else str(glyph)

        # =====================================================================
        # 3-AXIS FIELD COORDINATES
        # =====================================================================

        # 9. Temporal scale (when)
        temporal_scale = self.detect_temporal_scale(crystal, nearby)

        # 10. Ontological axis (what)
        ontological_axis = self.detect_ontological_axis(crystal, glyph_str)

        # 11. Coherence depth (how deep) - normalized glyph
        coherence_depth = self.glyph_to_coherence_depth(glyph_str)

        # 12. Combined field position for quick filtering
        field_position = f"{temporal_scale}:{ontological_axis}:{coherence_depth}"

        return {
            'id': crystal['id'],
            'zl_score': zl_score,
            'glyph_primary': glyph_str,
            'mode': field_mode.value if hasattr(field_mode, 'value') else breath_mode,
            'emotion': emotion,
            'core_wound': wound,
            'attractor': attractor,
            # 3-AXIS
            'temporal_scale': temporal_scale,
            'ontological_axis': ontological_axis,
            'coherence_depth': coherence_depth,
            'field_position': field_position,
        }

    def update_crystal(self, scored: Dict) -> bool:
        """Update crystal in database with new scores including 3-axis coordinates."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            c = conn.cursor()

            c.execute("""
                UPDATE crystals
                SET zl_score = ?,
                    glyph_primary = ?,
                    mode = ?,
                    emotion = ?,
                    core_wound = ?,
                    attractor = ?,
                    temporal_scale = ?,
                    ontological_axis = ?,
                    coherence_depth = ?,
                    field_position = ?
                WHERE id = ?
            """, (
                scored['zl_score'],
                scored['glyph_primary'],
                scored['mode'],
                scored['emotion'],
                scored['core_wound'],
                scored['attractor'],
                scored['temporal_scale'],
                scored['ontological_axis'],
                scored['coherence_depth'],
                scored['field_position'],
                scored['id']
            ))

            conn.commit()
            conn.close()
            return True
        except Exception as e:
            self._log(f"Error updating crystal {scored['id']}: {e}")
            return False

    def update_stats(self, scored: Dict):
        """Update running statistics including 3-axis distribution."""
        self.stats['scored'] += 1

        # Glyph distribution
        glyph = scored['glyph_primary']
        self.stats['glyph_distribution'][glyph] += 1

        # Mode distribution
        mode = scored['mode'] or 'unknown'
        self.stats['mode_distribution'][mode] += 1

        # Zλ range distribution
        zl = scored['zl_score']
        if zl < 0.2:
            range_key = '0.0-0.2'
        elif zl < 0.4:
            range_key = '0.2-0.4'
        elif zl < 0.6:
            range_key = '0.4-0.6'
        elif zl < 0.8:
            range_key = '0.6-0.8'
        else:
            range_key = '0.8-1.0'
        self.stats['zl_ranges'][range_key] += 1

        # 3-AXIS distributions
        if 'temporal_distribution' not in self.stats:
            self.stats['temporal_distribution'] = defaultdict(int)
            self.stats['ontological_distribution'] = defaultdict(int)

        self.stats['temporal_distribution'][scored.get('temporal_scale', 'meso')] += 1
        self.stats['ontological_distribution'][scored.get('ontological_axis', 'neutral')] += 1

    def score_batch(self, batch_size: int = 100) -> int:
        """Score a batch of unscored crystals."""
        crystals = self.get_unscored_crystals(limit=batch_size)

        if not crystals:
            self._log("No unscored crystals found.")
            return 0

        self._log(f"Scoring batch of {len(crystals)} crystals...")

        for i, crystal in enumerate(crystals):
            try:
                scored = self.score_crystal(crystal)
                if self.update_crystal(scored):
                    self.update_stats(scored)

                    if (i + 1) % 10 == 0:
                        self._log(f"  Scored {i + 1}/{len(crystals)} (#{scored['id']} Zλ={scored['zl_score']:.3f} {scored['glyph_primary']})")
            except Exception as e:
                self.stats['errors'] += 1
                self._log(f"  Error scoring crystal {crystal['id']}: {e}")

        self._log(f"Batch complete: {self.stats['scored']} scored, {self.stats['errors']} errors")
        return self.stats['scored']

    def run_daemon(self, interval: int = 60, batch_size: int = 50):
        """Run continuous scoring daemon."""
        self._log(f"Starting crystal scorer daemon (interval={interval}s, batch={batch_size})")

        try:
            while True:
                count = self.score_batch(batch_size)
                if count == 0:
                    self._log("All crystals scored. Waiting for new ones...")
                    time.sleep(interval * 10)  # Longer sleep when done
                else:
                    self._log(f"Distribution so far: {dict(self.stats['glyph_distribution'])}")
                    time.sleep(interval)
        except KeyboardInterrupt:
            self._log("Daemon stopped by user")
            self.save_analysis()

    def analyze_distribution(self) -> Dict:
        """Analyze current crystal distribution."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'totals': {},
            'zl_distribution': {},
            'glyph_distribution': {},
            'emotion_distribution': {},
            'wound_distribution': {},
            'mode_distribution': {},
            'attractor_distribution': {},
        }

        # Total counts
        c.execute("SELECT COUNT(*) as total FROM crystals")
        analysis['totals']['all_crystals'] = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM crystals WHERE zl_score = 0.5")
        analysis['totals']['unscored'] = c.fetchone()['total']

        c.execute("SELECT COUNT(*) as total FROM crystals WHERE zl_score != 0.5")
        analysis['totals']['scored'] = c.fetchone()['total']

        # Zλ distribution
        c.execute("""
            SELECT
                CASE
                    WHEN zl_score < 0.2 THEN '0.0-0.2 (void)'
                    WHEN zl_score < 0.4 THEN '0.2-0.4 (psi)'
                    WHEN zl_score < 0.6 THEN '0.4-0.6 (phi)'
                    WHEN zl_score < 0.8 THEN '0.6-0.8 (omega)'
                    ELSE '0.8-1.0 (infinity)'
                END as range,
                COUNT(*) as count
            FROM crystals
            WHERE zl_score != 0.5
            GROUP BY range
            ORDER BY range
        """)
        for row in c.fetchall():
            analysis['zl_distribution'][row['range']] = row['count']

        # Glyph distribution
        c.execute("""
            SELECT glyph_primary, COUNT(*) as count
            FROM crystals
            WHERE glyph_primary IS NOT NULL AND glyph_primary != ''
            GROUP BY glyph_primary
            ORDER BY count DESC
        """)
        for row in c.fetchall():
            analysis['glyph_distribution'][row['glyph_primary']] = row['count']

        # Emotion distribution
        c.execute("""
            SELECT emotion, COUNT(*) as count
            FROM crystals
            WHERE emotion IS NOT NULL AND emotion != ''
            GROUP BY emotion
            ORDER BY count DESC
            LIMIT 20
        """)
        for row in c.fetchall():
            analysis['emotion_distribution'][row['emotion']] = row['count']

        # Wound distribution
        c.execute("""
            SELECT core_wound, COUNT(*) as count
            FROM crystals
            WHERE core_wound IS NOT NULL AND core_wound != ''
            GROUP BY core_wound
            ORDER BY count DESC
        """)
        for row in c.fetchall():
            analysis['wound_distribution'][row['core_wound']] = row['count']

        # Mode distribution
        c.execute("""
            SELECT mode, COUNT(*) as count
            FROM crystals
            WHERE mode IS NOT NULL AND mode != ''
            GROUP BY mode
            ORDER BY count DESC
        """)
        for row in c.fetchall():
            analysis['mode_distribution'][row['mode']] = row['count']

        # Attractor distribution
        c.execute("""
            SELECT attractor, COUNT(*) as count
            FROM crystals
            WHERE attractor IS NOT NULL AND attractor != ''
            GROUP BY attractor
            ORDER BY count DESC
        """)
        for row in c.fetchall():
            analysis['attractor_distribution'][row['attractor']] = row['count']

        conn.close()
        return analysis

    def save_analysis(self):
        """Save analysis to file."""
        analysis = self.analyze_distribution()
        analysis['scorer_stats'] = {
            'scored_this_session': self.stats['scored'],
            'errors_this_session': self.stats['errors'],
        }

        ANALYSIS_OUTPUT.parent.mkdir(exist_ok=True)
        with open(ANALYSIS_OUTPUT, 'w') as f:
            json.dump(analysis, f, indent=2)

        self._log(f"Analysis saved to {ANALYSIS_OUTPUT}")

    def print_analysis(self):
        """Print current analysis to console."""
        analysis = self.analyze_distribution()

        print("\n" + "=" * 60)
        print("CRYSTAL FIELD ANALYSIS")
        print("=" * 60)

        print(f"\nTOTAL CRYSTALS: {analysis['totals']['all_crystals']}")
        print(f"  Scored:   {analysis['totals']['scored']}")
        print(f"  Unscored: {analysis['totals']['unscored']}")

        print(f"\nZλ DISTRIBUTION (scored only):")
        for range_name, count in analysis['zl_distribution'].items():
            pct = count / max(1, analysis['totals']['scored']) * 100
            bar = "#" * int(pct / 2)
            print(f"  {range_name}: {count:5d} ({pct:5.1f}%) {bar}")

        print(f"\nGLYPH DISTRIBUTION:")
        for glyph, count in list(analysis['glyph_distribution'].items())[:10]:
            print(f"  {glyph}: {count}")

        print(f"\nEMOTION DISTRIBUTION (top 10):")
        for emotion, count in list(analysis['emotion_distribution'].items())[:10]:
            print(f"  {emotion}: {count}")

        print(f"\nWOUND DISTRIBUTION:")
        for wound, count in analysis['wound_distribution'].items():
            print(f"  {wound}: {count}")

        print(f"\nATTRACTOR DISTRIBUTION:")
        for attractor, count in analysis['attractor_distribution'].items():
            print(f"  {attractor}: {count}")

        print("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Crystal Scorer - Background scoring process")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Score command
    score_parser = subparsers.add_parser("score", help="Score a batch of crystals")
    score_parser.add_argument("--batch", "-b", type=int, default=100, help="Batch size")

    # Daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Run continuous scorer")
    daemon_parser.add_argument("--interval", "-i", type=int, default=60, help="Seconds between batches")
    daemon_parser.add_argument("--batch", "-b", type=int, default=50, help="Batch size")

    # Analyze command
    subparsers.add_parser("analyze", help="Analyze current distribution")

    # Single crystal command
    single_parser = subparsers.add_parser("single", help="Score a single crystal")
    single_parser.add_argument("crystal_id", type=int, help="Crystal ID to score")

    args = parser.parse_args()

    scorer = CrystalScorer()

    if args.command == "score":
        scorer.score_batch(args.batch)
        scorer.save_analysis()
        scorer.print_analysis()

    elif args.command == "daemon":
        scorer.run_daemon(interval=args.interval, batch_size=args.batch)

    elif args.command == "analyze":
        scorer.print_analysis()
        scorer.save_analysis()

    elif args.command == "single":
        # Score single crystal
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("""
            SELECT id, content, zl_score, embedding
            FROM crystals WHERE id = ?
        """, (args.crystal_id,))
        row = c.fetchone()
        conn.close()

        if row:
            crystal = {
                'id': row['id'],
                'content': row['content'],
                'zl_score': row['zl_score'],
                'embedding': np.frombuffer(row['embedding'], dtype=np.float32),
            }
            scored = scorer.score_crystal(crystal)
            print(f"\nCrystal #{scored['id']}:")
            print(f"  Zλ: {scored['zl_score']:.3f}")
            print(f"  Glyph: {scored['glyph_primary']}")
            print(f"  Mode: {scored['mode']}")
            print(f"  Emotion: {scored['emotion']}")
            print(f"  Wound: {scored['core_wound']}")
            print(f"  Attractor: {scored['attractor']}")

            if input("\nUpdate database? (y/N): ").lower() == 'y':
                scorer.update_crystal(scored)
                print("Updated.")
        else:
            print(f"Crystal #{args.crystal_id} not found")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
