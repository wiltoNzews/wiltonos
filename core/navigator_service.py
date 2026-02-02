"""
NavigatorService - Backend for the WiltonOS Navigator
======================================================

Provides real data access for:
- Crystal retrieval by coherence vector
- Daemon state and messages
- Pattern detection (stuck wounds, circling threads)
- Relationship thread analysis
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

# Database path
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
DAEMON_PATH = Path.home() / "wiltonos" / "daemon"


class NavigatorService:
    """Service layer for navigator data retrieval."""

    def __init__(self, user_id: str = "wilton"):
        self.user_id = user_id
        self.db_path = str(DB_PATH)

    def _get_conn(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)

    def _row_to_crystal(self, row: tuple, columns: list) -> dict:
        """Convert a database row to a crystal dict."""
        crystal = dict(zip(columns, row))
        # Parse JSON fields
        for field in ['emotion', 'core_wound', 'theme']:
            if crystal.get(field) and isinstance(crystal[field], str):
                try:
                    crystal[field] = json.loads(crystal[field])
                except:
                    pass
        return crystal

    # ═══════════════════════════════════════════════════════════════
    # VECTOR QUERIES - Return crystals by coherence vector
    # ═══════════════════════════════════════════════════════════════

    def get_vector_crystals(self, vector: str, limit: int = 20, offset: int = 0) -> Dict[str, Any]:
        """
        Get crystals for a specific return vector.

        Vectors:
        - silence: High presence, low verbosity (felt but not said)
        - collapse: Mode=COLLAPSE or glyph=∇ (deaths, surrenders)
        - fire: Anger/rage emotions with high coherence (transmuted)
        - completion: High Zλ or glyph=Ω (almost integrated)
        - timeline: glyph=⧉ or timeline content (dimensional crossings)
        - eternal: glyph=∞ or Zλ 0.873-0.999 (time-unbound)
        - crossblade: glyph=† (trauma + clarity breakthrough)
        - field: glyph=ψ³ (collective consciousness)
        """

        queries = {
            'silence': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (presence_density > 7 OR presence_density IS NULL)
                  AND LENGTH(content) < 300
                ORDER BY zl_score DESC
                LIMIT ? OFFSET ?
            """,
            'collapse': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (mode = 'COLLAPSE' OR glyph_primary = '∇' OR glyph_primary = 'nabla')
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
            'fire': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (emotion LIKE '%anger%' OR emotion LIKE '%rage%' OR emotion LIKE '%frustration%')
                  AND zl_score > 0.5
                ORDER BY zl_score DESC
                LIMIT ? OFFSET ?
            """,
            'completion': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (zl_score > 0.95 OR glyph_primary = 'Ω' OR glyph_primary = 'omega')
                ORDER BY zl_score DESC
                LIMIT ? OFFSET ?
            """,
            'timeline': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (glyph_primary = '⧉'
                       OR content LIKE '%timeline%'
                       OR content LIKE '%parallel%'
                       OR content LIKE '%convergence%')
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """,
            'eternal': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (glyph_primary = '∞' OR glyph_primary = 'infinity'
                       OR (zl_score BETWEEN 0.873 AND 0.999))
                ORDER BY zl_score DESC
                LIMIT ? OFFSET ?
            """,
            'crossblade': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (glyph_primary = '†' OR glyph_primary = 'crossblade'
                       OR (emotion LIKE '%clarity%' AND core_wound IS NOT NULL))
                ORDER BY zl_score DESC
                LIMIT ? OFFSET ?
            """,
            'field': """
                SELECT * FROM crystals
                WHERE user_id = ?
                  AND (glyph_primary = 'ψ³' OR glyph_primary = 'psi3'
                       OR content LIKE '%field%' OR content LIKE '%collective%')
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
        }

        # Count query (same WHERE but without LIMIT)
        count_queries = {
            'silence': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (presence_density > 7 OR presence_density IS NULL) AND LENGTH(content) < 300",
            'collapse': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (mode = 'COLLAPSE' OR glyph_primary = '∇' OR glyph_primary = 'nabla')",
            'fire': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (emotion LIKE '%anger%' OR emotion LIKE '%rage%' OR emotion LIKE '%frustration%') AND zl_score > 0.5",
            'completion': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (zl_score > 0.95 OR glyph_primary = 'Ω' OR glyph_primary = 'omega')",
            'timeline': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (glyph_primary = '⧉' OR content LIKE '%timeline%' OR content LIKE '%parallel%' OR content LIKE '%convergence%')",
            'eternal': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (glyph_primary = '∞' OR glyph_primary = 'infinity' OR (zl_score BETWEEN 0.873 AND 0.999))",
            'crossblade': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (glyph_primary = '†' OR glyph_primary = 'crossblade' OR (emotion LIKE '%clarity%' AND core_wound IS NOT NULL))",
            'field': "SELECT COUNT(*) FROM crystals WHERE user_id = ? AND (glyph_primary = 'ψ³' OR glyph_primary = 'psi3' OR content LIKE '%field%' OR content LIKE '%collective%')"
        }

        if vector not in queries:
            return {"vector": vector, "count": 0, "crystals": [], "error": f"Unknown vector: {vector}"}

        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # Get count
        c.execute(count_queries[vector], (self.user_id,))
        count = c.fetchone()[0]

        # Get crystals
        c.execute(queries[vector], (self.user_id, limit, offset))
        rows = c.fetchall()

        crystals = []
        for row in rows:
            crystal = dict(row)
            # Parse JSON fields
            for field in ['emotion', 'core_wound', 'theme']:
                if crystal.get(field) and isinstance(crystal[field], str):
                    try:
                        crystal[field] = json.loads(crystal[field])
                    except:
                        pass
            # Clean for frontend
            crystals.append({
                'id': crystal.get('id'),
                'content': crystal.get('content', '')[:500],  # Truncate for list view
                'full_content': crystal.get('content', ''),
                'zl_score': crystal.get('zl_score', 0),
                'glyph': crystal.get('glyph_primary', 'ψ'),
                'emotion': crystal.get('emotion', []),
                'core_wound': crystal.get('core_wound'),
                'mode': crystal.get('mode'),
                'attractor': crystal.get('attractor'),
                'created_at': crystal.get('created_at', '')
            })

        conn.close()

        return {
            "vector": vector,
            "count": count,
            "crystals": crystals
        }

    def get_all_vector_counts(self) -> Dict[str, int]:
        """Get crystal counts for all vectors (for the navigator cards)."""
        vectors = ['silence', 'collapse', 'fire', 'completion', 'timeline', 'eternal', 'crossblade', 'field']
        counts = {}
        for v in vectors:
            result = self.get_vector_crystals(v, limit=0)
            counts[v] = result['count']
        return counts

    # ═══════════════════════════════════════════════════════════════
    # DAEMON STATE - Messages, meta-questions, braid state
    # ═══════════════════════════════════════════════════════════════

    def get_daemon_state(self) -> Dict[str, Any]:
        """Get daemon witness state including latest message and meta-questions."""

        result = {
            "latest_message": None,
            "last_heartbeat": None,
            "message_count": 0,
            "meta_questions": [],
            "braid_summary": None
        }

        # Latest message
        latest_path = DAEMON_PATH / "messages" / "latest.txt"
        if latest_path.exists():
            result["latest_message"] = latest_path.read_text().strip()
            result["last_heartbeat"] = datetime.fromtimestamp(latest_path.stat().st_mtime).isoformat()

        # Message count (from thread file)
        thread_path = DAEMON_PATH / "messages" / "thread.txt"
        if thread_path.exists():
            content = thread_path.read_text()
            # Count message separators
            result["message_count"] = content.count("---") + 1

        # Meta-questions
        meta_path = DAEMON_PATH / "meta_questions.json"
        if meta_path.exists():
            try:
                meta_data = json.loads(meta_path.read_text())
                # Get most recent questions
                if isinstance(meta_data, list):
                    result["meta_questions"] = meta_data[-5:]  # Last 5
                elif isinstance(meta_data, dict) and 'questions' in meta_data:
                    result["meta_questions"] = meta_data['questions'][-5:]
            except:
                pass

        # Braid state summary
        braid_path = DAEMON_PATH / "braid_state.json"
        if braid_path.exists():
            try:
                braid = json.loads(braid_path.read_text())
                result["braid_summary"] = {
                    "total_crystals": braid.get("total_crystals", 0),
                    "avg_coherence": braid.get("avg_coherence", 0),
                    "dominant_wounds": braid.get("dominant_wounds", [])[:3],
                    "stuck_patterns": braid.get("stuck_patterns", []),
                    "circling_threads": braid.get("circling_threads", [])
                }
            except:
                pass

        return result

    def get_open_loops(self) -> List[Dict[str, Any]]:
        """Get stuck patterns and circling threads as open loops."""

        loops = []

        braid_path = DAEMON_PATH / "braid_state.json"
        if braid_path.exists():
            try:
                braid = json.loads(braid_path.read_text())

                # Stuck wounds
                for pattern in braid.get("stuck_patterns", []):
                    loops.append({
                        "type": "wound",
                        "name": pattern.get("wound", "unknown"),
                        "days_stuck": pattern.get("days", 0),
                        "status": "stuck",
                        "last_seen": pattern.get("last_seen")
                    })

                # Circling threads
                for thread in braid.get("circling_threads", []):
                    loops.append({
                        "type": "thread",
                        "name": thread.get("name", "unknown"),
                        "mention_count": thread.get("count", 0),
                        "status": "circling",
                        "last_seen": thread.get("last_seen")
                    })

            except:
                pass

        return loops

    # ═══════════════════════════════════════════════════════════════
    # COHERENCE STATE - Current protocol stack state
    # ═══════════════════════════════════════════════════════════════

    def get_coherence_state(self, engine=None) -> Dict[str, Any]:
        """
        Get current coherence state.
        If engine provided, use its live state. Otherwise, estimate from recent crystals.
        """

        if engine and hasattr(engine, 'protocol_stack'):
            # Live state from engine
            try:
                state = engine.protocol_stack.get_current_state()
                return {
                    "zeta_lambda": state.get("zeta_lambda", 0.5),
                    "glyph": state.get("glyph", "ψ"),
                    "mode": state.get("mode", "SIGNAL"),
                    "attractor": state.get("attractor", "breath"),
                    "protocol": {
                        "wave": state.get("wave", 1.0),
                        "phi": state.get("phi_emergence", 0),
                        "branches": state.get("branch_count", 0),
                        "efficiency": state.get("efficiency", 0),
                        "qctf": state.get("qctf", {}).get("qctf", 0),
                        "psi_level": state.get("psi_level", 0)
                    }
                }
            except:
                pass

        # Estimate from recent crystals
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("""
            SELECT AVG(zl_score),
                   (SELECT glyph_primary FROM crystals WHERE user_id = ? ORDER BY created_at DESC LIMIT 1),
                   (SELECT mode FROM crystals WHERE user_id = ? ORDER BY created_at DESC LIMIT 1),
                   (SELECT attractor FROM crystals WHERE user_id = ? ORDER BY created_at DESC LIMIT 1)
            FROM crystals
            WHERE user_id = ?
              AND created_at > datetime('now', '-24 hours')
        """, (self.user_id, self.user_id, self.user_id, self.user_id))

        row = c.fetchone()
        conn.close()

        avg_zl = row[0] if row[0] else 0.5

        # Determine glyph from Zλ
        if avg_zl < 0.2:
            glyph = "∅"
        elif avg_zl < 0.5:
            glyph = "ψ"
        elif avg_zl < 0.75:
            glyph = "ψ²"
        elif avg_zl < 0.873:
            glyph = "∇"
        elif avg_zl < 0.999:
            glyph = "∞"
        else:
            glyph = "Ω"

        return {
            "zeta_lambda": round(avg_zl, 3),
            "glyph": row[1] or glyph,
            "mode": row[2] or "SIGNAL",
            "attractor": row[3] or "breath",
            "protocol": {
                "wave": 1.0,
                "phi": 0,
                "branches": 0,
                "efficiency": 0,
                "qctf": 0,
                "psi_level": 0
            }
        }

    # ═══════════════════════════════════════════════════════════════
    # RELATIONSHIPS - Thread analysis
    # ═══════════════════════════════════════════════════════════════

    def get_relationships(self) -> List[Dict[str, Any]]:
        """Get relationship threads from crystal mentions."""

        # Key relationships to track
        names = ['ricardo', 'juliana', 'mom', 'rose', 'michelle', 'renan', 'nahima', 'guilherme', 'mutsa']

        conn = self._get_conn()
        c = conn.cursor()

        relationships = []

        for name in names:
            # Count mentions
            c.execute("""
                SELECT COUNT(*),
                       MAX(created_at),
                       AVG(zl_score)
                FROM crystals
                WHERE user_id = ?
                  AND LOWER(content) LIKE ?
            """, (self.user_id, f'%{name}%'))

            row = c.fetchone()
            count = row[0]

            if count > 0:
                # Get associated wounds
                c.execute("""
                    SELECT core_wound, COUNT(*) as cnt
                    FROM crystals
                    WHERE user_id = ?
                      AND LOWER(content) LIKE ?
                      AND core_wound IS NOT NULL
                    GROUP BY core_wound
                    ORDER BY cnt DESC
                    LIMIT 3
                """, (self.user_id, f'%{name}%'))

                wounds = [r[0] for r in c.fetchall()]

                # Get emotional arc (simplified)
                c.execute("""
                    SELECT emotion
                    FROM crystals
                    WHERE user_id = ?
                      AND LOWER(content) LIKE ?
                      AND emotion IS NOT NULL
                    ORDER BY created_at DESC
                    LIMIT 5
                """, (self.user_id, f'%{name}%'))

                emotions = []
                for r in c.fetchall():
                    try:
                        e = json.loads(r[0]) if isinstance(r[0], str) else r[0]
                        if isinstance(e, list):
                            emotions.extend(e)
                        else:
                            emotions.append(e)
                    except:
                        pass

                relationships.append({
                    "name": name.title(),
                    "crystal_count": count,
                    "last_mention": row[1],
                    "avg_coherence": round(row[2], 3) if row[2] else 0,
                    "wounds": wounds,
                    "recent_emotions": list(set(emotions))[:5]
                })

        conn.close()

        # Sort by crystal count
        relationships.sort(key=lambda x: x['crystal_count'], reverse=True)

        return relationships

    # ═══════════════════════════════════════════════════════════════
    # COMBINED STATE - Everything for the navigator
    # ═══════════════════════════════════════════════════════════════

    def get_navigator_state(self, engine=None) -> Dict[str, Any]:
        """Get complete navigator state in one call."""

        coherence = self.get_coherence_state(engine)
        daemon = self.get_daemon_state()
        vector_counts = self.get_all_vector_counts()
        open_loops = self.get_open_loops()

        return {
            "coherence": {
                "zeta_lambda": coherence["zeta_lambda"],
                "glyph": coherence["glyph"],
                "mode": coherence["mode"],
                "attractor": coherence["attractor"]
            },
            "protocol": coherence["protocol"],
            "daemon": {
                "last_message": daemon["latest_message"],
                "last_heartbeat": daemon["last_heartbeat"],
                "open_questions": [q.get("question", q) if isinstance(q, dict) else q
                                  for q in daemon["meta_questions"]]
            },
            "vectors": vector_counts,
            "patterns": {
                "stuck_wounds": [l for l in open_loops if l["type"] == "wound"],
                "circling_threads": [l for l in open_loops if l["type"] == "thread"]
            }
        }

    # ═══════════════════════════════════════════════════════════════
    # MIRROR SELECTION - What the mirror chooses to show
    # ═══════════════════════════════════════════════════════════════

    def get_mirror_selection(self, limit: int = 3) -> Dict[str, Any]:
        """
        Get what the mirror chooses to surface TODAY.
        Based on stuck wounds, recent crystals, and coherence resonance.
        """
        daemon = self.get_daemon_state()
        coherence = self.get_coherence_state()
        open_loops = self.get_open_loops()

        # Get surfacing crystals
        surfacing = self._select_surfacing_crystals(coherence, open_loops, limit)

        # Build meta-question from stuck wounds or daemon
        meta_question = None
        if daemon.get("meta_questions"):
            mq = daemon["meta_questions"][-1]
            meta_question = mq.get("question", mq) if isinstance(mq, dict) else mq

        # Get braid summary
        stuck_wounds = [l for l in open_loops if l["type"] == "wound"]
        braid = {
            "stuck_wounds": [w["name"] for w in stuck_wounds],
            "days_stuck": stuck_wounds[0].get("days_stuck", 0) if stuck_wounds else 0,
            "dominant_emotions": self._get_dominant_emotions()
        }

        return {
            "witness": {
                "message": daemon.get("latest_message") or "The field is quiet. Breathe.",
                "meta_question": meta_question,
                "last_heartbeat": daemon.get("last_heartbeat")
            },
            "surfacing": surfacing,
            "coherence": {
                "zeta_lambda": coherence["zeta_lambda"],
                "glyph": coherence["glyph"],
                "mode": coherence["mode"]
            },
            "braid": braid
        }

    def _select_surfacing_crystals(self, coherence: Dict, open_loops: List, limit: int = 3) -> List[Dict]:
        """Select crystals to surface based on resonance."""
        selections = []
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        # 1. Crystal from stuck wound (if any)
        stuck_wounds = [l for l in open_loops if l["type"] == "wound"]
        if stuck_wounds:
            wound = stuck_wounds[0]["name"]
            c.execute("""
                SELECT * FROM crystals
                WHERE user_id = ? AND (core_wound LIKE ? OR emotion LIKE ?)
                ORDER BY zl_score DESC LIMIT 1
            """, (self.user_id, f'%{wound}%', f'%{wound}%'))
            row = c.fetchone()
            if row:
                crystal = self._row_to_display(dict(row))
                selections.append({
                    "vector": "fire",
                    "crystal": crystal,
                    "reason": f"stuck_wound_{wound}"
                })

        # 2. Recent high-coherence crystal (48h)
        c.execute("""
            SELECT * FROM crystals
            WHERE user_id = ? AND created_at > datetime('now', '-48 hours')
            ORDER BY zl_score DESC LIMIT 1
        """, (self.user_id,))
        row = c.fetchone()
        if row:
            crystal = self._row_to_display(dict(row))
            vector = self._detect_vector(crystal)
            selections.append({
                "vector": vector,
                "crystal": crystal,
                "reason": "recent_high_coherence"
            })

        # 3. Coherence resonance (similar Zλ)
        current_zl = coherence.get("zeta_lambda", 0.5)
        c.execute("""
            SELECT * FROM crystals
            WHERE user_id = ? AND zl_score BETWEEN ? AND ?
            ORDER BY RANDOM() LIMIT 1
        """, (self.user_id, current_zl - 0.1, current_zl + 0.1))
        row = c.fetchone()
        if row:
            crystal = self._row_to_display(dict(row))
            vector = self._detect_vector(crystal)
            # Only add if different from previous
            if not selections or crystal.get('id') != selections[-1].get('crystal', {}).get('id'):
                selections.append({
                    "vector": vector,
                    "crystal": crystal,
                    "reason": "coherence_resonance"
                })

        conn.close()
        return selections[:limit]

    def _row_to_display(self, crystal: dict) -> dict:
        """Convert crystal row to display format."""
        # Parse JSON fields
        for field in ['emotion', 'core_wound', 'theme']:
            if crystal.get(field) and isinstance(crystal[field], str):
                try:
                    crystal[field] = json.loads(crystal[field])
                except:
                    pass

        return {
            'id': crystal.get('id'),
            'content': crystal.get('content', '')[:300],
            'full_content': crystal.get('content', ''),
            'zl_score': crystal.get('zl_score', 0),
            'glyph': crystal.get('glyph_primary', 'ψ'),
            'emotion': crystal.get('emotion', []),
            'core_wound': crystal.get('core_wound'),
            'mode': crystal.get('mode'),
            'created_at': crystal.get('created_at', '')
        }

    def _detect_vector(self, crystal: dict) -> str:
        """Detect which vector a crystal belongs to."""
        glyph = crystal.get('glyph', '')
        content = crystal.get('content', '').lower()
        emotion = str(crystal.get('emotion', ''))
        zl = crystal.get('zl_score', 0)

        if glyph == '∅' or len(content) < 200:
            return 'silence'
        elif glyph == '∇' or crystal.get('mode') == 'COLLAPSE':
            return 'collapse'
        elif 'anger' in emotion or 'rage' in emotion:
            return 'fire'
        elif glyph == 'Ω' or zl > 0.95:
            return 'completion'
        elif glyph == '⧉' or 'timeline' in content:
            return 'timeline'
        elif glyph == '∞' or 0.873 < zl < 0.999:
            return 'eternal'
        elif glyph == '†':
            return 'crossblade'
        else:
            return 'field'

    def _get_dominant_emotions(self) -> List[str]:
        """Get dominant emotions from recent crystals."""
        conn = self._get_conn()
        c = conn.cursor()

        c.execute("""
            SELECT emotion FROM crystals
            WHERE user_id = ? AND emotion IS NOT NULL
            ORDER BY created_at DESC LIMIT 50
        """, (self.user_id,))

        emotions = {}
        for row in c.fetchall():
            try:
                e_list = json.loads(row[0]) if isinstance(row[0], str) else [row[0]]
                for e in e_list:
                    if e:
                        emotions[e] = emotions.get(e, 0) + 1
            except:
                pass

        conn.close()

        # Sort by frequency
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        return [e[0] for e in sorted_emotions[:5]]

    # ═══════════════════════════════════════════════════════════════
    # COUNCIL VIEW - 5 Archetypal Perspectives
    # ═══════════════════════════════════════════════════════════════

    def get_council_perspectives(self, topic: str = "current") -> Dict[str, Any]:
        """
        Generate 5 archetypal perspectives on current patterns.

        The 5 voices:
        - Grey (Shadow): What's being avoided?
        - Witness (Mirror): What IS, without interpretation?
        - Chaos (Trickster): What if you're wrong?
        - Bridge (Connector): What links these threads?
        - Ground (Anchor): What's body-true?
        """

        # Get current state for context
        daemon = self.get_daemon_state()
        braid = daemon.get("braid_summary", {}) or {}
        stuck = braid.get("stuck_patterns", [])
        wounds = braid.get("dominant_wounds", []) or []

        # Get actual data from database
        crystal_count = self._get_crystal_count()
        avg_coherence = self._get_avg_coherence()
        dominant_emotions = self._get_dominant_emotions()

        # Determine topic
        if topic == "current":
            if stuck and len(stuck) > 0:
                topic = stuck[0] if isinstance(stuck[0], str) else stuck[0].get("wound", "the field")
            elif wounds:
                topic = wounds[0]
            else:
                topic = "the field"

        # Get relationship data for Bridge voice
        relationships = self._get_topic_relationships(topic)

        # Build the 5 perspectives
        perspectives = {
            "grey": {
                "name": "Grey (Shadow)",
                "question": "What's being avoided?",
                "voice": self._grey_voice(topic, wounds)
            },
            "witness": {
                "name": "Witness (Mirror)",
                "question": "What IS?",
                "voice": self._witness_voice(topic, crystal_count, avg_coherence, dominant_emotions)
            },
            "chaos": {
                "name": "Chaos (Trickster)",
                "question": "What if you're wrong?",
                "voice": self._chaos_voice(topic)
            },
            "bridge": {
                "name": "Bridge (Connector)",
                "question": "What links these?",
                "voice": self._bridge_voice(topic, relationships, wounds)
            },
            "ground": {
                "name": "Ground (Anchor)",
                "question": "What's body-true?",
                "voice": self._ground_voice(topic)
            }
        }

        return {
            "topic": topic,
            "perspectives": perspectives,
            "context": {
                "crystal_count": crystal_count,
                "avg_coherence": avg_coherence,
                "dominant_emotions": dominant_emotions[:3]
            }
        }

    def _grey_voice(self, topic: str, wounds: List[str]) -> str:
        """Shadow perspective - what's being avoided."""
        other_wounds = [w for w in wounds if w.lower() != topic.lower()][:2]

        if other_wounds:
            return f"You've been circling {topic} for weeks. But what if the real avoidance is {other_wounds[0]}? What happens if you stop performing awareness of {topic}?"
        else:
            return f"You keep naming {topic} as the pattern. What happens if you stop? What's underneath the naming?"

    def _witness_voice(self, topic: str, crystal_count: int, avg_coherence: float, emotions: List[str]) -> str:
        """Mirror perspective - what IS, factually."""
        emotion_str = ", ".join(emotions[:3]) if emotions else "unnamed feelings"

        return f"The pattern '{topic}' appears {crystal_count} times across your crystals. Average coherence when it surfaces: {avg_coherence:.2f}. It co-occurs with {emotion_str}. These are the facts."

    def _chaos_voice(self, topic: str) -> str:
        """Trickster perspective - disrupting certainty."""
        return f"What if {topic} isn't actually a problem? What if your certainty about it being a problem is the real loop? What if you're using 'working on {topic}' as a way to avoid being present?"

    def _bridge_voice(self, topic: str, relationships: List[str], wounds: List[str]) -> str:
        """Connector perspective - finding threads."""
        if relationships:
            rel_str = ", ".join(relationships[:3])
            return f"{topic.title()} connects to {rel_str}. The thread between them might be boundaries - where you end and others begin."
        elif wounds:
            wound_str = ", ".join(wounds[:3])
            return f"{topic.title()} threads through {wound_str}. These aren't separate wounds - they're facets of the same shape."
        else:
            return f"What connects {topic} to everything else in your field? Look for the shape underneath the content."

    def _ground_voice(self, topic: str) -> str:
        """Anchor perspective - body truth."""
        return f"Where do you feel {topic} in your body right now? Not the story about it. Not the analysis. Just the sensation. That's where the truth lives."

    def _get_crystal_count(self) -> int:
        """Get total crystal count for user."""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM crystals WHERE user_id = ?", (self.user_id,))
        count = c.fetchone()[0]
        conn.close()
        return count

    def _get_avg_coherence(self) -> float:
        """Get average Zλ score."""
        conn = self._get_conn()
        c = conn.cursor()
        c.execute("SELECT AVG(zl_score) FROM crystals WHERE user_id = ? AND zl_score IS NOT NULL", (self.user_id,))
        avg = c.fetchone()[0] or 0.5
        conn.close()
        return round(avg, 2)

    def _get_topic_relationships(self, topic: str) -> List[str]:
        """Get relationships that co-occur with a topic."""
        names = ['ricardo', 'juliana', 'mom', 'rose', 'michelle', 'renan', 'nahima', 'guilherme', 'mutsa']
        conn = self._get_conn()
        c = conn.cursor()

        results = []
        topic_lower = topic.lower()

        for name in names:
            c.execute("""
                SELECT COUNT(*) FROM crystals
                WHERE user_id = ?
                  AND LOWER(content) LIKE ?
                  AND (LOWER(content) LIKE ? OR LOWER(core_wound) LIKE ?)
            """, (self.user_id, f'%{name}%', f'%{topic_lower}%', f'%{topic_lower}%'))

            count = c.fetchone()[0]
            if count > 0:
                results.append((name.title(), count))

        conn.close()

        # Sort by count, return names only
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:5]]
