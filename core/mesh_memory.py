#!/usr/bin/env python3
"""
Mesh Memory — Board persistence and cross-session querying.
============================================================
Stores mesh run outcomes in crystals_unified.db after each AgentMesh.run().
Enables cross-session loop detection (Torus) and glyph trajectory queries.

Each row in mesh_runs IS a glyph ledger entry:
- glyph, zl, mode, attractor, breath_phase = the state snapshot
- wounds_active, emotions_active = what the field was carrying
- halted, halt_reason = Ground's intervention
- bridge_connections, torus_loops = agent observations
- active_posts = mesh participation level

Usage:
    from mesh_memory import MeshMemory
    memory = MeshMemory(db_path)
    memory.persist(board, session_id)
    past = memory.query_recent_wounds("wilton", days=7)
    trajectory = memory.glyph_trajectory("wilton", limit=50)
"""

import json
import sqlite3
import time
from typing import Optional, List, Dict, Any


class MeshMemory:
    """
    Persist mesh run outcomes and query cross-session patterns.

    Table: mesh_runs in crystals_unified.db
    Self-initializing (CREATE TABLE IF NOT EXISTS in __init__).
    """

    def __init__(self, db_path: str):
        self.db_path = str(db_path)
        self._ensure_table()

    def _ensure_table(self):
        """Create mesh_runs table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT,
                query TEXT NOT NULL,

                -- Glyph ledger (every row IS a ledger entry)
                glyph TEXT,
                zl REAL,
                mode TEXT,
                attractor TEXT,
                breath_phase REAL,

                -- Agent observations (JSON lists)
                wounds_active TEXT,
                emotions_active TEXT,
                bridge_connections TEXT,
                torus_loops TEXT,
                pattern_confidence REAL,

                -- Board outcome
                halted INTEGER DEFAULT 0,
                halt_reason TEXT,
                active_posts INTEGER,
                rounds INTEGER,

                -- Timing
                timestamp REAL NOT NULL,
                duration_ms REAL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mesh_runs_user_ts
            ON mesh_runs(user_id, timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mesh_runs_wounds
            ON mesh_runs(wounds_active)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mesh_runs_glyph
            ON mesh_runs(glyph)
        """)
        conn.commit()
        conn.close()

    # ── Persist ──────────────────────────────────────────────────

    def persist(
        self,
        board: Any,
        session_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ):
        """
        Extract signals from a completed Board and store as one mesh_runs row.
        Called at the end of AgentMesh.run().
        """
        seed = board.seed
        state = seed.state

        # Extract glyph ledger fields from CoherenceState
        glyph = getattr(state, "glyph", None)
        glyph_val = glyph.value if glyph else None
        zl = getattr(state, "zeta_lambda", None)
        mode = getattr(state, "mode", None)
        mode_val = mode.value if mode else None
        attractor = getattr(state, "attractor", None)
        breath_phase = getattr(state, "breath_phase", None)

        # Extract agent observations from board posts
        wounds_active = []
        emotions_active = []
        bridge_connections = []
        torus_loops = []

        for post in board.posts:
            if post.agent == "pattern" and post.kind.value == "pattern":
                content = post.content
                if "Wounds:" in content:
                    wound_section = content.split("Wounds:")[1].split("|")[0]
                    for part in wound_section.split(","):
                        part = part.strip()
                        if "(" in part:
                            wound_name = part.split("(")[0].strip()
                            if wound_name:
                                wounds_active.append(wound_name)
                if "Emotions:" in content:
                    emo_section = content.split("Emotions:")[1].split("|")[0]
                    for part in emo_section.split(","):
                        emo = part.strip()
                        if emo:
                            emotions_active.append(emo)

            elif post.agent == "bridge" and post.kind.value == "bridge":
                bridge_connections = [
                    c.strip() for c in post.content.split(" | ") if c.strip()
                ]

            elif post.agent == "torus" and post.kind.value == "torus":
                torus_loops = [
                    l.strip() for l in post.content.split(" | ") if l.strip()
                ]

        # Fallback to pattern_match object if posts didn't yield wound names
        pm = seed.pattern_match
        if pm is not None:
            pattern_confidence = getattr(pm, "confidence", 0.0)
            if not wounds_active and hasattr(pm, "wounds"):
                wounds_active = [w for w, _ in pm.wounds[:5]]
            if not emotions_active and hasattr(pm, "emotions"):
                emotions_active = [e for e, _ in pm.emotions[:5]]
        else:
            pattern_confidence = 0.0

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO mesh_runs
               (user_id, session_id, query, glyph, zl, mode, attractor,
                breath_phase, wounds_active, emotions_active,
                bridge_connections, torus_loops, pattern_confidence,
                halted, halt_reason, active_posts, rounds,
                timestamp, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                seed.user_id,
                session_id,
                seed.query[:1000],
                glyph_val,
                zl,
                mode_val,
                attractor,
                breath_phase,
                json.dumps(wounds_active),
                json.dumps(emotions_active),
                json.dumps(bridge_connections),
                json.dumps(torus_loops),
                pattern_confidence,
                1 if board.halted else 0,
                board.halt_reason,
                board.active_post_count(),
                board.current_round,
                time.time(),
                duration_ms,
            ),
        )
        conn.commit()
        conn.close()

    # ── Cross-session queries (for Torus) ────────────────────────

    def query_recent_wounds(
        self,
        user_id: str,
        days: int = 7,
        limit: int = 20,
    ) -> Dict[str, int]:
        """
        Get wound frequency from recent mesh runs.
        Returns {wound_name: count}.
        Used by Torus for cross-session recurrence detection.
        """
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT wounds_active FROM mesh_runs
               WHERE user_id = ? AND timestamp > ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (user_id, cutoff, limit),
        ).fetchall()
        conn.close()

        from collections import Counter
        wound_counts: Counter = Counter()
        for (wounds_json,) in rows:
            if wounds_json:
                try:
                    for w in json.loads(wounds_json):
                        wound_counts[w] += 1
                except (json.JSONDecodeError, TypeError):
                    pass
        return dict(wound_counts)

    def query_wound_recurrence(
        self,
        user_id: str,
        wound_name: str,
        days: int = 30,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Find past mesh runs where a specific wound was active.
        Returns list of dicts with: timestamp, query, glyph, zl, wounds_active.
        """
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp, query, glyph, zl, wounds_active,
                      emotions_active, torus_loops, halted
               FROM mesh_runs
               WHERE user_id = ? AND timestamp > ?
                     AND wounds_active LIKE ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (user_id, cutoff, f'%"{wound_name}"%', limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def query_emotion_recurrence(
        self,
        user_id: str,
        emotion_name: str,
        days: int = 30,
        limit: int = 50,
    ) -> List[Dict]:
        """Find past mesh runs where a specific emotion was active."""
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT timestamp, query, glyph, zl, wounds_active,
                      emotions_active, halted
               FROM mesh_runs
               WHERE user_id = ? AND timestamp > ?
                     AND emotions_active LIKE ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (user_id, cutoff, f'%"{emotion_name}"%', limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    # ── Glyph ledger queries ────────────────────────────────────

    def glyph_trajectory(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Get the last N glyph states (the session glyph map).
        Returns chronological list of ledger entries.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT glyph, zl, mode, attractor, breath_phase,
                      halted, active_posts, timestamp, query
               FROM mesh_runs
               WHERE user_id = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in reversed(rows)]  # chronological

    def last_halt(self, user_id: str) -> Optional[Dict]:
        """When did Ground last halt?"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """SELECT glyph, zl, mode, halt_reason, timestamp, query
               FROM mesh_runs
               WHERE user_id = ? AND halted = 1
               ORDER BY timestamp DESC
               LIMIT 1""",
            (user_id,),
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def glyph_distribution(
        self, user_id: str, days: int = 30
    ) -> Dict[str, int]:
        """Count of each glyph state in the last N days."""
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT glyph, COUNT(*) as cnt FROM mesh_runs
               WHERE user_id = ? AND timestamp > ?
               GROUP BY glyph
               ORDER BY cnt DESC""",
            (user_id, cutoff),
        ).fetchall()
        conn.close()
        return {g: c for g, c in rows if g}

    def average_coherence(
        self, user_id: str, days: int = 7
    ) -> Optional[float]:
        """Average Zl over last N days."""
        cutoff = time.time() - (days * 86400)
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            """SELECT AVG(zl) FROM mesh_runs
               WHERE user_id = ? AND timestamp > ? AND zl IS NOT NULL""",
            (user_id, cutoff),
        ).fetchone()
        conn.close()
        return row[0] if row and row[0] is not None else None
