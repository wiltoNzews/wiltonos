"""
Session Continuity Module
=========================
Bridge conversations across platforms: OpenWebUI <-> Terminal <-> Phone
One field, many access points.

Usage:
    from session import SessionManager
    sm = SessionManager()

    # Save state
    sm.save_session(user='wilton', platform='openwebui',
                    key_insights=['...'], emotional_thread='grief -> clarity')

    # Load on any platform
    session = sm.get_latest_session(user='wilton')
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any


class SessionManager:
    """Manage session continuity across platforms."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or "/home/zews/wiltonos/data/crystals_unified.db")
        self._ensure_tables()

    def _ensure_tables(self):
        """Ensure session tables exist."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                platform TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                state TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS session_turns (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                glyphs TEXT,
                emotion TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)
        """)

        c.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_active ON sessions(is_active)
        """)

        conn.commit()
        conn.close()

    def create_session(
        self,
        user_id: str = 'wilton',
        platform: str = 'openwebui',
        initial_state: Dict = None
    ) -> str:
        """
        Create a new session.

        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())[:8]

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            INSERT INTO sessions (id, user_id, platform, state)
            VALUES (?, ?, ?, ?)
        """, (
            session_id,
            user_id,
            platform,
            json.dumps(initial_state) if initial_state else None
        ))

        conn.commit()
        conn.close()

        return session_id

    def save_session(
        self,
        session_id: str = None,
        user_id: str = 'wilton',
        platform: str = 'openwebui',
        key_insights: List[str] = None,
        emotional_thread: str = None,
        last_glyph: str = None,
        custom_state: Dict = None
    ) -> str:
        """
        Save or update session state.

        If session_id is None, creates new session.
        Returns session_id.
        """
        if session_id is None:
            session_id = self.create_session(user_id, platform)

        state = {
            'key_insights': key_insights or [],
            'emotional_thread': emotional_thread,
            'last_glyph': last_glyph,
            'updated_at': datetime.now().isoformat(),
            **(custom_state or {})
        }

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            UPDATE sessions
            SET state = ?, updated_at = CURRENT_TIMESTAMP, platform = ?
            WHERE id = ?
        """, (json.dumps(state), platform, session_id))

        conn.commit()
        conn.close()

        return session_id

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        glyphs: List[str] = None,
        emotion: str = None
    ):
        """Add a conversation turn to session history."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            INSERT INTO session_turns (session_id, role, content, glyphs, emotion)
            VALUES (?, ?, ?, ?, ?)
        """, (
            session_id,
            role,
            content[:5000],  # Limit content size
            json.dumps(glyphs) if glyphs else None,
            emotion
        ))

        # Update session timestamp
        c.execute("""
            UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (session_id,))

        conn.commit()
        conn.close()

    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get a specific session by ID."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT id, user_id, platform, created_at, updated_at, state, is_active
            FROM sessions WHERE id = ?
        """, (session_id,))

        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'id': row[0],
            'user_id': row[1],
            'platform': row[2],
            'created_at': row[3],
            'updated_at': row[4],
            'state': json.loads(row[5]) if row[5] else {},
            'is_active': bool(row[6])
        }

    def get_latest_session(
        self,
        user_id: str = 'wilton',
        max_age_hours: int = 24
    ) -> Optional[Dict]:
        """
        Get the most recent active session for a user.

        Args:
            user_id: User to get session for
            max_age_hours: Only return if session is newer than this

        Returns:
            Session dict or None
        """
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        cutoff = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()

        c.execute("""
            SELECT id, user_id, platform, created_at, updated_at, state, is_active
            FROM sessions
            WHERE user_id = ? AND is_active = 1 AND updated_at > ?
            ORDER BY updated_at DESC
            LIMIT 1
        """, (user_id, cutoff))

        row = c.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'id': row[0],
            'user_id': row[1],
            'platform': row[2],
            'created_at': row[3],
            'updated_at': row[4],
            'state': json.loads(row[5]) if row[5] else {},
            'is_active': bool(row[6])
        }

    def get_session_turns(
        self,
        session_id: str,
        limit: int = 20
    ) -> List[Dict]:
        """Get recent turns from a session."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT role, content, glyphs, emotion, created_at
            FROM session_turns
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (session_id, limit))

        turns = []
        for row in c.fetchall():
            turns.append({
                'role': row[0],
                'content': row[1],
                'glyphs': json.loads(row[2]) if row[2] else [],
                'emotion': row[3],
                'created_at': row[4]
            })

        conn.close()
        return list(reversed(turns))  # Chronological order

    def build_session_context(
        self,
        session_id: str = None,
        user_id: str = 'wilton'
    ) -> str:
        """
        Build context string from session for LLM prompts.

        Returns a formatted string with session continuity info.
        """
        session = None

        if session_id:
            session = self.get_session(session_id)
        else:
            session = self.get_latest_session(user_id)

        if not session:
            return ""

        state = session.get('state', {})
        turns = self.get_session_turns(session['id'], limit=5)

        context_parts = []

        # Session metadata
        context_parts.append(f"[Continuing session from {session['platform']}]")

        # Emotional thread
        if state.get('emotional_thread'):
            context_parts.append(f"Emotional thread: {state['emotional_thread']}")

        # Key insights
        if state.get('key_insights'):
            insights = state['key_insights'][-3:]  # Last 3
            context_parts.append("Recent insights:")
            for insight in insights:
                context_parts.append(f"  - {insight[:100]}")

        # Last glyph
        if state.get('last_glyph'):
            context_parts.append(f"Last glyph: {state['last_glyph']}")

        # Recent turns summary
        if turns:
            context_parts.append("\nRecent conversation:")
            for turn in turns[-3:]:
                role = "You" if turn['role'] == 'assistant' else "Wilton"
                content = turn['content'][:200]
                context_parts.append(f"  {role}: {content}...")

        return "\n".join(context_parts)

    def close_session(self, session_id: str):
        """Mark a session as inactive."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            UPDATE sessions SET is_active = 0 WHERE id = ?
        """, (session_id,))

        conn.commit()
        conn.close()

    def get_session_stats(self, user_id: str = 'wilton') -> Dict:
        """Get stats about user's sessions."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # Total sessions
        c.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ?", (user_id,))
        total = c.fetchone()[0]

        # Active sessions
        c.execute("""
            SELECT COUNT(*) FROM sessions
            WHERE user_id = ? AND is_active = 1
        """, (user_id,))
        active = c.fetchone()[0]

        # Sessions by platform
        c.execute("""
            SELECT platform, COUNT(*) FROM sessions
            WHERE user_id = ?
            GROUP BY platform
        """, (user_id,))
        by_platform = dict(c.fetchall())

        # Total turns
        c.execute("""
            SELECT COUNT(*) FROM session_turns st
            JOIN sessions s ON st.session_id = s.id
            WHERE s.user_id = ?
        """, (user_id,))
        total_turns = c.fetchone()[0]

        conn.close()

        return {
            'total_sessions': total,
            'active_sessions': active,
            'by_platform': by_platform,
            'total_turns': total_turns
        }


# CLI for testing
if __name__ == "__main__":
    sm = SessionManager()

    print("=== Session Manager ===")
    print(f"Database: {sm.db_path}")

    stats = sm.get_session_stats()
    print(f"\nSession Stats:")
    print(f"  Total sessions: {stats['total_sessions']}")
    print(f"  Active sessions: {stats['active_sessions']}")
    print(f"  By platform: {stats['by_platform']}")
    print(f"  Total turns: {stats['total_turns']}")

    # Test creating a session
    session_id = sm.create_session(
        user_id='wilton',
        platform='terminal',
        initial_state={'test': True}
    )
    print(f"\nCreated test session: {session_id}")

    # Save some state
    sm.save_session(
        session_id=session_id,
        key_insights=['The field recognizes itself', 'Breath routes the prompt'],
        emotional_thread='building -> clarity',
        last_glyph='ψ'
    )
    print("Saved session state")

    # Add a turn
    sm.add_turn(
        session_id=session_id,
        role='user',
        content='Testing session continuity',
        glyphs=['ψ'],
        emotion='curious'
    )
    print("Added turn")

    # Get context
    context = sm.build_session_context(session_id=session_id)
    print(f"\nSession context:\n{context}")

    # Clean up test
    sm.close_session(session_id)
    print("\nClosed test session")
