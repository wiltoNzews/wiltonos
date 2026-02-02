#!/usr/bin/env python3
"""
Field Intake Protocol
=====================
Orchestrator: gap detection, question generation, answer processing, field events.

The system identifies its own knowledge gaps, asks the user questions,
and stores answers as crystals + structured events.

Tables: field_events, intake_sessions, intake_questions

Usage:
    from field_intake import FieldIntake
    fi = FieldIntake(db_path=db_path, field_vocab=vocab, mesh_memory=mm, writer=writer)
    trigger = fi.should_trigger(user_id)
    if trigger:
        data = fi.start_intake(user_id)
        for q in data['questions']:
            result = fi.process_answer(data['session_id'], q['id'], answer, user_id)
        fi.complete_intake(data['session_id'])
"""

import re
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from entity_index import EntityIndex
from intake_questions import IntakeQuestionGenerator

_DEFAULT_DB = str(Path.home() / "wiltonos" / "data" / "crystals_unified.db")

# Event detection patterns
EVENT_PATTERNS = {
    "relationship": [
        re.compile(p, re.IGNORECASE) for p in [
            r"broke up", r"got (?:married|engaged|together)",
            r"started (?:seeing|dating)", r"we'?re done",
            r"left me", r"moved in(?: together)?",
            r"divorce", r"separated",
        ]
    ],
    "move": [
        re.compile(p, re.IGNORECASE) for p in [
            r"moved to", r"left (?:the country|home|town)",
            r"came back", r"relocated", r"new (?:apartment|house|place)",
        ]
    ],
    "health": [
        re.compile(p, re.IGNORECASE) for p in [
            r"surgery", r"diagnosed", r"recovering",
            r"hospital", r"sick", r"treatment",
        ]
    ],
    "work": [
        re.compile(p, re.IGNORECASE) for p in [
            r"new job", r"got fired", r"quit(?:ting)?",
            r"started working", r"laid off", r"promoted",
            r"retired",
        ]
    ],
    "loss": [
        re.compile(p, re.IGNORECASE) for p in [
            r"died", r"passed (?:away)?", r"funeral",
            r"lost (?:my|the)", r"grief",
        ]
    ],
    "project": [
        re.compile(p, re.IGNORECASE) for p in [
            r"launched", r"shipped", r"released",
            r"finished (?:building|making)",
        ]
    ],
    "milestone": [
        re.compile(p, re.IGNORECASE) for p in [
            r"graduated", r"birthday", r"anniversary",
            r"turned \d+",
        ]
    ],
}

# Acute distress markers (pause intake if detected)
DISTRESS_MARKERS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bhelp me\b", r"\bcan'?t (?:breathe|think|stop)\b",
        r"\bi'?m scared\b", r"\bbreaking\b", r"\bfalling apart\b",
        r"\bcrying\b", r"\bhurting\b", r"\bdon'?t know what to do\b",
        r"\bwant to (?:die|end|disappear)\b", r"\bsuicid",
    ]
]


@dataclass
class GapReport:
    """Result of gap detection for a user."""
    user_id: str
    session_type: str                          # first_contact, active, returning, gap_bridge
    crystal_gap_hours: float
    last_crystal_date: Optional[str]
    crystal_count: int
    is_new_user: bool
    stale_entities: List[Dict] = field(default_factory=list)
    unresolved_wounds: List[Dict] = field(default_factory=list)
    coherence_delta: Optional[float] = None
    open_threads: List[Dict] = field(default_factory=list)
    unresolved_events: List[Dict] = field(default_factory=list)


class GapDetector:
    """Detects knowledge gaps for a user based on crystal recency, entities, wounds, events."""

    def __init__(self, db_path: str, mesh_memory=None):
        self.db_path = db_path
        self.mesh_memory = mesh_memory
        self.entity_index = EntityIndex(db_path)

    def detect_gaps(self, user_id: str) -> GapReport:
        """Run full gap detection and return a GapReport."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Step 1: Crystal recency → session_type
        crystal_info = self._crystal_recency(conn)
        count = crystal_info['count']
        gap_hours = crystal_info['gap_hours']
        last_date = crystal_info['last_date']

        if count == 0:
            session_type = "first_contact"
        elif gap_hours < 24:
            session_type = "active"
        elif gap_hours < 168:  # 7 days
            session_type = "returning"
        else:
            session_type = "gap_bridge"

        is_new = count == 0

        # Step 2: Entity recency
        stale_entities = []
        if not is_new:
            stale_days = 14 if session_type in ("returning", "gap_bridge") else 30
            stale_entities = self.entity_index.get_stale_entities(
                user_id, days=stale_days, limit=10
            )

        # Step 3: Unresolved wound arcs (via MeshMemory)
        unresolved_wounds = []
        if self.mesh_memory and not is_new:
            try:
                recent_wounds = self.mesh_memory.query_recent_wounds(user_id, days=30)
                for wound_name, count in recent_wounds.items():
                    if count >= 3:  # Active for 3+ mesh runs
                        unresolved_wounds.append({
                            'wound': wound_name,
                            'last_active': None,
                            'intensity': min(count / 10.0, 1.0),
                        })
            except Exception:
                pass

        # Step 4: Coherence delta
        coherence_delta = None
        if self.mesh_memory and not is_new:
            try:
                avg = self.mesh_memory.average_coherence(user_id, days=7)
                baseline = self.mesh_memory.average_coherence(user_id, days=30)
                if avg is not None and baseline is not None:
                    coherence_delta = avg - baseline
            except Exception:
                pass

        # Step 5: Open threads (unresolved field_events + last session state)
        open_threads = []
        unresolved_events = []
        try:
            rows = conn.execute(
                """SELECT id, event_type, summary, reported_at
                   FROM field_events
                   WHERE user_id = ? AND is_resolved = 0
                   ORDER BY reported_at DESC
                   LIMIT 5""",
                (user_id,),
            ).fetchall()
            for r in rows:
                event = {
                    'event_id': r['id'],
                    'summary': r['summary'],
                    'reported_at': r['reported_at'],
                    'topic': r['summary'],
                    'source': 'field_event',
                }
                unresolved_events.append(event)
                open_threads.append(event)
        except Exception:
            pass  # Table may not exist yet

        # Last session's emotional thread
        try:
            session_row = conn.execute(
                """SELECT state FROM sessions
                   WHERE user_id = ? AND is_active = 0
                   ORDER BY updated_at DESC
                   LIMIT 1""",
                (user_id,),
            ).fetchone()
            if session_row and session_row['state']:
                state = json.loads(session_row['state'])
                if state.get('emotional_thread'):
                    open_threads.append({
                        'topic': state['emotional_thread'],
                        'source': 'session',
                        'last_mentioned': state.get('updated_at'),
                    })
                if state.get('key_insights'):
                    for insight in state['key_insights'][-2:]:
                        open_threads.append({
                            'topic': insight,
                            'source': 'session_insight',
                        })
        except Exception:
            pass

        conn.close()

        return GapReport(
            user_id=user_id,
            session_type=session_type,
            crystal_gap_hours=gap_hours,
            last_crystal_date=last_date,
            crystal_count=count,
            is_new_user=is_new,
            stale_entities=stale_entities,
            unresolved_wounds=unresolved_wounds,
            coherence_delta=coherence_delta,
            open_threads=open_threads,
            unresolved_events=unresolved_events,
        )

    def _crystal_recency(self, conn) -> Dict:
        """Get crystal count, last date, and gap hours."""
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt, MAX(created_at) as last_date FROM auto_insights"
            ).fetchone()
            count = row['cnt'] or 0
            last_date = row['last_date']

            if count == 0 or not last_date:
                return {'count': 0, 'gap_hours': float('inf'), 'last_date': None}

            try:
                last_dt = datetime.fromisoformat(last_date)
            except ValueError:
                # Try parsing as timestamp
                try:
                    last_dt = datetime.fromtimestamp(float(last_date))
                except (ValueError, TypeError):
                    return {'count': count, 'gap_hours': 0, 'last_date': last_date}

            gap = (datetime.now() - last_dt).total_seconds() / 3600
            return {'count': count, 'gap_hours': gap, 'last_date': last_date}

        except Exception:
            return {'count': 0, 'gap_hours': float('inf'), 'last_date': None}


class AnswerProcessor:
    """Processes intake answers: extracts entities, emotions, wounds, events, stores crystal."""

    def __init__(self, db_path: str, field_vocab=None, entity_index: EntityIndex = None, writer=None):
        self.db_path = db_path
        self.field_vocab = field_vocab
        self.entity_index = entity_index or EntityIndex(db_path)
        self.writer = writer

    def process(self, answer_text: str, question: Dict, user_id: str, intake_session_id: int) -> Dict:
        """
        Process a single intake answer through the full pipeline.

        Returns {crystal_stored, entities, emotions, wounds, events, distress_detected}
        """
        category = question.get('category', 'unknown')
        question_text = question.get('question', '')
        question_id = question.get('id')

        # Check for acute distress
        distress = self._check_distress(answer_text)

        # Stage 1: Wound/emotion extraction via FieldVocabulary
        wounds = []
        emotions = []
        if self.field_vocab:
            try:
                wounds = self.field_vocab.scan_wounds(answer_text)      # [(name, confidence)]
                emotions = self.field_vocab.scan_emotions(answer_text)  # [(name, valence)]
            except Exception:
                pass

        # Stage 2: Entity extraction via EntityIndex
        entities = self.entity_index.extract_entities(answer_text, user_id)

        # Stage 3: Event detection (keyword patterns)
        events = self._detect_events(answer_text, entities, user_id)

        # Stage 4: Crystal storage via CrystalWriter
        crystal_stored = False
        if self.writer:
            try:
                top_emotion = emotions[0][0] if emotions else None
                content = f"[Intake:{category}] Q: {question_text}\nA: {answer_text}"
                crystal_stored = self.writer.store_insight(
                    content=content,
                    source='field_intake',
                    emotion=top_emotion,
                    topology=category,
                )
            except Exception:
                pass

        # Stage 5: Entity index update
        wound_str = ",".join(w[0] for w in wounds[:3]) if wounds else None
        emotion_str = ",".join(e[0] for e in emotions[:3]) if emotions else None
        for entity in entities:
            try:
                self.entity_index.record_mention(
                    user_id=user_id,
                    entity_name=entity['name'],
                    entity_type=entity['type'],
                    context=entity.get('context'),
                    wound_active=wound_str,
                    emotion_active=emotion_str,
                )
            except Exception:
                pass

        # Stage 6: Update intake_questions row
        event_ids = [e.get('event_id') for e in events if e.get('event_id')]
        if question_id:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute(
                    """UPDATE intake_questions
                       SET answer_text = ?,
                           answered_at = ?,
                           entities_extracted = ?,
                           emotions_detected = ?,
                           wounds_detected = ?,
                           events_created = ?
                       WHERE id = ?""",
                    (
                        answer_text,
                        datetime.now().isoformat(),
                        json.dumps([e['name'] for e in entities]),
                        json.dumps([e[0] for e in emotions]),
                        json.dumps([w[0] for w in wounds]),
                        json.dumps(event_ids) if event_ids else None,
                        question_id,
                    ),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

        return {
            'crystal_stored': crystal_stored,
            'entities': [e['name'] for e in entities],
            'emotions': [e[0] for e in emotions],
            'wounds': [w[0] for w in wounds],
            'events': events,
            'distress_detected': distress,
        }

    def _detect_events(self, text: str, entities: List[Dict], user_id: str) -> List[Dict]:
        """Detect life events from answer text using keyword patterns."""
        detected = []

        for event_type, patterns in EVENT_PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    # Build event
                    entity_names = [e['name'] for e in entities]
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    summary = text[start:end].strip()

                    # Store field_event
                    event_id = self._store_field_event(
                        user_id=user_id,
                        event_type=event_type,
                        summary=summary,
                        entities=entity_names,
                    )

                    detected.append({
                        'event_type': event_type,
                        'summary': summary,
                        'entities': entity_names,
                        'event_id': event_id,
                    })
                    break  # One event per type per answer

        return detected

    def _store_field_event(self, user_id: str, event_type: str, summary: str,
                           entities: List[str] = None) -> Optional[int]:
        """Insert a field_event and return its ID."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.execute(
                """INSERT INTO field_events
                   (user_id, event_type, summary, entities_involved,
                    emotional_valence, source, reported_at)
                   VALUES (?, ?, ?, ?, ?, 'intake', ?)""",
                (
                    user_id,
                    event_type,
                    summary,
                    json.dumps(entities) if entities else None,
                    'neutral',  # Will be refined by wound/emotion data later
                    datetime.now().isoformat(),
                ),
            )
            event_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return event_id
        except Exception:
            return None

    def _check_distress(self, text: str) -> bool:
        """Check for acute distress markers in text."""
        for pattern in DISTRESS_MARKERS:
            if pattern.search(text):
                return True
        return False


class FieldIntake:
    """
    Main orchestrator for field intake protocol.
    Coordinates gap detection, question generation, answer processing.
    """

    def __init__(self, db_path: str = None, field_vocab=None, mesh_memory=None, writer=None):
        self.db_path = db_path or _DEFAULT_DB
        self.entity_index = EntityIndex(self.db_path)
        self.gap_detector = GapDetector(self.db_path, mesh_memory)
        self.question_gen = IntakeQuestionGenerator(self.db_path)
        self.processor = AnswerProcessor(self.db_path, field_vocab, self.entity_index, writer)
        self._ensure_tables()

    def _ensure_tables(self):
        """Create field_events, intake_sessions, intake_questions tables."""
        conn = sqlite3.connect(self.db_path)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS field_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                entities_involved TEXT,
                emotional_valence TEXT,
                wound_context TEXT,
                reported_at TEXT DEFAULT CURRENT_TIMESTAMP,
                occurred_at TEXT,
                source TEXT DEFAULT 'intake',
                crystal_id INTEGER,
                is_resolved INTEGER DEFAULT 0,
                resolution_notes TEXT,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_field_events_user_reported
            ON field_events(user_id, reported_at)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_field_events_user_type
            ON field_events(user_id, event_type)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS intake_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_type TEXT NOT NULL,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                questions_asked INTEGER DEFAULT 0,
                questions_answered INTEGER DEFAULT 0,
                questions_skipped INTEGER DEFAULT 0,
                gap_days REAL,
                metadata TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS intake_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intake_session_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                question_category TEXT NOT NULL,
                question_text TEXT NOT NULL,
                answer_text TEXT,
                was_skipped INTEGER DEFAULT 0,
                entities_extracted TEXT,
                emotions_detected TEXT,
                wounds_detected TEXT,
                events_created TEXT,
                asked_at TEXT DEFAULT CURRENT_TIMESTAMP,
                answered_at TEXT,
                FOREIGN KEY (intake_session_id) REFERENCES intake_sessions(id)
            )
        """)

        conn.commit()
        conn.close()

    def should_trigger(self, user_id: str) -> Optional[Dict]:
        """
        Fast check: should we run intake?

        Returns {reason, session_type, gap_days} or None.

        Rules:
          New user (no intake history, no entity mentions) → first_contact
          0 crystals globally → first_contact
          crystal gap > 3 days AND last intake > 7 days → trigger
          Never trigger if last intake < 24 hours
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Check if this is a brand new user (no intake sessions and no entity mentions)
        has_user_data = False
        try:
            intake_count = conn.execute(
                "SELECT COUNT(*) FROM intake_sessions WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            entity_count = conn.execute(
                "SELECT COUNT(*) FROM entity_mentions WHERE user_id = ?",
                (user_id,),
            ).fetchone()[0]
            has_user_data = (intake_count > 0) or (entity_count > 0)
        except Exception:
            pass

        # Check global crystal count
        try:
            row = conn.execute(
                "SELECT COUNT(*) as cnt, MAX(created_at) as last_date FROM auto_insights"
            ).fetchone()
            crystal_count = row['cnt'] or 0
        except Exception:
            crystal_count = 0

        if crystal_count == 0 or not has_user_data:
            # Check if we already did an intake recently for this user
            try:
                recent_intake = conn.execute(
                    """SELECT started_at FROM intake_sessions
                       WHERE user_id = ? ORDER BY started_at DESC LIMIT 1""",
                    (user_id,),
                ).fetchone()
                if recent_intake and recent_intake['started_at']:
                    last_intake = datetime.fromisoformat(recent_intake['started_at'])
                    if (datetime.now() - last_intake).total_seconds() / 3600 < 24:
                        conn.close()
                        return None  # Already did intake recently
            except Exception:
                pass

            if crystal_count == 0:
                conn.close()
                return {
                    'reason': 'New system — first contact',
                    'session_type': 'first_contact',
                    'gap_days': 0,
                }
            elif not has_user_data:
                conn.close()
                return {
                    'reason': 'New user — first contact',
                    'session_type': 'first_contact',
                    'gap_days': 0,
                }

        # Check last intake recency
        intake_row = None
        try:
            intake_row = conn.execute(
                """SELECT started_at FROM intake_sessions
                   WHERE user_id = ?
                   ORDER BY started_at DESC LIMIT 1""",
                (user_id,),
            ).fetchone()

            if intake_row and intake_row['started_at']:
                try:
                    last_intake = datetime.fromisoformat(intake_row['started_at'])
                    hours_since_intake = (datetime.now() - last_intake).total_seconds() / 3600
                    if hours_since_intake < 24:
                        conn.close()
                        return None  # Too recent
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass  # Table may not exist yet

        # Check crystal gap
        try:
            last_crystal = conn.execute(
                "SELECT MAX(created_at) as last_date FROM auto_insights"
            ).fetchone()
            if last_crystal and last_crystal['last_date']:
                try:
                    last_dt = datetime.fromisoformat(last_crystal['last_date'])
                    gap_hours = (datetime.now() - last_dt).total_seconds() / 3600
                    gap_days = gap_hours / 24
                except (ValueError, TypeError):
                    gap_hours = 0
                    gap_days = 0
            else:
                gap_hours = 0
                gap_days = 0
        except Exception:
            gap_hours = 0
            gap_days = 0

        conn.close()

        # Crystal gap > 3 days
        if gap_days > 3:
            # Check last intake was > 7 days ago (or never)
            last_intake_ok = True
            if intake_row and intake_row['started_at']:
                try:
                    last_intake = datetime.fromisoformat(intake_row['started_at'])
                    days_since_intake = (datetime.now() - last_intake).total_seconds() / 86400
                    last_intake_ok = days_since_intake > 7
                except (ValueError, TypeError):
                    pass

            if last_intake_ok:
                if gap_days > 7:
                    session_type = "gap_bridge"
                    reason = f"{int(gap_days)} days since last activity"
                else:
                    session_type = "returning"
                    reason = f"{int(gap_days)} days since last activity"

                return {
                    'reason': reason,
                    'session_type': session_type,
                    'gap_days': gap_days,
                }

        return None

    def start_intake(self, user_id: str) -> Dict:
        """
        Start an intake session.

        1. Run gap detection
        2. Generate questions from gaps
        3. Create intake_sessions row
        4. Create intake_questions rows

        Returns {session_id, questions: [...], session_type, gap_summary}
        """
        # 1. Gap detection
        gap_report = self.gap_detector.detect_gaps(user_id)

        # 2. Generate questions
        raw_questions = self.question_gen.generate(gap_report)

        # 3. Create session
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            """INSERT INTO intake_sessions
               (user_id, session_type, started_at, questions_asked, gap_days)
               VALUES (?, ?, ?, ?, ?)""",
            (
                user_id,
                gap_report.session_type,
                datetime.now().isoformat(),
                len(raw_questions),
                gap_report.crystal_gap_hours / 24,
            ),
        )
        session_id = cursor.lastrowid

        # 4. Create question rows
        questions_with_ids = []
        for q in raw_questions:
            cursor = conn.execute(
                """INSERT INTO intake_questions
                   (intake_session_id, user_id, question_category, question_text, asked_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, user_id, q['category'], q['question'], datetime.now().isoformat()),
            )
            q_with_id = dict(q)
            q_with_id['id'] = cursor.lastrowid
            questions_with_ids.append(q_with_id)

        conn.commit()
        conn.close()

        return {
            'session_id': session_id,
            'questions': questions_with_ids,
            'session_type': gap_report.session_type,
            'gap_days': gap_report.crystal_gap_hours / 24,
            'gap_summary': {
                'crystal_count': gap_report.crystal_count,
                'stale_entities': len(gap_report.stale_entities),
                'unresolved_wounds': len(gap_report.unresolved_wounds),
                'open_threads': len(gap_report.open_threads),
            },
        }

    def process_answer(self, intake_session_id: int, question_id: int,
                       answer_text: str, user_id: str) -> Dict:
        """Process a single intake answer. Delegates to AnswerProcessor."""
        # Load the question
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM intake_questions WHERE id = ?",
            (question_id,),
        ).fetchone()
        conn.close()

        question = dict(row) if row else {'id': question_id, 'category': 'unknown', 'question': ''}
        question['question'] = question.get('question_text', question.get('question', ''))

        return self.processor.process(answer_text, question, user_id, intake_session_id)

    def skip_question(self, question_id: int):
        """Mark a question as skipped."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE intake_questions SET was_skipped = 1 WHERE id = ?",
                (question_id,),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def complete_intake(self, intake_session_id: int):
        """Mark an intake session as complete. Update question counts."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row

            # Count answered and skipped
            answered = conn.execute(
                "SELECT COUNT(*) FROM intake_questions WHERE intake_session_id = ? AND answer_text IS NOT NULL",
                (intake_session_id,),
            ).fetchone()[0]

            skipped = conn.execute(
                "SELECT COUNT(*) FROM intake_questions WHERE intake_session_id = ? AND was_skipped = 1",
                (intake_session_id,),
            ).fetchone()[0]

            conn.execute(
                """UPDATE intake_sessions
                   SET completed_at = ?,
                       questions_answered = ?,
                       questions_skipped = ?
                   WHERE id = ?""",
                (datetime.now().isoformat(), answered, skipped, intake_session_id),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
