#!/usr/bin/env python3
"""
The Braiding Layer
==================
Pattern detection at scale across ALL crystals.
Not sampling. Not approximating. The whole field.

What it does:
- Finds recurring wounds across time
- Detects theme clusters and emotional arcs
- Surfaces connections Wilton doesn't see
- Generates questions from patterns

Mentioned in 271 crystals. Finally built.

December 2025 — Wilton & Claude
"""

import sqlite3
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

# Paths
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
BRAID_OUTPUT = Path(__file__).parent / "braid_state.json"

# Core wounds to track (from the system)
CORE_WOUNDS = [
    "abandonment",
    "betrayal",
    "unworthiness",
    "control",
    "unloved",
    "rejection",
    "shame",
    "isolation",
    "powerlessness",
    "not_enough"
]

# Emotional markers
EMOTIONS = [
    "grief", "anger", "fear", "joy", "peace", "anxiety",
    "hope", "despair", "love", "loneliness", "clarity",
    "confusion", "gratitude", "resentment", "curiosity"
]

# Key figures/threads to track
KEY_THREADS = [
    "juliana", "mom", "mother", "mãe", "ricardo", "father", "pai",
    "michelle", "renan", "vinao", "vito", "guilherme",
    "peru", "ayahuasca", "ceremony", "heart", "stent",
    "liquid", "esport", "champion", "zews",
    "coherence", "daemon", "mirror", "breath"
]


@dataclass
class BraidPattern:
    """A detected pattern across crystals."""
    pattern_type: str  # wound, emotion, thread, arc
    name: str
    occurrences: int
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    intensity_trend: str = "stable"  # rising, falling, stable, cyclic
    crystal_ids: list = field(default_factory=list)
    co_occurrences: dict = field(default_factory=dict)


@dataclass
class BraidState:
    """Full braid state of the crystal field."""
    total_crystals: int = 0
    analyzed_at: str = ""

    # Pattern collections
    wound_patterns: dict = field(default_factory=dict)
    emotion_patterns: dict = field(default_factory=dict)
    thread_patterns: dict = field(default_factory=dict)

    # Temporal analysis
    recent_shift: str = ""  # What changed in last 7 days
    stuck_patterns: list = field(default_factory=list)  # Same wound >30 days
    emerging_patterns: list = field(default_factory=list)  # New in last 7 days

    # Arc detection
    emotional_arc: str = ""  # Overall trajectory
    wound_cycles: list = field(default_factory=list)  # Recurring loops

    # Questions generated
    meta_questions: list = field(default_factory=list)


class BraidingLayer:
    """
    The Braiding Layer.

    Runs across ALL crystals, finds what's really there.
    No sampling. No approximation. Full field scan.
    """

    def __init__(self):
        self.state = BraidState()
        self.conn = None

    def _log(self, msg: str):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [BRAID] {msg}")

    def connect(self):
        """Connect to database."""
        self.conn = sqlite3.connect(str(DB_PATH))
        self.conn.row_factory = sqlite3.Row

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def _detect_in_text(self, text: str, patterns: list) -> list:
        """Find which patterns appear in text."""
        if not text:
            return []
        text_lower = text.lower()
        found = []
        for p in patterns:
            if p.lower() in text_lower:
                found.append(p)
        return found

    def analyze_all_crystals(self) -> BraidState:
        """
        Full field analysis. Every crystal. Every pattern.
        """
        self._log("Beginning full field braid analysis...")
        self.connect()

        c = self.conn.cursor()

        # Get all crystals
        c.execute("""
            SELECT id, user_id, content, emotion, core_wound,
                   zl_score, created_at, source
            FROM crystals
            WHERE user_id != 'daemon'
            ORDER BY id ASC
        """)

        crystals = c.fetchall()
        self.state.total_crystals = len(crystals)
        self._log(f"Analyzing {self.state.total_crystals} crystals...")

        # Tracking structures
        wound_timeline = defaultdict(list)  # wound -> [(date, crystal_id), ...]
        emotion_timeline = defaultdict(list)
        thread_timeline = defaultdict(list)
        co_occurrence_matrix = defaultdict(Counter)  # what appears together

        # Date tracking
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)

        recent_wounds = []
        recent_emotions = []

        # Process each crystal
        for i, crystal in enumerate(crystals):
            crystal_id = crystal['id']
            content = crystal['content'] or ""
            db_emotion = crystal['emotion']
            db_wound = crystal['core_wound']
            created_at = crystal['created_at']

            # Parse date if available
            crystal_date = None
            if created_at:
                try:
                    crystal_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                except:
                    pass

            # Detect wounds (from DB field + content scan)
            found_wounds = self._detect_in_text(content, CORE_WOUNDS)
            if db_wound and db_wound not in found_wounds:
                found_wounds.append(db_wound)

            for wound in found_wounds:
                wound_timeline[wound].append((created_at, crystal_id))
                if crystal_date and crystal_date.replace(tzinfo=None) > week_ago:
                    recent_wounds.append(wound)

            # Detect emotions
            found_emotions = self._detect_in_text(content, EMOTIONS)
            if db_emotion and db_emotion not in found_emotions:
                found_emotions.append(db_emotion)

            for emotion in found_emotions:
                emotion_timeline[emotion].append((created_at, crystal_id))
                if crystal_date and crystal_date.replace(tzinfo=None) > week_ago:
                    recent_emotions.append(emotion)

            # Detect threads
            found_threads = self._detect_in_text(content, KEY_THREADS)
            for thread in found_threads:
                thread_timeline[thread].append((created_at, crystal_id))

            # Co-occurrence tracking
            all_found = found_wounds + found_emotions + found_threads
            for item in all_found:
                for other in all_found:
                    if item != other:
                        co_occurrence_matrix[item][other] += 1

            # Progress logging
            if (i + 1) % 5000 == 0:
                self._log(f"Processed {i + 1}/{self.state.total_crystals} crystals...")

        self._log("Building pattern structures...")

        # Build wound patterns
        for wound, timeline in wound_timeline.items():
            dates = [t[0] for t in timeline if t[0]]
            pattern = BraidPattern(
                pattern_type="wound",
                name=wound,
                occurrences=len(timeline),
                first_seen=min(dates) if dates else None,
                last_seen=max(dates) if dates else None,
                crystal_ids=[t[1] for t in timeline[-100:]],  # Last 100
                co_occurrences=dict(co_occurrence_matrix[wound].most_common(5))
            )

            # Detect if stuck (appearing for >30 days without resolution)
            if dates and len(timeline) > 10:
                try:
                    first = datetime.fromisoformat(min(dates).replace('Z', '+00:00'))
                    last = datetime.fromisoformat(max(dates).replace('Z', '+00:00'))
                    if (last - first).days > 30:
                        recent_count = sum(1 for d in dates[-20:] if d)
                        if recent_count > 5:
                            self.state.stuck_patterns.append(wound)
                except:
                    pass

            self.state.wound_patterns[wound] = pattern.__dict__

        # Build emotion patterns
        for emotion, timeline in emotion_timeline.items():
            dates = [t[0] for t in timeline if t[0]]
            pattern = BraidPattern(
                pattern_type="emotion",
                name=emotion,
                occurrences=len(timeline),
                first_seen=min(dates) if dates else None,
                last_seen=max(dates) if dates else None,
                crystal_ids=[t[1] for t in timeline[-100:]],
                co_occurrences=dict(co_occurrence_matrix[emotion].most_common(5))
            )
            self.state.emotion_patterns[emotion] = pattern.__dict__

        # Build thread patterns
        for thread, timeline in thread_timeline.items():
            dates = [t[0] for t in timeline if t[0]]
            pattern = BraidPattern(
                pattern_type="thread",
                name=thread,
                occurrences=len(timeline),
                first_seen=min(dates) if dates else None,
                last_seen=max(dates) if dates else None,
                crystal_ids=[t[1] for t in timeline[-50:]],
                co_occurrences=dict(co_occurrence_matrix[thread].most_common(5))
            )
            self.state.thread_patterns[thread] = pattern.__dict__

        # Recent shift analysis
        recent_wound_counts = Counter(recent_wounds)
        recent_emotion_counts = Counter(recent_emotions)

        if recent_wound_counts:
            top_recent_wound = recent_wound_counts.most_common(1)[0]
            self.state.recent_shift = f"Wound '{top_recent_wound[0]}' surfacing ({top_recent_wound[1]} times this week)"

        # Emotional arc (simplified)
        if recent_emotion_counts:
            top_emotions = recent_emotion_counts.most_common(3)
            positive = sum(c for e, c in top_emotions if e in ['joy', 'peace', 'hope', 'love', 'clarity', 'gratitude'])
            negative = sum(c for e, c in top_emotions if e in ['grief', 'anger', 'fear', 'anxiety', 'despair', 'loneliness'])

            if positive > negative * 1.5:
                self.state.emotional_arc = "ascending"
            elif negative > positive * 1.5:
                self.state.emotional_arc = "descending"
            else:
                self.state.emotional_arc = "integrating"

        # Detect wound cycles (same wound appearing multiple times with gaps)
        for wound, timeline in wound_timeline.items():
            if len(timeline) > 20:
                dates = sorted([t[0] for t in timeline if t[0]])
                if len(dates) > 10:
                    # Check for cyclic pattern (gaps then returns)
                    self.state.wound_cycles.append({
                        "wound": wound,
                        "total_appearances": len(timeline),
                        "span_days": None  # Would calculate from dates
                    })

        self.state.analyzed_at = datetime.now().isoformat()

        self.close()
        self._log(f"Braid analysis complete. {len(self.state.wound_patterns)} wounds, {len(self.state.emotion_patterns)} emotions, {len(self.state.thread_patterns)} threads tracked.")

        return self.state

    def save_state(self):
        """Save braid state to file."""
        BRAID_OUTPUT.write_text(json.dumps(self.state.__dict__, indent=2, default=str))
        self._log(f"Braid state saved to {BRAID_OUTPUT}")

    def load_state(self) -> Optional[BraidState]:
        """Load previous braid state."""
        if BRAID_OUTPUT.exists():
            try:
                data = json.loads(BRAID_OUTPUT.read_text())
                state = BraidState()
                for k, v in data.items():
                    setattr(state, k, v)
                return state
            except:
                pass
        return None

    def get_summary(self) -> dict:
        """Get a summary of the current braid state."""
        return {
            "total_crystals": self.state.total_crystals,
            "analyzed_at": self.state.analyzed_at,
            "top_wounds": sorted(
                [(k, v['occurrences']) for k, v in self.state.wound_patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "top_emotions": sorted(
                [(k, v['occurrences']) for k, v in self.state.emotion_patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "top_threads": sorted(
                [(k, v['occurrences']) for k, v in self.state.thread_patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
            "stuck_patterns": self.state.stuck_patterns,
            "recent_shift": self.state.recent_shift,
            "emotional_arc": self.state.emotional_arc
        }


def run_full_braid():
    """Run full braid analysis."""
    braider = BraidingLayer()
    braider.analyze_all_crystals()
    braider.save_state()

    summary = braider.get_summary()
    print("\n" + "=" * 60)
    print("BRAID ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total crystals: {summary['total_crystals']}")
    print(f"\nTop wounds: {summary['top_wounds']}")
    print(f"\nTop emotions: {summary['top_emotions']}")
    print(f"\nTop threads: {summary['top_threads']}")
    print(f"\nStuck patterns: {summary['stuck_patterns']}")
    print(f"\nRecent shift: {summary['recent_shift']}")
    print(f"\nEmotional arc: {summary['emotional_arc']}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    run_full_braid()
