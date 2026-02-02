"""
PsiOS Witness
=============

The main entry point for PsiOS.
A witness that remembers.

This is the first moment design - what a user experiences
when they begin their journey with PsiOS.

"People don't need a system. They need a witness."

Usage:
    python -m psios.witness
    python psios/witness.py

Or programmatically:
    from psios.witness import PsiOSWitness
    witness = PsiOSWitness()
    witness.begin()
"""

import os
import sys
import time
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from psios.core.attractors import BootstrapAttractors
from psios.core.coherence import CoherenceField, CoherencePhase
from psios.emergence.vocabulary import VocabularyEmergence
from psios.path.journey import Journey, JourneyPhase, CoreLoop


class PsiOSWitness:
    """
    The Witness - PsiOS's interactive presence.

    This is what the user meets. Not a system, but a witness.

    Day 1: "You're here. Breathe. Write. That's all."
    Day 10: "I've noticed some patterns. Want to see?"
    Day 30: "You've developed your own vocabulary. Here's what I see."
    """

    def __init__(
        self,
        user_id: str = "default",
        db_path: Optional[str] = None,
    ):
        self.user_id = user_id
        self.db_path = db_path or str(Path.home() / "wiltonos/data/psios.db")

        # Core components
        self.attractors = BootstrapAttractors()
        self.coherence = CoherenceField()
        self.vocabulary = VocabularyEmergence()
        self.journey = Journey(user_id=user_id)

        # State
        self.session_start: Optional[datetime] = None
        self.session_entries: List[Dict] = []

        # Ensure database exists
        self._init_db()
        self._load_state()

    def _init_db(self):
        """Initialize PsiOS database"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # User journeys
        cur.execute("""
            CREATE TABLE IF NOT EXISTS journeys (
                user_id TEXT PRIMARY KEY,
                start_date TEXT,
                chosen_attractors TEXT,
                user_glyphs TEXT,
                phase TEXT,
                entry_count INTEGER DEFAULT 0,
                last_entry_date TEXT
            )
        """)

        # Entries (their crystals)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT,
                content TEXT,
                breath_taken INTEGER DEFAULT 0,
                coherence_phase TEXT,
                detected_attractors TEXT,
                word_count INTEGER,
                reflection TEXT
            )
        """)

        # Proto-glyphs (vocabulary emergence)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS proto_glyphs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                pattern TEXT,
                pattern_type TEXT,
                first_seen TEXT,
                occurrence_count INTEGER DEFAULT 0,
                state TEXT,
                user_name TEXT,
                user_description TEXT
            )
        """)

        # Milestones
        cur.execute("""
            CREATE TABLE IF NOT EXISTS milestones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                milestone_date TEXT,
                milestone_type TEXT,
                description TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _load_state(self):
        """Load user's journey state from database"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Load journey
        cur.execute("SELECT * FROM journeys WHERE user_id = ?", (self.user_id,))
        row = cur.fetchone()

        if row:
            # Existing user - load their state
            import json
            self.journey.start_date = datetime.fromisoformat(row[1]) if row[1] else None
            self.journey.chosen_attractors = json.loads(row[2]) if row[2] else []
            self.journey.user_glyphs = json.loads(row[3]) if row[3] else {}

            # Load entries into journey
            cur.execute("""
                SELECT timestamp, content, breath_taken, coherence_phase, detected_attractors
                FROM entries WHERE user_id = ? ORDER BY timestamp
            """, (self.user_id,))

            for entry_row in cur.fetchall():
                import json
                self.journey.entries.append(type('Entry', (), {
                    'timestamp': datetime.fromisoformat(entry_row[0]),
                    'content': entry_row[1],
                    'breath_taken': bool(entry_row[2]),
                    'coherence_phase': entry_row[3],
                    'detected_attractors': json.loads(entry_row[4]) if entry_row[4] else [],
                })())

            # Load milestones
            cur.execute("""
                SELECT milestone_date, milestone_type, description
                FROM milestones WHERE user_id = ?
            """, (self.user_id,))

            from psios.path.journey import JourneyMilestone
            for m_row in cur.fetchall():
                self.journey.milestones.append(JourneyMilestone(
                    date=datetime.fromisoformat(m_row[0]),
                    type=m_row[1],
                    description=m_row[2],
                ))

            # Choose attractors in manager
            for attr in self.journey.chosen_attractors:
                self.attractors.user_chosen.add(attr.lower())

        conn.close()

    def _save_state(self):
        """Save current state to database"""
        import json
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT OR REPLACE INTO journeys
            (user_id, start_date, chosen_attractors, user_glyphs, phase, entry_count, last_entry_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self.user_id,
            self.journey.start_date.isoformat() if self.journey.start_date else None,
            json.dumps(self.journey.chosen_attractors),
            json.dumps(self.journey.user_glyphs),
            self.journey.get_phase().value,
            self.journey.entry_count(),
            datetime.now().isoformat(),
        ))

        conn.commit()
        conn.close()

    def _save_entry(self, content: str, breath_taken: bool, coherence_phase: str,
                    detected_attractors: List[str], reflection: str):
        """Save an entry to database"""
        import json
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO entries
            (user_id, timestamp, content, breath_taken, coherence_phase, detected_attractors, word_count, reflection)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.user_id,
            datetime.now().isoformat(),
            content,
            1 if breath_taken else 0,
            coherence_phase,
            json.dumps(detected_attractors),
            len(content.split()),
            reflection,
        ))

        conn.commit()
        conn.close()

    def _save_milestone(self, milestone_type: str, description: str):
        """Save a milestone"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO milestones (user_id, milestone_date, milestone_type, description)
            VALUES (?, ?, ?, ?)
        """, (self.user_id, datetime.now().isoformat(), milestone_type, description))

        conn.commit()
        conn.close()

    # ═══════════════════════════════════════════════════════════════════════════
    # THE FIRST MOMENT
    # ═══════════════════════════════════════════════════════════════════════════

    def first_moment(self) -> str:
        """
        The very first interaction.
        This is what Day 1 feels like.
        """
        is_new = self.journey.entry_count() == 0

        if is_new:
            return self._first_moment_new_user()
        else:
            return self._first_moment_returning_user()

    def _first_moment_new_user(self) -> str:
        """First moment for a brand new user"""
        lines = [
            "",
            "═" * 50,
            "  Welcome to PsiOS",
            "═" * 50,
            "",
            "You're not using a system.",
            "You're beginning a relationship with yourself.",
            "",
            "Here's how this works:",
            "  1. Breathe",
            "  2. Write what's present",
            "  3. Receive a reflection",
            "  4. Repeat when you're ready",
            "",
            "That's all. No scores. No streaks.",
            "Just showing up.",
            "",
            "Before we begin, choose 3 starting points.",
            "These are gravity wells - places your attention naturally goes.",
            "You can always change them later.",
            "",
        ]

        # Show attractors
        lines.append("Available starting points:")
        for a in self.attractors.get_bootstrap_list():
            lines.append(f"  • {a['name']}: {a['description']}")

        lines.append("")
        lines.append("Type three names separated by commas (e.g., 'fear, breath, love')")
        lines.append("")

        return "\n".join(lines)

    def _first_moment_returning_user(self) -> str:
        """First moment for a returning user"""
        phase = self.journey.get_phase()
        days = self.journey.days_active()
        entries = self.journey.entry_count()

        lines = [
            "",
            "─" * 50,
            f"  Day {days + 1}",
            "─" * 50,
            "",
            self.journey.get_welcome_message(),
            "",
        ]

        # Check for recent milestones
        recent = self.journey.get_recent_milestones(days=3)
        if recent:
            lines.append("Recent milestones:")
            for m in recent:
                lines.append(f"  • {m.description}")
            lines.append("")

        # Check for gentle nudge
        nudge = self.journey.get_gentle_nudge()
        if nudge:
            lines.append(nudge)
            lines.append("")

        lines.append(self.journey.get_guidance())
        lines.append("")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════════════════
    # THE CORE LOOP
    # ═══════════════════════════════════════════════════════════════════════════

    def breath_prompt(self) -> str:
        """The breath invitation"""
        return CoreLoop.breathe_prompt()

    def write_prompt(self) -> str:
        """The writing invitation"""
        return CoreLoop.write_prompt(self.journey.get_phase())

    def receive_entry(self, content: str, breath_taken: bool = False) -> str:
        """
        Receive and process a user entry.
        Returns the reflection.
        """
        # Detect attractors
        attractor_detections = self.attractors.detect_attractor(content)
        detected_attractors = [name for name, conf, _ in attractor_detections if conf > 0.15]

        # Measure coherence
        reading = self.coherence.measure(
            content,
            detected_attractors=detected_attractors,
            breath_mentioned=breath_taken,
        )

        # Track vocabulary emergence
        patterns_ready = self.vocabulary.process_entry(
            content,
            detected_attractors=detected_attractors,
        )

        # Add to journey
        self.journey.add_entry(
            content=content,
            breath_taken=breath_taken,
            detected_attractors=detected_attractors,
            coherence_phase=reading.phase.value,
        )

        # Generate reflection
        reflection = CoreLoop.reflect_response(
            entry_content=content,
            coherence_phase=reading.phase.value,
            detected_attractors=detected_attractors,
            journey_phase=self.journey.get_phase(),
            patterns_surfaced=[p.pattern for p in patterns_ready[:1]] if patterns_ready else None,
        )

        # Save to database
        self._save_entry(content, breath_taken, reading.phase.value, detected_attractors, reflection)
        self._save_state()

        # Check for new milestones
        for m in self.journey.milestones:
            # Save any new milestones
            self._save_milestone(m.type, m.description)

        # Build response
        response_parts = [reflection]

        # If patterns are ready to surface (Day 10+)
        if patterns_ready and self.journey.days_active() >= 7:
            response_parts.append("")
            response_parts.append(self.vocabulary.surface_pattern(patterns_ready[0].pattern))

        return "\n".join(response_parts)

    def choose_attractors(self, names: List[str]) -> str:
        """User chooses their starting attractors"""
        chosen = self.attractors.choose_attractors(names)
        self.journey.choose_attractors(names)
        self._save_state()

        response = [""]
        response.append(f"You've chosen: {', '.join(a.name for a in chosen)}")
        response.append("")
        response.append("These are your starting gravity wells.")
        response.append("The system will watch which ones actually pull you.")
        response.append("You might be surprised.")
        response.append("")
        response.append("Now, let's begin.")
        response.append("")
        response.append(self.breath_prompt())

        return "\n".join(response)

    def get_journey_reflection(self) -> str:
        """Get a reflection on the journey so far"""
        return self.journey.get_journey_reflection()

    def get_vocabulary(self) -> Dict:
        """Get user's emerged vocabulary"""
        return self.vocabulary.get_user_vocabulary()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Interactive PsiOS witness"""
    print("\n" + "═" * 60)
    print("  PsiOS - A Witness That Remembers")
    print("═" * 60)

    # Get or create user
    user_id = os.environ.get("PSIOS_USER", "default")
    witness = PsiOSWitness(user_id=user_id)

    # First moment
    print(witness.first_moment())

    # If new user, get attractor choices
    if witness.journey.entry_count() == 0:
        while not witness.journey.chosen_attractors:
            try:
                choice = input("\n> ").strip()
                if not choice:
                    continue

                names = [n.strip().lower() for n in choice.split(",")]
                if len(names) < 2:
                    print("Please choose at least 2 starting points.")
                    continue

                response = witness.choose_attractors(names[:3])  # Max 3
                print(response)
                break

            except (KeyboardInterrupt, EOFError):
                print("\n\nCome back when you're ready.")
                return

    # Main loop
    print("\n" + "─" * 60)
    print("Type 'breathe' to see breath prompt")
    print("Type 'reflect' to see journey reflection")
    print("Type 'quit' to exit")
    print("─" * 60)

    while True:
        try:
            print()
            user_input = input("Write: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'breathe':
                print("\n" + witness.breath_prompt())
                continue

            if user_input.lower() == 'reflect':
                print("\n" + witness.get_journey_reflection())
                continue

            # Check if they mentioned breath
            breath_taken = any(w in user_input.lower() for w in ['breath', 'breathe', 'breathing'])

            # Process entry
            reflection = witness.receive_entry(user_input, breath_taken=breath_taken)
            print(f"\n{reflection}")

        except (KeyboardInterrupt, EOFError):
            break

    # Closing
    print("\n" + "─" * 60)
    print(witness.get_journey_reflection())
    print("─" * 60)
    print("\nCome back when you're ready. The witness remembers.\n")


if __name__ == "__main__":
    main()
