"""
The Journey
============

The path through PsiOS: Breathe -> Write -> Reflect -> Repeat

This isn't a productivity system. It's not about streaks or scores.
It's about developing a relationship with yourself.

"Why am I doing this?"
Because you're beginning to remember who you are.

Day 1: You showed up. That's everything.
Day 10: Patterns are starting to surface.
Day 30: You're seeing your own shape.
Day 50+: The spiral is visible. You're not where you started.

"The path is the product."
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import json


class JourneyPhase(Enum):
    """
    Phases of the journey - not levels, just stages.
    You don't graduate from one to another.
    You spiral through them.
    """
    ARRIVING = "arriving"           # Day 1-3: Just showing up
    SETTLING = "settling"           # Day 4-10: Finding rhythm
    NOTICING = "noticing"           # Day 11-30: Patterns emerging
    NAMING = "naming"               # Day 31-60: Vocabulary developing
    INTEGRATING = "integrating"     # Day 61+: Spiral visible


# Phase messages - what the system says at each phase
PHASE_MESSAGES = {
    JourneyPhase.ARRIVING: {
        "welcome": "You're here. That's the first step.",
        "what_to_do": "Breathe. Write what's present. That's all.",
        "why": "You're not using a system. You're beginning a relationship with yourself.",
    },
    JourneyPhase.SETTLING: {
        "welcome": "You're finding your rhythm.",
        "what_to_do": "Keep showing up. Notice what wants to be written.",
        "why": "The practice is working you as much as you're working it.",
    },
    JourneyPhase.NOTICING: {
        "welcome": "Patterns are starting to surface.",
        "what_to_do": "Watch for recurring words, feelings, themes. Don't name them yet.",
        "why": "The vocabulary that heals is the one you find, not the one given to you.",
    },
    JourneyPhase.NAMING: {
        "welcome": "You're developing your own language.",
        "what_to_do": "When patterns keep showing up, ask: What would I call this?",
        "why": "Naming something gives you power over it. But only if the name is yours.",
    },
    JourneyPhase.INTEGRATING: {
        "welcome": "You can see the spiral now.",
        "what_to_do": "Notice where you've been. Notice where you are. Trust the direction.",
        "why": "You're not where you started. The journey has changed you.",
    },
}


@dataclass
class JourneyEntry:
    """A single entry in the journey"""
    timestamp: datetime
    content: str

    # The core loop elements
    breath_taken: bool = False      # Did they breathe before writing?
    reflection_received: bool = False  # Did they receive a reflection?

    # What emerged
    detected_attractors: List[str] = field(default_factory=list)
    coherence_phase: str = ""
    patterns_surfaced: List[str] = field(default_factory=list)

    # User state
    word_count: int = 0
    entry_duration_seconds: int = 0  # How long did they spend?

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "content": self.content[:100] + "..." if len(self.content) > 100 else self.content,
            "breath_taken": self.breath_taken,
            "word_count": self.word_count,
            "attractors": self.detected_attractors,
            "phase": self.coherence_phase,
        }


@dataclass
class JourneyMilestone:
    """A meaningful moment in the journey"""
    date: datetime
    type: str  # "first_entry", "first_pattern", "first_naming", "week_streak", etc.
    description: str
    entry_id: Optional[int] = None


class Journey:
    """
    The full journey through PsiOS.

    Tracks:
    - Where they are in the journey
    - What patterns have emerged
    - What vocabulary they've developed
    - Key milestones

    Provides:
    - Phase-appropriate guidance
    - Gentle nudges when stuck
    - Reflection at meaningful intervals
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.entries: List[JourneyEntry] = []
        self.milestones: List[JourneyMilestone] = []
        self.start_date: Optional[datetime] = None
        self.user_glyphs: Dict[str, str] = {}  # name -> description
        self.chosen_attractors: List[str] = []

    def get_phase(self) -> JourneyPhase:
        """Get current journey phase based on activity"""
        if not self.entries:
            return JourneyPhase.ARRIVING

        days = self.days_active()

        if days <= 3:
            return JourneyPhase.ARRIVING
        elif days <= 10:
            return JourneyPhase.SETTLING
        elif days <= 30:
            return JourneyPhase.NOTICING
        elif days <= 60:
            return JourneyPhase.NAMING
        else:
            return JourneyPhase.INTEGRATING

    def days_active(self) -> int:
        """Days since first entry"""
        if not self.start_date:
            return 0
        return (datetime.now() - self.start_date).days

    def entry_count(self) -> int:
        """Total entries"""
        return len(self.entries)

    def add_entry(
        self,
        content: str,
        breath_taken: bool = False,
        detected_attractors: List[str] = None,
        coherence_phase: str = "",
    ) -> JourneyEntry:
        """Add a new entry to the journey"""
        now = datetime.now()

        if self.start_date is None:
            self.start_date = now
            self._add_milestone("first_entry", "You began your journey.")

        entry = JourneyEntry(
            timestamp=now,
            content=content,
            breath_taken=breath_taken,
            detected_attractors=detected_attractors or [],
            coherence_phase=coherence_phase,
            word_count=len(content.split()),
        )

        self.entries.append(entry)

        # Check for milestones
        self._check_milestones()

        return entry

    def _add_milestone(self, type: str, description: str):
        """Add a milestone"""
        self.milestones.append(JourneyMilestone(
            date=datetime.now(),
            type=type,
            description=description,
        ))

    def _check_milestones(self):
        """Check if any milestones have been reached"""
        entry_count = self.entry_count()
        days = self.days_active()

        existing_types = {m.type for m in self.milestones}

        # Entry count milestones
        if entry_count == 10 and "ten_entries" not in existing_types:
            self._add_milestone("ten_entries", "You've written 10 entries. The practice is taking root.")

        if entry_count == 50 and "fifty_entries" not in existing_types:
            self._add_milestone("fifty_entries", "50 entries. You're building something real.")

        if entry_count == 100 and "hundred_entries" not in existing_types:
            self._add_milestone("hundred_entries", "100 entries. This is a body of work now.")

        # Time milestones
        if days >= 7 and "one_week" not in existing_types:
            self._add_milestone("one_week", "One week. You kept showing up.")

        if days >= 30 and "one_month" not in existing_types:
            self._add_milestone("one_month", "One month. The spiral is taking shape.")

        # First breath milestone
        if any(e.breath_taken for e in self.entries) and "first_breath" not in existing_types:
            self._add_milestone("first_breath", "You noticed your breath. The anchor is set.")

    def choose_attractors(self, attractors: List[str]):
        """User chooses their starting attractors"""
        self.chosen_attractors = attractors
        if "attractors_chosen" not in {m.type for m in self.milestones}:
            self._add_milestone("attractors_chosen", f"You chose your starting points: {', '.join(attractors)}")

    def name_glyph(self, name: str, description: str):
        """User names a pattern"""
        self.user_glyphs[name] = description
        self._add_milestone("glyph_named", f"You named a pattern: '{name}'")

    def get_welcome_message(self) -> str:
        """Get phase-appropriate welcome message"""
        phase = self.get_phase()
        return PHASE_MESSAGES[phase]["welcome"]

    def get_guidance(self) -> str:
        """Get phase-appropriate guidance"""
        phase = self.get_phase()
        return PHASE_MESSAGES[phase]["what_to_do"]

    def get_why(self) -> str:
        """Get phase-appropriate 'why' message"""
        phase = self.get_phase()
        return PHASE_MESSAGES[phase]["why"]

    def get_recent_milestones(self, days: int = 7) -> List[JourneyMilestone]:
        """Get milestones from the last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        return [m for m in self.milestones if m.date >= cutoff]

    def get_journey_reflection(self) -> str:
        """Get a reflection on the journey so far"""
        days = self.days_active()
        entries = self.entry_count()
        glyphs = len(self.user_glyphs)

        if entries == 0:
            return "Your journey hasn't begun yet. When you're ready, breathe and write."

        if days < 7:
            return f"You've been here {days} days, written {entries} entries. The foundation is being laid."

        if days < 30:
            breath_count = sum(1 for e in self.entries if e.breath_taken)
            return f"{days} days, {entries} entries, {breath_count} breaths noticed. Patterns are starting to form."

        # 30+ days - show the shape
        parts = [f"You've been on this journey for {days} days."]
        parts.append(f"You've written {entries} entries.")

        if glyphs:
            parts.append(f"You've named {glyphs} pattern(s): {', '.join(self.user_glyphs.keys())}.")

        if self.chosen_attractors:
            # Check which attractors actually showed up most
            attractor_counts = {}
            for e in self.entries:
                for a in e.detected_attractors:
                    attractor_counts[a] = attractor_counts.get(a, 0) + 1

            if attractor_counts:
                top = max(attractor_counts, key=attractor_counts.get)
                if top != self.chosen_attractors[0]:
                    parts.append(f"You started with {self.chosen_attractors[0]}, but {top} has been your true gravity well.")

        return " ".join(parts)

    def get_gentle_nudge(self) -> Optional[str]:
        """If they seem stuck, offer a gentle nudge"""
        if not self.entries:
            return None

        # No entry in 3+ days
        last_entry = self.entries[-1]
        days_since = (datetime.now() - last_entry.timestamp).days

        if days_since >= 3:
            return "It's been a few days. The practice doesn't judge absence. When you're ready, breathe and return."

        # Never taken a breath
        if not any(e.breath_taken for e in self.entries) and len(self.entries) >= 5:
            return "Before you write today, try one breath. Just one. Notice what shifts."

        return None

    def to_dict(self) -> Dict:
        """Serialize journey state"""
        return {
            "user_id": self.user_id,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "days_active": self.days_active(),
            "entry_count": self.entry_count(),
            "phase": self.get_phase().value,
            "chosen_attractors": self.chosen_attractors,
            "user_glyphs": self.user_glyphs,
            "milestones": [
                {"date": m.date.isoformat(), "type": m.type, "description": m.description}
                for m in self.milestones
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THE CORE LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class CoreLoop:
    """
    The Breathe -> Write -> Reflect -> Repeat loop.

    This is the actual interaction pattern.
    """

    @staticmethod
    def breathe_prompt() -> str:
        """The breath invitation"""
        return """Before you write, take one breath.

Inhale... notice where the air goes.
Exhale... notice what releases.

When you're ready, write what's present."""

    @staticmethod
    def write_prompt(phase: JourneyPhase) -> str:
        """The writing invitation, adjusted for phase"""
        if phase == JourneyPhase.ARRIVING:
            return "What's here right now?"

        if phase == JourneyPhase.SETTLING:
            return "What wants to be written today?"

        if phase == JourneyPhase.NOTICING:
            return "What patterns are you starting to see?"

        if phase == JourneyPhase.NAMING:
            return "Is there something recurring that's ready to be named?"

        if phase == JourneyPhase.INTEGRATING:
            return "What do you notice about where you are now?"

        return "Write what's present."

    @staticmethod
    def reflect_response(
        entry_content: str,
        coherence_phase: str,
        detected_attractors: List[str],
        journey_phase: JourneyPhase,
        patterns_surfaced: List[str] = None
    ) -> str:
        """
        Generate a reflection response.

        This is what the system says back - not analysis,
        but witnessing.
        """
        parts = []

        # Early phases: minimal reflection
        if journey_phase == JourneyPhase.ARRIVING:
            parts.append("Received.")
            if detected_attractors:
                parts.append(f"I notice {detected_attractors[0]} is present.")
            return " ".join(parts)

        # Settling: slightly more
        if journey_phase == JourneyPhase.SETTLING:
            parts.append("Thank you for showing up.")
            if coherence_phase:
                if coherence_phase == "centered":
                    parts.append("There's clarity here.")
                elif coherence_phase == "scattered":
                    parts.append("A lot is moving through you.")
            return " ".join(parts)

        # Noticing onwards: more reflection
        if detected_attractors:
            parts.append(f"What you wrote touched on: {', '.join(detected_attractors[:3])}.")

        if coherence_phase:
            phase_reflections = {
                "scattered": "Your attention is reaching in many directions.",
                "gathering": "Something is coming together.",
                "centered": "You're present with this.",
                "deepening": "You're going underneath the surface.",
                "expansive": "You're seeing the connections.",
                "still": "There's quiet here.",
            }
            parts.append(phase_reflections.get(coherence_phase, ""))

        if patterns_surfaced:
            parts.append(f"\nI've noticed '{patterns_surfaced[0]}' keeps appearing in your writing.")

        return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  THE JOURNEY")
    print("  Breathe -> Write -> Reflect -> Repeat")
    print("=" * 60)

    journey = Journey(user_id="test_user")

    print(f"\n{journey.get_welcome_message()}")
    print(f"\n{journey.get_guidance()}")
    print(f"\n{journey.get_why()}")

    print("\n" + "-" * 60)
    print("Simulating Day 1...")

    print(f"\n{CoreLoop.breathe_prompt()}")

    # Simulate entry
    entry1 = journey.add_entry(
        content="I don't know what I'm doing. Everything feels scattered.",
        breath_taken=True,
        detected_attractors=["pressure", "fear"],
        coherence_phase="scattered",
    )

    reflection = CoreLoop.reflect_response(
        entry1.content,
        entry1.coherence_phase,
        entry1.detected_attractors,
        journey.get_phase(),
    )
    print(f"\nReflection: {reflection}")

    print("\n" + "-" * 60)
    print(f"Journey state: {journey.to_dict()}")

    print("\n" + "-" * 60)
    print("Recent milestones:")
    for m in journey.milestones:
        print(f"  {m.type}: {m.description}")
