"""
Coherence Field
===============

Universal coherence measurement for PsiOS.

Unlike WiltonOS's Zλ (which measures Wilton's coherence-in-time),
this measures field coherence - patterns that emerge across writings.

Day 1: No score shown. Just reflection.
Day 10: "Your writings seem more grounded lately."
Day 30: "Here's the shape of your last month."

Coherence isn't a number to chase. It's a mirror to notice.

"The system doesn't judge coherence. It witnesses it."
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class CoherencePhase(Enum):
    """
    Phases of coherence - not better/worse, just different.
    Uses accessible language, not glyphs (yet).
    """
    SCATTERED = "scattered"      # Many threads, no center
    GATHERING = "gathering"      # Starting to coalesce
    CENTERED = "centered"        # Clear presence
    DEEPENING = "deepening"      # Going under the surface
    EXPANSIVE = "expansive"      # Wide awareness, connected
    STILL = "still"              # Deep quiet, few words needed


# Phase descriptions for users (no jargon)
PHASE_DESCRIPTIONS = {
    CoherencePhase.SCATTERED: "Your attention is moving between many things. That's okay - sometimes we need to touch everything before we know what matters.",
    CoherencePhase.GATHERING: "Something is starting to come together. You're circling around a center, even if you can't name it yet.",
    CoherencePhase.CENTERED: "You're present. Your words have weight. This is a good place to ask important questions.",
    CoherencePhase.DEEPENING: "You're going underneath the surface. This might feel uncomfortable - that's often where the real material lives.",
    CoherencePhase.EXPANSIVE: "Your awareness is wide. You're seeing connections, patterns, the bigger picture.",
    CoherencePhase.STILL: "Few words. Deep ground. Sometimes the most coherent state has nothing to say.",
}


@dataclass
class CoherenceReading:
    """A single coherence measurement"""
    timestamp: datetime
    phase: CoherencePhase
    indicators: Dict[str, float]  # What contributed to this reading
    breath_present: bool = False  # Was breath awareness detected?
    body_present: bool = False    # Was body awareness detected?

    # The raw signal (0-1) - but we don't show this to users early on
    raw_signal: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "phase": self.phase.value,
            "indicators": self.indicators,
            "breath_present": self.breath_present,
            "body_present": self.body_present,
        }


@dataclass
class CoherenceArc:
    """
    A span of coherence over time.
    This is what we show users at Day 30+ - not numbers, but shapes.
    """
    start_date: datetime
    end_date: datetime
    readings: List[CoherenceReading] = field(default_factory=list)

    # Derived patterns
    dominant_phase: Optional[CoherencePhase] = None
    phase_transitions: List[Tuple[CoherencePhase, CoherencePhase]] = field(default_factory=list)
    recurring_patterns: List[str] = field(default_factory=list)

    def get_shape_description(self) -> str:
        """Describe the arc in human terms, not metrics"""
        if not self.readings:
            return "Not enough time has passed to see a shape yet."

        # Count phases
        phase_counts = {}
        for r in self.readings:
            phase_counts[r.phase] = phase_counts.get(r.phase, 0) + 1

        dominant = max(phase_counts, key=phase_counts.get) if phase_counts else None

        # Detect trajectory
        if len(self.readings) >= 3:
            early = self.readings[:len(self.readings)//3]
            late = self.readings[-len(self.readings)//3:]

            early_avg = sum(r.raw_signal for r in early) / len(early)
            late_avg = sum(r.raw_signal for r in late) / len(late)

            if late_avg > early_avg + 0.15:
                trajectory = "Your coherence has been deepening over this period."
            elif late_avg < early_avg - 0.15:
                trajectory = "You've been in more scattered states lately. That's often part of processing something."
            else:
                trajectory = "Your coherence has been relatively steady."
        else:
            trajectory = ""

        # Build description
        desc = f"Over this period, you've spent most time in a {dominant.value} state. "
        desc += PHASE_DESCRIPTIONS.get(dominant, "")
        if trajectory:
            desc += f" {trajectory}"

        return desc


class CoherenceField:
    """
    The coherence measurement engine for PsiOS.

    Unlike a score to optimize, this is a mirror that reflects
    the shape of someone's attention over time.

    Key principle: Show patterns, not numbers.
    Day 1: "You're here. That's enough."
    Day 10: "Your attention tends to gather around these themes."
    Day 30: "Here's the shape of your month."
    """

    # Minimum days before showing coherence patterns
    PATTERN_THRESHOLD_DAYS = 7

    # Minimum entries before showing arc shapes
    ARC_THRESHOLD_ENTRIES = 10

    def __init__(self):
        self.readings: List[CoherenceReading] = []
        self.first_entry_date: Optional[datetime] = None

    def measure(
        self,
        text: str,
        detected_attractors: List[str] = None,
        breath_mentioned: bool = False,
        body_mentioned: bool = False,
    ) -> CoherenceReading:
        """
        Measure coherence from a piece of writing.

        This isn't judgment - it's witnessing.
        """
        now = datetime.now()

        if self.first_entry_date is None:
            self.first_entry_date = now

        indicators = {}

        # Length as indicator (very short or very long = different states)
        word_count = len(text.split())
        if word_count < 10:
            indicators["brevity"] = 0.8  # Could be still/deep or just quick
        elif word_count > 200:
            indicators["verbosity"] = 0.7  # Could be processing or scattered
        else:
            indicators["moderate_length"] = 0.5

        # Question marks = seeking
        question_count = text.count("?")
        if question_count > 3:
            indicators["questioning"] = 0.7

        # Exclamation = intensity
        exclaim_count = text.count("!")
        if exclaim_count > 2:
            indicators["intensity"] = 0.6

        # First person = self-focus (not bad, just noting)
        i_count = text.lower().split().count("i")
        if i_count > word_count * 0.1:  # More than 10% "I"
            indicators["self_focus"] = 0.6

        # Breath/body awareness markers
        breath_markers = ["breath", "breathe", "breathing", "inhale", "exhale"]
        body_markers = ["body", "chest", "stomach", "hands", "feet", "tension", "relax"]

        breath_present = breath_mentioned or any(m in text.lower() for m in breath_markers)
        body_present = body_mentioned or any(m in text.lower() for m in body_markers)

        if breath_present:
            indicators["breath_awareness"] = 0.8
        if body_present:
            indicators["body_awareness"] = 0.7

        # Attractor concentration
        if detected_attractors:
            if len(set(detected_attractors)) == 1:
                indicators["focused"] = 0.7  # Single attractor = focused
            elif len(detected_attractors) > 3:
                indicators["diffuse"] = 0.5  # Many attractors = processing

        # Calculate raw signal (but we don't show this early)
        raw_signal = self._calculate_raw_signal(indicators, breath_present, body_present)

        # Detect phase
        phase = self._detect_phase(indicators, raw_signal, word_count)

        reading = CoherenceReading(
            timestamp=now,
            phase=phase,
            indicators=indicators,
            breath_present=breath_present,
            body_present=body_present,
            raw_signal=raw_signal,
        )

        self.readings.append(reading)
        return reading

    def _calculate_raw_signal(
        self,
        indicators: Dict[str, float],
        breath_present: bool,
        body_present: bool
    ) -> float:
        """
        Calculate a raw coherence signal (0-1).
        This is internal - not shown to users directly.
        """
        # Presence indicators boost coherence
        presence_bonus = 0.0
        if breath_present:
            presence_bonus += 0.15
        if body_present:
            presence_bonus += 0.1

        # Focused attention boosts coherence
        focus_bonus = indicators.get("focused", 0) * 0.1
        focus_bonus -= indicators.get("diffuse", 0) * 0.1

        # Start at baseline
        signal = 0.5 + presence_bonus + focus_bonus

        # Moderate length is slightly coherent
        if "moderate_length" in indicators:
            signal += 0.05

        # Very brief could be stillness (high) or avoidance (low) - middle ground
        if "brevity" in indicators:
            signal += 0.0  # Neutral - context determines

        return max(0.0, min(1.0, signal))

    def _detect_phase(
        self,
        indicators: Dict[str, float],
        raw_signal: float,
        word_count: int
    ) -> CoherencePhase:
        """Detect coherence phase from indicators"""

        # Very brief + high presence markers = STILL
        if word_count < 20 and (indicators.get("breath_awareness") or indicators.get("body_awareness")):
            return CoherencePhase.STILL

        # Many questions = GATHERING or DEEPENING
        if indicators.get("questioning"):
            if raw_signal > 0.6:
                return CoherencePhase.DEEPENING
            else:
                return CoherencePhase.GATHERING

        # High focus + moderate signal = CENTERED
        if indicators.get("focused") and raw_signal > 0.55:
            return CoherencePhase.CENTERED

        # Diffuse attention = SCATTERED or EXPANSIVE
        if indicators.get("diffuse"):
            if raw_signal > 0.6:
                return CoherencePhase.EXPANSIVE  # High signal diffuse = seeing connections
            else:
                return CoherencePhase.SCATTERED  # Low signal diffuse = unfocused

        # Default based on signal
        if raw_signal > 0.7:
            return CoherencePhase.CENTERED
        elif raw_signal > 0.5:
            return CoherencePhase.GATHERING
        else:
            return CoherencePhase.SCATTERED

    def should_show_patterns(self) -> bool:
        """Have enough entries accumulated to show patterns?"""
        if self.first_entry_date is None:
            return False

        days_active = (datetime.now() - self.first_entry_date).days
        return days_active >= self.PATTERN_THRESHOLD_DAYS and len(self.readings) >= self.ARC_THRESHOLD_ENTRIES

    def get_current_reflection(self) -> str:
        """
        Get a reflection for the user.
        Content changes based on how long they've been using the system.
        """
        if not self.readings:
            return "You're here. That's the first step."

        days_active = (datetime.now() - self.first_entry_date).days if self.first_entry_date else 0
        latest = self.readings[-1]

        # Day 1-7: Just presence, no analysis
        if days_active < 7:
            phase_desc = PHASE_DESCRIPTIONS.get(latest.phase, "")
            if latest.breath_present:
                return f"You noticed your breath. {phase_desc}"
            else:
                return f"You showed up. {phase_desc}"

        # Day 7-30: Start showing patterns
        if days_active < 30:
            # Find most common phase
            phase_counts = {}
            for r in self.readings[-10:]:  # Last 10 entries
                phase_counts[r.phase] = phase_counts.get(r.phase, 0) + 1

            if phase_counts:
                common_phase = max(phase_counts, key=phase_counts.get)
                return f"Lately you've been in a {common_phase.value} state often. {PHASE_DESCRIPTIONS.get(common_phase, '')}"

        # Day 30+: Show the arc
        arc = self.get_arc(days=30)
        return arc.get_shape_description()

    def get_arc(self, days: int = 30) -> CoherenceArc:
        """Get coherence arc for last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        relevant_readings = [r for r in self.readings if r.timestamp >= cutoff]

        arc = CoherenceArc(
            start_date=cutoff,
            end_date=datetime.now(),
            readings=relevant_readings,
        )

        if relevant_readings:
            # Find dominant phase
            phase_counts = {}
            for r in relevant_readings:
                phase_counts[r.phase] = phase_counts.get(r.phase, 0) + 1
            arc.dominant_phase = max(phase_counts, key=phase_counts.get)

            # Find transitions
            for i in range(1, len(relevant_readings)):
                prev_phase = relevant_readings[i-1].phase
                curr_phase = relevant_readings[i].phase
                if prev_phase != curr_phase:
                    arc.phase_transitions.append((prev_phase, curr_phase))

        return arc

    def get_gentle_nudge(self) -> Optional[str]:
        """
        If the user seems stuck, offer a gentle nudge.
        Not prescriptive - just an invitation.
        """
        if len(self.readings) < 3:
            return None

        recent = self.readings[-3:]

        # Three scattered readings in a row
        if all(r.phase == CoherencePhase.SCATTERED for r in recent):
            return "A lot is moving through you. Would a breath help?"

        # No body/breath awareness in recent entries
        if not any(r.breath_present or r.body_present for r in recent):
            return "What do you notice in your body right now?"

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  COHERENCE FIELD")
    print("  Witnessing, not judging")
    print("=" * 60)

    field = CoherenceField()

    test_entries = [
        "I don't know what I'm doing. Everything feels scattered. Work, relationships, this project. I can't focus on any of it.",
        "I noticed my chest is tight. Taking a breath. The tension is in my shoulders too.",
        "What am I afraid of? Why do I keep avoiding this? What would happen if I just let go?",
        "Breath. Here. Now. That's all.",
        "I see it now. The pattern. The fear connects to the control, and the control is what's creating the pressure. It's all one thing.",
    ]

    print("\nProcessing entries over simulated time...\n")

    for i, text in enumerate(test_entries):
        # Simulate some attractors being detected
        attractors = ["fear", "control"] if "afraid" in text or "control" in text else []

        reading = field.measure(
            text,
            detected_attractors=attractors,
        )

        print(f"Entry {i+1}:")
        print(f"  \"{text[:60]}...\"")
        print(f"  Phase: {reading.phase.value}")
        print(f"  Breath aware: {reading.breath_present}")
        print(f"  Indicators: {list(reading.indicators.keys())}")
        print()

    print("-" * 60)
    print("Reflection for user:")
    print(f"  {field.get_current_reflection()}")

    nudge = field.get_gentle_nudge()
    if nudge:
        print(f"\nGentle nudge: {nudge}")
