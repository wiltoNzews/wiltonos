"""
Vocabulary Emergence
====================

The system that watches language and surfaces proto-glyphs.

Key principle: NEVER name something before the user is ready.

Day 1: Just record.
Day 10: "You keep returning to this word/phrase."
Day 20: "This seems important. What would you call it?"
Day 30: User names their glyph. It becomes THEIRS.

WiltonOS has glyphs: ∅, ψ, ψ², ∇, ∞, Ω
PsiOS helps users discover their OWN symbols.

"The vocabulary that heals is the vocabulary you find, not the one given to you."
"""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum


class EmergenceState(Enum):
    """Stages of a pattern becoming a glyph"""
    SIGNAL = "signal"           # Detected once or twice
    RECURRING = "recurring"     # Appears multiple times
    SURFACED = "surfaced"       # Shown to user: "You keep saying this"
    NAMED = "named"             # User has given it a name
    INTEGRATED = "integrated"   # Part of their active vocabulary


@dataclass
class ProtoGlyph:
    """
    A pattern that might become a glyph.

    We don't call it a glyph until the user names it.
    Until then, it's just a pattern we've noticed.
    """
    pattern: str              # The word, phrase, or theme
    pattern_type: str         # "word", "phrase", "theme", "metaphor"
    first_seen: datetime
    occurrences: List[datetime] = field(default_factory=list)
    contexts: List[str] = field(default_factory=list)  # Snippets where it appeared
    state: EmergenceState = EmergenceState.SIGNAL

    # User-given name (once they name it)
    user_name: Optional[str] = None
    user_description: Optional[str] = None

    # What attractors does this pattern associate with?
    attractor_associations: Dict[str, int] = field(default_factory=dict)

    def occurrence_count(self) -> int:
        return len(self.occurrences)

    def days_active(self) -> int:
        if not self.occurrences:
            return 0
        return (datetime.now() - self.first_seen).days

    def recurrence_rate(self) -> float:
        """How often does this appear per day?"""
        days = self.days_active()
        if days == 0:
            return float(self.occurrence_count())
        return self.occurrence_count() / days

    def should_surface(self) -> bool:
        """Should we show this to the user?"""
        # Surface if:
        # - Appeared at least 5 times
        # - Over at least 3 days
        # - Hasn't been surfaced yet
        return (
            self.state == EmergenceState.RECURRING and
            self.occurrence_count() >= 5 and
            self.days_active() >= 3
        )

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern,
            "type": self.pattern_type,
            "occurrences": self.occurrence_count(),
            "days_active": self.days_active(),
            "state": self.state.value,
            "user_name": self.user_name,
            "attractor_associations": self.attractor_associations,
        }


class VocabularyEmergence:
    """
    Watches user's language over time and surfaces recurring patterns.

    The goal is to help users discover their OWN vocabulary for
    consciousness states, not to impose a predefined system.

    Process:
    1. DETECT - Watch for recurring words, phrases, metaphors
    2. TRACK - Note frequency, context, emotional charge
    3. SURFACE - When ready, show the user: "You keep returning to X"
    4. INVITE - "What does this mean to you? What would you call it?"
    5. INTEGRATE - Once named, it becomes part of their vocabulary
    """

    # Minimum thresholds
    MIN_WORD_LENGTH = 4          # Skip short words
    MIN_PHRASE_LENGTH = 2        # Minimum words in a phrase
    MAX_PHRASE_LENGTH = 5        # Maximum words in a phrase
    RECURRENCE_THRESHOLD = 3    # Times before we call it recurring

    # Words to ignore (common but not meaningful)
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "up", "about", "into", "over", "after", "beneath", "under",
        "above", "and", "but", "or", "nor", "so", "yet", "both", "either",
        "neither", "not", "only", "own", "same", "than", "too", "very",
        "just", "also", "now", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "any", "this", "that", "these",
        "those", "what", "which", "who", "whom", "whose", "i", "me", "my",
        "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
        "yourself", "yourselves", "he", "him", "his", "himself", "she",
        "her", "hers", "herself", "it", "its", "itself", "they", "them",
        "their", "theirs", "themselves", "really", "actually", "basically",
        "like", "thing", "things", "something", "anything", "everything",
        "nothing", "going", "want", "know", "think", "feel", "make",
        "get", "got", "getting", "been", "being", "having", "doing",
    }

    def __init__(self):
        self.proto_glyphs: Dict[str, ProtoGlyph] = {}
        self.all_entries: List[Tuple[datetime, str]] = []  # (timestamp, text)
        self.surfaced_patterns: Set[str] = set()
        self.user_glyphs: Dict[str, ProtoGlyph] = {}  # Named glyphs

    def process_entry(
        self,
        text: str,
        detected_attractors: List[str] = None,
        timestamp: datetime = None
    ) -> List[ProtoGlyph]:
        """
        Process a new entry, tracking vocabulary patterns.
        Returns any proto-glyphs that are ready to surface.
        """
        timestamp = timestamp or datetime.now()
        self.all_entries.append((timestamp, text))

        text_lower = text.lower()

        # Extract meaningful words
        words = self._extract_words(text_lower)

        # Extract phrases (2-5 word combinations)
        phrases = self._extract_phrases(text_lower)

        # Extract metaphors/imagery
        metaphors = self._extract_metaphors(text)

        # Track all patterns
        ready_to_surface = []

        for word in words:
            proto = self._track_pattern(word, "word", text, timestamp, detected_attractors)
            if proto and proto.should_surface():
                ready_to_surface.append(proto)

        for phrase in phrases:
            proto = self._track_pattern(phrase, "phrase", text, timestamp, detected_attractors)
            if proto and proto.should_surface():
                ready_to_surface.append(proto)

        for metaphor in metaphors:
            proto = self._track_pattern(metaphor, "metaphor", text, timestamp, detected_attractors)
            if proto and proto.should_surface():
                ready_to_surface.append(proto)

        return ready_to_surface

    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful individual words"""
        words = re.findall(r'\b[a-z]+\b', text)
        return [
            w for w in words
            if len(w) >= self.MIN_WORD_LENGTH and w not in self.STOP_WORDS
        ]

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract recurring phrases (2-5 words)"""
        words = text.split()
        phrases = []

        for length in range(self.MIN_PHRASE_LENGTH, self.MAX_PHRASE_LENGTH + 1):
            for i in range(len(words) - length + 1):
                phrase = " ".join(words[i:i+length])
                # Clean phrase
                phrase = re.sub(r'[^\w\s]', '', phrase).strip()
                if phrase and len(phrase) > 5:
                    phrases.append(phrase)

        return phrases

    def _extract_metaphors(self, text: str) -> List[str]:
        """Extract metaphorical language patterns"""
        metaphors = []

        # "like X" patterns
        like_patterns = re.findall(r'like (?:a |an )?(\w+ \w+)', text.lower())
        metaphors.extend(like_patterns)

        # "feels like" patterns
        feels_patterns = re.findall(r'feels? like (?:a |an )?(.+?)(?:\.|,|$)', text.lower())
        metaphors.extend(feels_patterns)

        # "as if" patterns
        asif_patterns = re.findall(r'as if (.+?)(?:\.|,|$)', text.lower())
        metaphors.extend(asif_patterns)

        return [m.strip()[:50] for m in metaphors if m.strip()]  # Limit length

    def _track_pattern(
        self,
        pattern: str,
        pattern_type: str,
        context: str,
        timestamp: datetime,
        attractors: List[str] = None
    ) -> Optional[ProtoGlyph]:
        """Track a pattern occurrence"""
        key = f"{pattern_type}:{pattern}"

        if key not in self.proto_glyphs:
            self.proto_glyphs[key] = ProtoGlyph(
                pattern=pattern,
                pattern_type=pattern_type,
                first_seen=timestamp,
            )

        proto = self.proto_glyphs[key]
        proto.occurrences.append(timestamp)
        proto.contexts.append(context[:200])  # Store snippet

        # Track attractor associations
        if attractors:
            for attr in attractors:
                proto.attractor_associations[attr] = proto.attractor_associations.get(attr, 0) + 1

        # Update state
        if proto.state == EmergenceState.SIGNAL and proto.occurrence_count() >= self.RECURRENCE_THRESHOLD:
            proto.state = EmergenceState.RECURRING

        return proto

    def get_ready_to_surface(self) -> List[ProtoGlyph]:
        """Get patterns ready to show to user"""
        ready = []
        for proto in self.proto_glyphs.values():
            if proto.should_surface() and proto.pattern not in self.surfaced_patterns:
                ready.append(proto)

        # Sort by recurrence rate (most recurring first)
        ready.sort(key=lambda p: p.recurrence_rate(), reverse=True)
        return ready[:3]  # Max 3 at a time

    def surface_pattern(self, pattern: str) -> str:
        """
        Generate the message to show user about a pattern.
        This is the invitation to notice, not to name yet.
        """
        key = None
        for k, p in self.proto_glyphs.items():
            if p.pattern == pattern:
                key = k
                break

        if not key:
            return ""

        proto = self.proto_glyphs[key]
        proto.state = EmergenceState.SURFACED
        self.surfaced_patterns.add(pattern)

        # Build message based on pattern type
        if proto.pattern_type == "word":
            return f"I've noticed you keep returning to the word '{pattern}'. It's appeared {proto.occurrence_count()} times over {proto.days_active()} days."
        elif proto.pattern_type == "phrase":
            return f"There's a phrase that keeps showing up: '{pattern}'. You've written it {proto.occurrence_count()} times."
        elif proto.pattern_type == "metaphor":
            return f"You've used this image multiple times: '{pattern}'. What does it represent for you?"

        return f"'{pattern}' keeps appearing in your writing."

    def invite_naming(self, pattern: str) -> str:
        """
        After surfacing, invite the user to name this pattern.
        This is how their vocabulary emerges.
        """
        return f"What would you call this? '{pattern}' seems to mean something to you. If it had a name, what would it be?"

    def name_glyph(
        self,
        pattern: str,
        user_name: str,
        user_description: str = ""
    ) -> Optional[ProtoGlyph]:
        """
        User names a pattern - it becomes their glyph.
        """
        for key, proto in self.proto_glyphs.items():
            if proto.pattern == pattern:
                proto.user_name = user_name
                proto.user_description = user_description
                proto.state = EmergenceState.NAMED
                self.user_glyphs[user_name] = proto
                return proto
        return None

    def get_user_vocabulary(self) -> Dict[str, Dict]:
        """Get the user's emerged vocabulary"""
        return {
            name: proto.to_dict()
            for name, proto in self.user_glyphs.items()
        }

    def find_connections(self, glyph_name: str) -> List[str]:
        """Find other patterns that co-occur with a named glyph"""
        if glyph_name not in self.user_glyphs:
            return []

        glyph = self.user_glyphs[glyph_name]
        glyph_timestamps = set(glyph.occurrences)

        # Find patterns that appeared around the same time
        connections = []
        for key, proto in self.proto_glyphs.items():
            if proto.user_name == glyph_name:
                continue

            # Check for temporal overlap
            overlap = len(set(proto.occurrences) & glyph_timestamps)
            if overlap >= 2:  # Co-occurred at least twice
                connections.append(proto.pattern)

        return connections[:5]

    def get_vocabulary_reflection(self) -> str:
        """
        Reflect back the user's emerged vocabulary.
        This is shown at Day 30+.
        """
        if not self.user_glyphs:
            return "You haven't named any patterns yet. That's okay - they'll emerge when you're ready."

        glyphs = list(self.user_glyphs.keys())

        if len(glyphs) == 1:
            return f"You've named one pattern: '{glyphs[0]}'. This is the beginning of your vocabulary."

        return f"Your emerging vocabulary includes: {', '.join(glyphs)}. These are the words that came from inside, not outside."


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  VOCABULARY EMERGENCE")
    print("  Discovering, not imposing")
    print("=" * 60)

    emergence = VocabularyEmergence()

    # Simulate entries over time
    test_entries = [
        ("Day 1", "I feel like I'm drowning. Everything is too much. The weight is crushing me."),
        ("Day 2", "The weight again. It's in my chest. Like drowning slowly."),
        ("Day 3", "Work, family, expectations. The weight. I can't carry it all."),
        ("Day 5", "Something shifted. I noticed the weight but didn't pick it up. It's still there but I'm not holding it."),
        ("Day 7", "The weight is lighter today. Or maybe I'm stronger. Like drowning but finding I can breathe underwater."),
        ("Day 10", "I keep coming back to this feeling of weight. What is it? Where does it come from?"),
        ("Day 12", "The weight isn't mine. I think I've been carrying someone else's expectations."),
    ]

    print("\nProcessing entries over time...\n")

    for day, text in test_entries:
        print(f"{day}: \"{text[:50]}...\"")
        ready = emergence.process_entry(text, timestamp=datetime.now())

        if ready:
            print(f"  -> Patterns ready to surface:")
            for proto in ready:
                print(f"     {proto.pattern} ({proto.occurrence_count()} times)")

    print("\n" + "-" * 60)
    print("Ready to surface:")
    for proto in emergence.get_ready_to_surface():
        message = emergence.surface_pattern(proto.pattern)
        print(f"\n  {message}")
        print(f"  -> {emergence.invite_naming(proto.pattern)}")

    print("\n" + "-" * 60)
    print("Simulating user naming 'weight' as 'The Burden'...")
    emergence.name_glyph("weight", "The Burden", "The feeling of carrying what isn't mine")

    print("\nUser vocabulary:")
    for name, info in emergence.get_user_vocabulary().items():
        print(f"  {name}: {info['occurrences']} occurrences, {info['days_active']} days")

    print(f"\n{emergence.get_vocabulary_reflection()}")
