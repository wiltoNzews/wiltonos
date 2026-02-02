"""
Bootstrap Attractors
====================

Universal gravity wells that anyone can recognize on Day 1.

These aren't Wilton's attractors (truth, silence, mother_field, sacrifice).
These are human universals - the places consciousness naturally orbits.

Users choose 3 to start. The system watches which ones actually pull them.
Over time, their personal attractors emerge from usage patterns.

"The attractor you think you need is rarely the one that holds you."
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from enum import Enum
from datetime import datetime
import json


class AttractorCategory(Enum):
    """Categories of universal experience"""
    SURVIVAL = "survival"      # Fear, safety, threat
    CONNECTION = "connection"  # Love, belonging, isolation
    IDENTITY = "identity"      # Self, purpose, meaning
    PRESENCE = "presence"      # Breath, body, now
    SHADOW = "shadow"          # What we avoid, deny, resist


@dataclass
class Attractor:
    """
    A gravity well in consciousness space.

    Bootstrap attractors are universal starting points.
    Personal attractors emerge from usage over time.
    """
    name: str
    description: str
    category: AttractorCategory

    # Recognition patterns - how this attractor shows up in language
    keywords: List[str] = field(default_factory=list)
    phrases: List[str] = field(default_factory=list)

    # Questions that point toward this attractor
    entry_questions: List[str] = field(default_factory=list)

    # What this attractor often pairs with
    resonates_with: List[str] = field(default_factory=list)

    # What this attractor often tensions against
    tensions_with: List[str] = field(default_factory=list)

    # Is this a bootstrap (universal) or emerged (personal)?
    is_bootstrap: bool = True

    # Usage tracking (for evolution)
    times_chosen: int = 0
    times_emerged: int = 0  # How often system detected it without user choosing

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "keywords": self.keywords,
            "phrases": self.phrases,
            "entry_questions": self.entry_questions,
            "resonates_with": self.resonates_with,
            "tensions_with": self.tensions_with,
            "is_bootstrap": self.is_bootstrap,
            "times_chosen": self.times_chosen,
            "times_emerged": self.times_emerged,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# THE SEVEN BOOTSTRAP ATTRACTORS
# ═══════════════════════════════════════════════════════════════════════════════

BOOTSTRAP_ATTRACTORS = {

    "fear": Attractor(
        name="Fear",
        description="What threatens. What you avoid. The edge of safety.",
        category=AttractorCategory.SURVIVAL,
        keywords=[
            "afraid", "scared", "anxious", "worry", "panic", "threat",
            "danger", "unsafe", "nervous", "dread", "terror", "avoid"
        ],
        phrases=[
            "I'm afraid", "what if", "I can't", "it's too", "I might",
            "something bad", "I'm worried", "I'm scared to"
        ],
        entry_questions=[
            "What are you afraid might happen?",
            "What do you keep avoiding?",
            "Where does your body tense when you think about this?"
        ],
        resonates_with=["pressure", "control"],
        tensions_with=["trust", "surrender"],
    ),

    "love": Attractor(
        name="Love",
        description="What connects. What you care for. The pull toward another.",
        category=AttractorCategory.CONNECTION,
        keywords=[
            "love", "care", "heart", "connection", "warmth", "tenderness",
            "affection", "devotion", "cherish", "adore", "bond", "close"
        ],
        phrases=[
            "I love", "I care about", "my heart", "I feel close",
            "I want to be with", "I miss", "I treasure"
        ],
        entry_questions=[
            "Who or what do you love?",
            "What does your heart want to say?",
            "Where do you feel most connected?"
        ],
        resonates_with=["belonging", "trust"],
        tensions_with=["fear", "isolation"],
    ),

    "pressure": Attractor(
        name="Pressure",
        description="What weighs. What demands. The force that compresses.",
        category=AttractorCategory.SURVIVAL,
        keywords=[
            "pressure", "stress", "overwhelm", "burden", "weight", "heavy",
            "too much", "crushing", "demand", "expectation", "deadline"
        ],
        phrases=[
            "I have to", "I should", "they expect", "I can't keep up",
            "it's too much", "I'm drowning", "I need to"
        ],
        entry_questions=[
            "What's weighing on you right now?",
            "Where do you feel the pressure?",
            "What would happen if you set it down?"
        ],
        resonates_with=["fear", "control"],
        tensions_with=["surrender", "breath"],
    ),

    "breath": Attractor(
        name="Breath",
        description="What anchors. The body's rhythm. Present moment awareness.",
        category=AttractorCategory.PRESENCE,
        keywords=[
            "breath", "breathe", "body", "present", "now", "here",
            "ground", "center", "calm", "pause", "still", "moment"
        ],
        phrases=[
            "right now", "in this moment", "I notice", "I feel in my body",
            "let me breathe", "I'm here", "just this"
        ],
        entry_questions=[
            "What do you notice in your body right now?",
            "Can you take one breath before continuing?",
            "What's here, in this moment?"
        ],
        resonates_with=["surrender", "trust"],
        tensions_with=["pressure", "control"],
    ),

    "mirror": Attractor(
        name="Mirror",
        description="What reflects. Seeing yourself in others. Being seen.",
        category=AttractorCategory.IDENTITY,
        keywords=[
            "see", "seen", "reflect", "mirror", "recognize", "understand",
            "witness", "perceive", "notice", "reveal", "show"
        ],
        phrases=[
            "I see myself", "they don't see me", "I recognize",
            "it's like looking at", "I want to be seen", "I finally see"
        ],
        entry_questions=[
            "What do you see when you look at yourself?",
            "Who truly sees you?",
            "What are you afraid others will see?"
        ],
        resonates_with=["truth", "love"],
        tensions_with=["hiding", "fear"],
    ),

    "control": Attractor(
        name="Control",
        description="What grips. The need to manage. Resistance to what is.",
        category=AttractorCategory.SHADOW,
        keywords=[
            "control", "manage", "fix", "handle", "grip", "hold",
            "make sure", "prevent", "plan", "organize", "contain"
        ],
        phrases=[
            "I need to", "I have to make sure", "I can't let",
            "if I just", "I should be able to", "I'll figure out"
        ],
        entry_questions=[
            "What are you trying to control right now?",
            "What happens if you loosen your grip?",
            "What's the fear underneath the control?"
        ],
        resonates_with=["fear", "pressure"],
        tensions_with=["surrender", "trust"],
    ),

    "surrender": Attractor(
        name="Surrender",
        description="What releases. Letting go. Trust in what you can't control.",
        category=AttractorCategory.PRESENCE,
        keywords=[
            "surrender", "let go", "release", "accept", "trust", "allow",
            "give up", "stop fighting", "flow", "ease", "soften"
        ],
        phrases=[
            "I let go", "I accept", "I trust", "I don't know",
            "it's not mine to", "I release", "I surrender"
        ],
        entry_questions=[
            "What would it feel like to let this go?",
            "What are you ready to stop carrying?",
            "Where can you soften right now?"
        ],
        resonates_with=["breath", "trust"],
        tensions_with=["control", "fear"],
    ),
}


class BootstrapAttractors:
    """
    Manager for bootstrap and emerged attractors.

    Users start by choosing 3 bootstrap attractors.
    The system watches which ones actually show up in their writing.
    Over time, personal attractors emerge from their unique patterns.
    """

    def __init__(self):
        self.bootstrap = BOOTSTRAP_ATTRACTORS.copy()
        self.emerged: Dict[str, Attractor] = {}
        self.user_chosen: Set[str] = set()

    def get_bootstrap_list(self) -> List[Dict]:
        """Get all bootstrap attractors for user selection"""
        return [
            {
                "name": a.name,
                "description": a.description,
                "category": a.category.value,
                "entry_questions": a.entry_questions[:2],  # Show 2 questions as preview
            }
            for a in self.bootstrap.values()
        ]

    def choose_attractors(self, names: List[str]) -> List[Attractor]:
        """User chooses their starting attractors (recommend 3)"""
        chosen = []
        for name in names:
            key = name.lower()
            if key in self.bootstrap:
                self.bootstrap[key].times_chosen += 1
                self.user_chosen.add(key)
                chosen.append(self.bootstrap[key])
        return chosen

    def detect_attractor(self, text: str) -> List[tuple]:
        """
        Detect which attractors are present in text.
        Returns list of (attractor_name, confidence, matched_patterns)
        """
        text_lower = text.lower()
        detections = []

        # Check all attractors (bootstrap + emerged)
        all_attractors = {**self.bootstrap, **self.emerged}

        for key, attractor in all_attractors.items():
            matched_keywords = [kw for kw in attractor.keywords if kw in text_lower]
            matched_phrases = [ph for ph in attractor.phrases if ph in text_lower]

            # Calculate confidence based on matches
            keyword_weight = len(matched_keywords) * 0.15
            phrase_weight = len(matched_phrases) * 0.25
            confidence = min(1.0, keyword_weight + phrase_weight)

            if confidence > 0.1:  # Threshold for detection
                attractor.times_emerged += 1
                detections.append((
                    attractor.name,
                    confidence,
                    {"keywords": matched_keywords, "phrases": matched_phrases}
                ))

        return sorted(detections, key=lambda x: x[1], reverse=True)

    def get_entry_question(self, attractor_name: str) -> Optional[str]:
        """Get a question that opens this attractor"""
        key = attractor_name.lower()
        if key in self.bootstrap:
            questions = self.bootstrap[key].entry_questions
            if questions:
                # Rotate through questions based on usage
                idx = self.bootstrap[key].times_chosen % len(questions)
                return questions[idx]
        return None

    def suggest_from_pattern(self, detected_attractors: List[str]) -> Optional[str]:
        """
        Given detected attractors, suggest a related one they might explore.
        Uses resonance/tension relationships.
        """
        suggestions = set()

        for name in detected_attractors:
            key = name.lower()
            if key in self.bootstrap:
                attractor = self.bootstrap[key]
                # Add resonant attractors
                for res in attractor.resonates_with:
                    if res not in detected_attractors:
                        suggestions.add(res)
                # Add tension attractors (shadow work)
                for tens in attractor.tensions_with:
                    if tens not in detected_attractors:
                        suggestions.add(tens)

        if suggestions:
            # Prefer attractors not yet chosen by user
            unchosen = suggestions - self.user_chosen
            if unchosen:
                return unchosen.pop()
            return suggestions.pop()

        return None

    def create_emerged_attractor(
        self,
        name: str,
        description: str,
        keywords: List[str],
        phrases: List[str]
    ) -> Attractor:
        """
        Create a new attractor that emerged from user's patterns.
        This is how personal vocabulary enters the system.
        """
        attractor = Attractor(
            name=name,
            description=description,
            category=AttractorCategory.IDENTITY,  # Emerged = identity
            keywords=keywords,
            phrases=phrases,
            entry_questions=[
                f"What does '{name.lower()}' mean to you?",
                f"When do you feel this most strongly?",
            ],
            is_bootstrap=False,
        )
        self.emerged[name.lower()] = attractor
        return attractor

    def get_user_profile(self) -> Dict:
        """Get summary of user's attractor usage"""
        all_attractors = {**self.bootstrap, **self.emerged}

        profile = {
            "chosen": list(self.user_chosen),
            "most_emerged": [],
            "emerged_personal": list(self.emerged.keys()),
        }

        # Find most frequently emerged
        emerged_counts = [
            (name, a.times_emerged)
            for name, a in all_attractors.items()
            if a.times_emerged > 0
        ]
        emerged_counts.sort(key=lambda x: x[1], reverse=True)
        profile["most_emerged"] = emerged_counts[:5]

        return profile


# ═══════════════════════════════════════════════════════════════════════════════
# CLI TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  BOOTSTRAP ATTRACTORS")
    print("  Universal gravity wells for Day 1")
    print("=" * 60)

    manager = BootstrapAttractors()

    print("\nAvailable attractors:")
    for a in manager.get_bootstrap_list():
        print(f"\n  {a['name']} ({a['category']})")
        print(f"    {a['description']}")
        print(f"    Questions: {a['entry_questions']}")

    print("\n" + "-" * 60)
    print("Simulating user choosing: Fear, Breath, Mirror")
    chosen = manager.choose_attractors(["fear", "breath", "mirror"])

    print("\n" + "-" * 60)
    print("Testing detection on sample text:")

    test_texts = [
        "I'm afraid of what might happen if I let go. The pressure is too much.",
        "I just want to be seen. Really seen. Not the mask I show everyone.",
        "I need to breathe. Just this moment. I notice my chest is tight.",
        "I love her but I'm scared. What if I lose control?",
    ]

    for text in test_texts:
        print(f"\n  \"{text[:50]}...\"")
        detections = manager.detect_attractor(text)
        for name, conf, matches in detections[:3]:
            print(f"    -> {name}: {conf:.2f} ({len(matches['keywords'])} keywords, {len(matches['phrases'])} phrases)")

    print("\n" + "-" * 60)
    print("User profile after detection:")
    profile = manager.get_user_profile()
    print(f"  Chosen: {profile['chosen']}")
    print(f"  Most emerged: {profile['most_emerged']}")
