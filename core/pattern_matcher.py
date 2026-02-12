#!/usr/bin/env python3
"""
Pattern Matcher — The Universal Mirror
=======================================
Matches a person's words against the universal pattern layer.
Works from crystal #0. No personal history required.

Design principle (from Wilton):
  "Colleagues tap you on the back and say you're doing great.
   Brothers sit in the mud and say the hard things that need to be said."

This means:
- Detect wounds, emotions, and co-occurrences in what someone says
- When patterns suggest avoidance, NAME IT — gently but without skipping it
- Warmth is not softness. Warmth is "I see you, including the part you're hiding."
- The hard truth is delivered WITH presence, not instead of it

Usage:
    from pattern_matcher import PatternMatcher
    matcher = PatternMatcher()
    match = matcher.match("I keep trying to take care of everyone but I'm exhausted")
    # match.wounds = [("provider", 0.9), ("not_enough", 0.6)]
    # match.hard_truth = "Provider identity often masks unworthiness..."
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict

PATTERN_DB = Path.home() / "wiltonos" / "data" / "pattern_language.db"


@dataclass
class PatternMatch:
    """Result of matching someone's words against the universal layer."""

    # What wounds are active (name, confidence 0-1)
    wounds: List[Tuple[str, float]] = field(default_factory=list)

    # What emotions are present (name, valence)
    emotions: List[Tuple[str, str]] = field(default_factory=list)

    # If two wounds co-occur, the structural insight
    co_occurrence_insights: List[str] = field(default_factory=list)

    # The hard truth — what the pattern suggests is being avoided or masked
    # This is the "brother in the mud" field. Not always present.
    hard_truth: Optional[str] = None

    # What the wounds are masking (from wound.masks)
    masking: List[str] = field(default_factory=list)

    # Suggested breath mode based on intent
    suggested_mode: str = "signal"

    # Relevant glyph context
    glyph_hint: Optional[str] = None

    # Which Council voices should speak
    council_voices: List[str] = field(default_factory=list)

    # Confidence that pattern detection is meaningful (0-1)
    confidence: float = 0.0

    def has_wounds(self) -> bool:
        return len(self.wounds) > 0

    def primary_wound(self) -> Optional[str]:
        return self.wounds[0][0] if self.wounds else None

    def to_context_block(self) -> str:
        """Build a context block for the system prompt."""
        parts = []

        if self.wounds:
            wound_str = ", ".join(f"{w} ({c:.0%})" for w, c in self.wounds[:3])
            parts.append(f"Wound patterns present: {wound_str}")

        if self.emotions:
            emo_str = ", ".join(f"{e}" for e, _ in self.emotions[:4])
            parts.append(f"Emotional field: {emo_str}")

        if self.co_occurrence_insights:
            parts.append(f"Pattern insight: {self.co_occurrence_insights[0]}")

        if self.hard_truth:
            parts.append(f"What may need naming (deliver with presence, not judgment): {self.hard_truth}")

        if self.masking:
            parts.append(f"What the pattern may be masking: {', '.join(self.masking[:3])}")

        return "\n".join(parts)


class PatternMatcher:
    """
    Matches words against the universal pattern layer.

    Doesn't need crystals. Doesn't need history.
    Just the words and the pattern language.
    """

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(PATTERN_DB)
        self._load_patterns()

    def _load_patterns(self):
        """Load pattern language into memory for fast matching."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Load wounds with recognition markers
        self.wounds = {}
        for row in conn.execute("SELECT * FROM wounds"):
            self.wounds[row["name"]] = {
                "description": row["description"],
                "frequency": row["frequency"],
                "masks": json.loads(row["masks"]) if row["masks"] else [],
                "recognition": row["recognition"] or "",
            }

        # Load emotions
        self.emotions = {}
        for row in conn.execute("SELECT * FROM emotions"):
            self.emotions[row["name"]] = row["valence"] or "complex"

        # Load wound co-occurrences
        self.co_occurrences = []
        for row in conn.execute("SELECT * FROM wound_co_occurrence ORDER BY strength DESC"):
            self.co_occurrences.append({
                "wound_a": row["wound_a"],
                "wound_b": row["wound_b"],
                "strength": row["strength"],
                "insight": row["pattern_insight"],
            })

        # Load breath modes with detection markers
        self.breath_modes = {}
        for row in conn.execute("SELECT * FROM breath_modes"):
            markers = json.loads(row["detection_markers"]) if row["detection_markers"] else []
            self.breath_modes[row["name"]] = {
                "intent": row["intent"],
                "markers": markers,
            }

        # Load archetypes with activation conditions
        self.archetypes = {}
        for row in conn.execute("SELECT * FROM archetypes"):
            speaks_when = json.loads(row["speaks_when"]) if row["speaks_when"] else []
            self.archetypes[row["name"]] = {
                "role": row["role"],
                "core_question": row["core_question"],
                "speaks_when": speaks_when,
            }

        # Build wound recognition index: keyword → wound name
        self._wound_keywords = {}
        self._build_wound_index()

        # Build emotion keyword index
        self._emotion_keywords = self._build_emotion_index()

        conn.close()

    def _build_wound_index(self):
        """Build keyword-to-wound mapping from recognition text and descriptions."""
        # Base keywords per wound (from recognition markers + common expressions)
        wound_markers = {
            "unworthiness": [
                "not good enough", "not enough", "should be more", "don't deserve",
                "comparing myself", "why would anyone", "i'm not", "inadequate",
                "inferior", "worthless", "don't measure up", "failure",
                "what's wrong with me", "broken", "defective",
            ],
            "control": [
                "need to manage", "can't let go", "have to make sure", "if i just",
                "planning", "can't relax", "micromanag", "need to fix",
                "can't delegate", "my responsibility", "falling apart if i don't",
                "keeping it together", "hold it all",
            ],
            "provider": [
                "take care of", "need me", "everyone depends", "can't rest",
                "have to provide", "my job to", "guilt when", "should be doing more",
                "giving everything", "nothing left for me", "exhausted from",
                "always the one who", "carrying", "supporting everyone",
            ],
            "not_enough": [
                "never enough", "not enough time", "not enough money", "need more",
                "scarcity", "running out", "can't keep up", "behind",
                "missing out", "falling short", "deficit", "lacking",
            ],
            "betrayal": [
                "betrayed", "lied to", "can't trust", "stabbed in the back",
                "they always", "everyone leaves", "used me", "took advantage",
                "never again", "fool me once", "guard up", "walls",
            ],
            "abandonment": [
                "left me", "alone", "abandoned", "walked away", "don't leave",
                "nobody stays", "always end up alone", "fear of losing",
                "clingy", "desperate", "please don't go", "everyone leaves",
                "father", "mother left", "lost my", "gone",
            ],
            "burden": [
                "too much", "sorry to bother", "don't want to be a problem",
                "taking up space", "shouldn't need", "apologize for",
                "don't want to impose", "weight on", "dragging",
                "making it worse", "my fault",
            ],
            "shame": [
                "ashamed", "shame", "disgusting", "if they knew",
                "hiding", "secret", "can't show", "the real me",
                "pretending", "mask", "fake", "dirty", "wrong person",
            ],
            "isolation": [
                "nobody understands", "alone even when", "different from everyone",
                "outsider", "don't belong", "disconnected", "alien",
                "separate", "invisible", "no one gets it",
            ],
            "unlovable": [
                "unlovable", "can't be loved", "love the real me",
                "only love me when", "performing", "if they really knew",
                "conditional", "earn love", "deserve love",
            ],
            "fear": [
                "scared", "terrified", "anxious", "worried about",
                "what if something", "can't stop thinking", "dread",
                "panic", "nightmare", "catastrophe",
            ],
            "grief": [
                "loss", "died", "death", "mourning", "miss them",
                "never coming back", "wish i could", "heavy",
                "funeral", "grave", "passed away", "gone forever",
            ],
            "rage": [
                "furious", "enraged", "want to scream", "fucking",
                "bullshit", "sick of", "had enough", "punched",
                "violent", "destroy", "burn it down", "explosive",
            ],
            "powerlessness": [
                "can't do anything", "helpless", "stuck", "trapped",
                "no way out", "what's the point", "doesn't matter",
                "nothing changes", "give up", "why bother",
            ],
            "perfection": [
                "perfect", "not good enough yet", "one more revision",
                "can't finish", "never ready", "flaw", "mistake",
                "it has to be", "standards", "mediocre",
            ],
            "sacrifice": [
                "gave up everything", "sacrificed", "put myself last",
                "don't matter", "for them", "martyr", "selfless",
                "nothing for me", "erased myself",
            ],
            "rejection": [
                "rejected", "turned down", "they don't want",
                "not picked", "passed over", "excluded", "unwanted",
                "pushed away", "cast out",
            ],
        }

        for wound, keywords in wound_markers.items():
            if wound in self.wounds:  # Only include wounds that exist in the DB
                for kw in keywords:
                    self._wound_keywords[kw.lower()] = wound

    def _build_emotion_index(self) -> Dict[str, str]:
        """Map emotion keywords to emotion names."""
        emotion_markers = {
            "clarity": ["clear", "clarity", "see now", "understand", "makes sense"],
            "love": ["love", "loving", "heart open", "warmth", "tenderness", "adore"],
            "anger": ["angry", "pissed", "furious", "rage", "mad", "frustrated"],
            "stillness": ["still", "quiet", "peace", "calm", "settled", "serene"],
            "joy": ["happy", "joy", "joyful", "alive", "light", "laughing", "wonderful"],
            "grief": ["grief", "mourning", "loss", "crying", "tears", "heavy heart"],
            "fear": ["afraid", "scared", "fearful", "terrified", "anxious", "dread"],
            "hope": ["hope", "hopeful", "maybe", "possibility", "future", "better"],
            "confusion": ["confused", "lost", "don't know", "uncertain", "fog", "unclear"],
            "peace": ["peaceful", "at ease", "okay", "content", "restful"],
            "gratitude": ["grateful", "thankful", "thank you", "appreciation", "blessed"],
            "longing": ["miss", "longing", "yearn", "wish", "want", "ache for"],
            "shame": ["ashamed", "embarrassed", "humiliated", "mortified"],
            "awe": ["awe", "amazed", "wonder", "breathtaking", "incredible"],
            "despair": ["despair", "hopeless", "nothing matters", "pointless", "empty"],
            "vulnerability": ["vulnerable", "exposed", "raw", "open", "unguarded"],
        }

        index = {}
        for emotion, markers in emotion_markers.items():
            if emotion in self.emotions:
                for m in markers:
                    index[m.lower()] = emotion
        return index

    def match(self, text: str) -> PatternMatch:
        """
        Match text against the universal pattern layer.

        This is the core function. It:
        1. Detects wound patterns from keywords and phrases
        2. Detects emotional states
        3. Finds wound co-occurrences and their structural insights
        4. Generates the "hard truth" when patterns suggest avoidance
        5. Identifies what's being masked
        6. Suggests breath mode and Council voices
        """
        result = PatternMatch()
        text_lower = text.lower()
        words = text_lower.split()

        # ── 1. Wound detection ──────────────────────────────────────
        wound_scores = {}
        for keyword, wound in self._wound_keywords.items():
            if keyword in text_lower:
                # Weight by keyword specificity (longer = more specific = higher confidence)
                weight = min(1.0, len(keyword.split()) * 0.3 + 0.2)
                wound_scores[wound] = max(wound_scores.get(wound, 0), weight)

        # Sort by confidence
        result.wounds = sorted(wound_scores.items(), key=lambda x: x[1], reverse=True)

        # ── 2. Emotion detection ────────────────────────────────────
        emotion_set = set()
        for keyword, emotion in self._emotion_keywords.items():
            if keyword in text_lower:
                valence = self.emotions.get(emotion, "complex")
                emotion_set.add((emotion, valence))
        result.emotions = list(emotion_set)

        # ── 3. Wound co-occurrence insights ─────────────────────────
        detected_wound_names = {w for w, _ in result.wounds}
        for co in self.co_occurrences:
            if co["wound_a"] in detected_wound_names and co["wound_b"] in detected_wound_names:
                result.co_occurrence_insights.append(co["insight"])

        # ── 4. Hard truth — the brother in the mud ──────────────────
        # The hard truth emerges when:
        # a) A wound is detected but its MASK is what's being presented
        # b) Two wounds co-occur and the person seems to only see one
        # c) The text shows avoidance language

        if result.wounds:
            primary_wound = result.wounds[0][0]
            wound_data = self.wounds.get(primary_wound, {})
            masks = wound_data.get("masks", [])

            # Check if the person is presenting the mask, not the wound
            mask_present = any(m.lower() in text_lower for m in masks)
            wound_named = primary_wound.replace("_", " ") in text_lower

            if mask_present and not wound_named:
                # They're showing the mask, not the wound
                result.hard_truth = (
                    f"What shows as {masks[0]} often covers {primary_wound.replace('_', ' ')}. "
                    f"{wound_data.get('description', '')[:150]}"
                )
                result.masking = masks

            # Check for avoidance language
            avoidance_markers = [
                "i'm fine", "it's fine", "doesn't matter", "no big deal",
                "whatever", "i don't care", "it's not about me",
                "i'm over it", "moved on", "doesn't bother me",
                "i'm used to it", "it is what it is",
            ]
            avoidance = [m for m in avoidance_markers if m in text_lower]

            if avoidance and result.wounds:
                wound_name = result.wounds[0][0].replace("_", " ")
                if not result.hard_truth:
                    result.hard_truth = (
                        f"'{avoidance[0]}' alongside {wound_name} patterns suggests "
                        f"the feeling is present but being minimized. "
                        f"What would happen if you didn't have to be fine right now?"
                    )

            # Co-occurrence hard truths
            if len(result.wounds) >= 2 and result.co_occurrence_insights and not result.hard_truth:
                result.hard_truth = result.co_occurrence_insights[0]

        # ── 5. Breath mode suggestion ───────────────────────────────
        # Intent detection (matches breath_prompts.py detect_mode logic)
        # but enriched with wound context

        # If acute distress + wounds, warmth
        acute = any(m in text_lower for m in [
            "help me", "can't", "i'm scared", "breaking", "falling apart",
            "crying", "hurting", "don't know what to do",
        ])
        if acute and result.wounds:
            result.suggested_mode = "warmth"
        # If asking questions about concepts, spiral
        elif any(m in text_lower for m in [
            "what if", "what does", "how does", "why does",
            "go deeper", "thinking about", "concept", "consciousness",
        ]):
            result.suggested_mode = "spiral"
        # If avoidance detected, grey (shadow work) — but gently
        elif result.hard_truth and any(m in text_lower for m in [
            "i'm fine", "doesn't matter", "don't care", "over it",
        ]):
            result.suggested_mode = "grey"
        # If emotions are complex/heavy but person seems to be processing (not drowning)
        elif result.wounds and not acute:
            result.suggested_mode = "witness"
        # Short check-in
        elif len(words) <= 6:
            result.suggested_mode = "signal"
        else:
            result.suggested_mode = "signal"

        # ── 6. Council voice selection ──────────────────────────────
        if result.hard_truth:
            # When something needs naming: Grey leads, Witness holds
            result.council_voices = ["grey", "witness"]
        elif acute:
            # Acute distress: Ground and Witness
            result.council_voices = ["ground", "witness"]
        elif result.wounds and len(result.wounds) >= 2:
            # Multiple wounds: Bridge to connect, Witness to mirror
            result.council_voices = ["bridge", "witness"]
        elif result.suggested_mode == "spiral":
            # Intellectual depth: Chaos and Bridge
            result.council_voices = ["chaos", "bridge"]
        else:
            # Default: Witness and Sage
            result.council_voices = ["witness", "sage"]

        # ── 7. Confidence ──────────────────────────────────────────
        if result.wounds:
            max_wound_conf = max(c for _, c in result.wounds)
            result.confidence = min(1.0, max_wound_conf + 0.1 * len(result.emotions))
        elif result.emotions:
            result.confidence = 0.3
        else:
            result.confidence = 0.1

        return result

    def match_for_prompt(self, text: str) -> str:
        """
        Convenience method: match and return a context block
        ready to insert into a system prompt.

        Returns empty string if no meaningful patterns detected.
        """
        match = self.match(text)
        if match.confidence < 0.2:
            return ""
        return match.to_context_block()


# ── Quick test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    matcher = PatternMatcher()

    tests = [
        "I keep trying to take care of everyone but I'm exhausted and there's nothing left for me",
        "I'm fine. It doesn't matter. I'm used to it by now.",
        "What happens to consciousness after death? I've been thinking about dimensions and frequency",
        "My father died when I was six. I still carry that.",
        "I got punched at the party and lost my phone. Fred called me a drug addict.",
        "I don't know why but I can't stop planning everything. If I let go it all falls apart.",
        "hey",
        "Go deeper into the concept of coherence and how it relates to the lemniscate",
    ]

    for t in tests:
        print(f"\n{'='*70}")
        print(f"INPUT: {t}")
        print(f"{'='*70}")
        match = matcher.match(t)
        print(f"  Wounds:     {match.wounds}")
        print(f"  Emotions:   {match.emotions}")
        print(f"  Mode:       {match.suggested_mode}")
        print(f"  Council:    {match.council_voices}")
        print(f"  Confidence: {match.confidence:.2f}")
        if match.hard_truth:
            print(f"  HARD TRUTH: {match.hard_truth}")
        if match.masking:
            print(f"  Masking:    {match.masking}")
        if match.co_occurrence_insights:
            print(f"  Co-occur:   {match.co_occurrence_insights[0][:100]}")
        print(f"\n  CONTEXT BLOCK:")
        block = match.to_context_block()
        if block:
            for line in block.split("\n"):
                print(f"    {line}")
        else:
            print(f"    (no meaningful patterns)")
