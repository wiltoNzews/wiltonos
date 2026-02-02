#!/usr/bin/env python3
"""
Field Vocabulary — Shared wound/emotion index for mesh agents.
==============================================================
Wraps PatternMatcher's internal keyword indexes into a stable API
that BridgeAgent and TorusAgent can call instead of hardcoded maps.

If no PatternMatcher is provided, all methods return empty results
and agents fall back to their own hardcoded maps.

Usage:
    from field_vocab import FieldVocabulary
    vocab = FieldVocabulary(pattern_matcher)
    wounds = vocab.scan_wounds("I keep taking care of everyone")
    # [("provider", 0.8), ("unworthiness", 0.5)]
    emotions = vocab.scan_emotions("I'm scared and angry")
    # [("fear", "negative"), ("anger", "negative")]
"""

from typing import List, Tuple, Optional, Dict, Set, Any


class FieldVocabulary:
    """
    Shared vocabulary index for mesh agents.

    Wraps PatternMatcher's _wound_keywords and _emotion_keywords dicts
    plus the wounds metadata (masks, recognition, co_occurrences).

    Thread-safe: reads from dicts built at PatternMatcher init time.
    """

    def __init__(self, pattern_matcher: Optional[Any] = None):
        self._available = pattern_matcher is not None

        if self._available:
            # Direct references (these are read-only after PatternMatcher init)
            self._wound_keywords: Dict[str, str] = pattern_matcher._wound_keywords
            self._emotion_keywords: Dict[str, str] = pattern_matcher._emotion_keywords
            self._wounds_meta: Dict[str, dict] = pattern_matcher.wounds
            self._emotions_meta: Dict[str, str] = pattern_matcher.emotions
            self._co_occurrences: list = pattern_matcher.co_occurrences
        else:
            self._wound_keywords = {}
            self._emotion_keywords = {}
            self._wounds_meta = {}
            self._emotions_meta = {}
            self._co_occurrences = []

    @property
    def available(self) -> bool:
        """True if vocabulary data is loaded."""
        return self._available

    def scan_wounds(self, text: str) -> List[Tuple[str, float]]:
        """
        Scan text for wound patterns. Returns [(wound_name, confidence), ...]
        sorted by confidence descending.

        Confidence = min(1.0, word_count_in_keyword * 0.3 + 0.2)
        matching PatternMatcher.match() logic.
        """
        if not self._available:
            return []

        text_lower = text.lower()
        wound_scores: Dict[str, float] = {}
        for keyword, wound in self._wound_keywords.items():
            if keyword in text_lower:
                weight = min(1.0, len(keyword.split()) * 0.3 + 0.2)
                wound_scores[wound] = max(wound_scores.get(wound, 0), weight)

        return sorted(wound_scores.items(), key=lambda x: x[1], reverse=True)

    def scan_emotions(self, text: str) -> List[Tuple[str, str]]:
        """
        Scan text for emotions. Returns [(emotion_name, valence), ...]
        where valence is "positive", "negative", or "complex".
        """
        if not self._available:
            return []

        text_lower = text.lower()
        found: Set[Tuple[str, str]] = set()
        for keyword, emotion in self._emotion_keywords.items():
            if keyword in text_lower:
                valence = self._emotions_meta.get(emotion, "complex")
                found.add((emotion, valence))
        return list(found)

    def get_wound_names(self, text: str) -> List[str]:
        """Convenience: just wound names, no scores."""
        return [w for w, _ in self.scan_wounds(text)]

    def get_emotion_names(self, text: str) -> List[str]:
        """Convenience: just emotion names, no valence."""
        return [e for e, _ in self.scan_emotions(text)]

    def get_co_occurrences(self, wound_names: Set[str]) -> List[Dict]:
        """
        Given a set of detected wound names, return matching co-occurrence
        insights where both wound_a and wound_b are in the set.
        """
        if not self._available:
            return []
        return [
            co for co in self._co_occurrences
            if co["wound_a"] in wound_names and co["wound_b"] in wound_names
        ]

    def get_wound_masks(self, wound_name: str) -> List[str]:
        """Get the masks list for a wound (what it disguises as)."""
        meta = self._wounds_meta.get(wound_name, {})
        return meta.get("masks", [])

    def wound_count(self) -> int:
        """Number of wound types in the vocabulary."""
        return len(self._wounds_meta)

    def emotion_count(self) -> int:
        """Number of emotion types in the vocabulary."""
        return len(self._emotions_meta)
