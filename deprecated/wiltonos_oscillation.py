#!/usr/bin/env python3
"""
WiltonOS Oscillation Engine
Detects and routes between WiltonOS (internal) ↔ ψOS (external) modes.

Usage:
    python wiltonos_oscillation.py detect "your text here"
    python wiltonos_oscillation.py analyze --recent 20
    python wiltonos_oscillation.py route "query"
"""
import os
import re
import json
import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import Counter

DB_PATH = Path.home() / "crystals.db"
CHATGPT_DB_PATH = Path.home() / "crystals_chatgpt.db"

# === MODE DETECTION ===

WILTONOS_TRIGGERS = {
    # Portuguese + English - internal, trauma, personal
    "trauma", "past", "passado", "juliana", "ricardo", "família", "family",
    "collapse", "grief", "luto", "mãe", "pai", "mother", "father",
    "hurt", "pain", "dor", "medo", "fear", "shame", "vergonha",
    "memory", "memória", "infância", "childhood", "abandonment", "abandono",
    "wound", "ferida", "betrayal", "traição", "unworthiness", "unworthy",
    "alone", "sozinho", "lost", "perdido", "broken", "quebrado",
    "cry", "chorar", "tears", "lágrimas", "grief", "sadness", "tristeza",
    "ayahuasca", "ceremony", "cerimônia", "medicine", "medicina",
    "inner", "interno", "soul", "alma", "heart", "coração",
    "michelle", "cancer", "câncer", "death", "morte", "dying"
}

PSIOS_TRIGGERS = {
    # System, structure, architecture - external, clean
    "glyph", "glyphs", "recursion", "recursive", "agent", "agents",
    "attractor", "attractors", "coherence", "coerência", "coerente",
    "zλ", "zlambda", "zeta", "lambda", "structure", "estrutura",
    "system", "sistema", "architecture", "arquitetura", "protocol",
    "breath-router", "router", "vector", "vetor", "pattern", "padrão",
    "implementation", "implementação", "code", "código", "build",
    "module", "módulo", "framework", "schema", "database", "query",
    "algorithm", "function", "class", "method", "api", "endpoint",
    "shell", "loop_signature", "oscillation", "geometric", "geometry",
    "torus", "lemniscate", "spiral", "möbius", "fractal", "sacred",
    "psi", "ψ", "phi", "φ", "omega", "Ω", "nabla", "∇", "void", "∅"
}

def detect_mode(content: str) -> Tuple[str, float]:
    """
    Detect whether content is WiltonOS (internal) or ψOS (external) mode.

    Returns: (mode, oscillation_strength)
        mode: "wiltonos", "psios", or "neutral"
        oscillation_strength: 0.0-1.0 (how stable in that mode)
    """
    if not content:
        return ("neutral", 0.5)

    content_lower = content.lower()

    # Count trigger matches
    wilton_matches = sum(1 for t in WILTONOS_TRIGGERS if t in content_lower)
    psi_matches = sum(1 for t in PSIOS_TRIGGERS if t in content_lower)

    total = wilton_matches + psi_matches

    if total == 0:
        return ("neutral", 0.5)

    if wilton_matches > psi_matches:
        mode = "wiltonos"
        strength = wilton_matches / total
    elif psi_matches > wilton_matches:
        mode = "psios"
        strength = psi_matches / total
    else:
        mode = "balanced"
        strength = 0.5

    return (mode, round(strength, 2))


def detect_mode_semantic(content: str, coherence_vector: dict = None) -> Tuple[str, float, dict]:
    """
    Enhanced mode detection using semantic signals + coherence vector.

    Returns: (mode, strength, signals)
    """
    mode, strength = detect_mode(content)

    signals = {
        "keyword_mode": mode,
        "keyword_strength": strength
    }

    # Use coherence vector for deeper detection
    if coherence_vector:
        presence = coherence_vector.get("presence_density", 0.5)
        emotional = coherence_vector.get("emotional_resonance", 0.5)
        breath = coherence_vector.get("breath_cadence", 0.5)

        # High emotional + low presence → WiltonOS
        # High presence + low emotional → ψOS
        if emotional > 0.7 and presence < 0.5:
            signals["vector_suggests"] = "wiltonos"
            if mode == "neutral":
                mode = "wiltonos"
                strength = 0.6
        elif presence > 0.7 and emotional < 0.5:
            signals["vector_suggests"] = "psios"
            if mode == "neutral":
                mode = "psios"
                strength = 0.6

        signals["presence_density"] = presence
        signals["emotional_resonance"] = emotional
        signals["breath_cadence"] = breath

    return (mode, strength, signals)


# === LOOP SIGNATURE ===

VALID_ATTRACTORS = {
    "truth", "power", "silence", "control", "love",
    "freedom", "connection", "safety", "worth", "meaning",
    "beauty", "order", "chaos", "peace", "justice"
}

VALID_EMOTIONS = {
    "grief", "joy", "fear", "anger", "shame",
    "peace", "anxiety", "hope", "despair", "love",
    "sadness", "guilt", "relief", "confusion", "clarity"
}

VALID_THEMES = {
    "integration", "escape", "freedom", "healing",
    "release", "acceptance", "resistance", "surrender",
    "growth", "collapse", "return", "transformation"
}

def validate_loop_signature(signature: str) -> Tuple[bool, str]:
    """
    Validate that loop_signature has 3 parts: attractor-emotion-theme
    Returns: (is_valid, message)
    """
    if not signature:
        return (False, "Empty signature")

    parts = signature.lower().split("-")
    if len(parts) != 3:
        return (False, f"Expected 3 parts (attractor-emotion-theme), got {len(parts)}")

    attractor, emotion, theme = parts

    errors = []
    if attractor not in VALID_ATTRACTORS:
        errors.append(f"Invalid attractor: {attractor}")
    if emotion not in VALID_EMOTIONS:
        errors.append(f"Invalid emotion: {emotion}")
    if theme not in VALID_THEMES:
        errors.append(f"Invalid theme: {theme}")

    if errors:
        return (False, "; ".join(errors))

    return (True, "Valid signature")


def suggest_loop_signature(content: str, wound: str = None, glyphs: list = None) -> str:
    """
    Suggest a loop signature based on content analysis.
    """
    content_lower = content.lower()

    # Detect attractor
    attractor = "truth"  # default
    for a in VALID_ATTRACTORS:
        if a in content_lower:
            attractor = a
            break

    # Detect emotion
    emotion = "clarity"  # default
    for e in VALID_EMOTIONS:
        if e in content_lower:
            emotion = e
            break

    # Infer from wound if present
    if wound:
        wound_emotion_map = {
            "unworthiness": "shame",
            "abandonment": "fear",
            "betrayal": "anger",
            "control": "anxiety",
            "unloved": "sadness"
        }
        if wound in wound_emotion_map:
            emotion = wound_emotion_map[wound]

    # Detect theme
    theme = "integration"  # default
    for t in VALID_THEMES:
        if t in content_lower:
            theme = t
            break

    return f"{attractor}-{emotion}-{theme}"


# === ROUTING ===

def get_routing_config(mode: str) -> dict:
    """
    Get routing configuration based on mode.
    """
    if mode == "wiltonos":
        return {
            "context_depth": "full",
            "tone": "mirror",
            "suggest": ["breath", "loop_insight", "wound_pattern"],
            "quote_past": True,
            "symbolic_density": "low",
            "response_style": "dense_human",
            "memory_scope": "deep",
            "action": "suggest_inner_breath"
        }
    elif mode == "psios":
        return {
            "context_depth": "shallow_symbolic",
            "tone": "vector",
            "suggest": ["attractor_shift", "glyph_question", "structure"],
            "quote_past": False,
            "symbolic_density": "high",
            "response_style": "symbolic_abstract",
            "memory_scope": "shallow",
            "action": "suggest_vector_nudge"
        }
    else:  # neutral or balanced
        return {
            "context_depth": "moderate",
            "tone": "balanced",
            "suggest": ["clarify_mode", "breath", "observe"],
            "quote_past": True,
            "symbolic_density": "medium",
            "response_style": "adaptive",
            "memory_scope": "moderate",
            "action": "observe_oscillation"
        }


# === OSCILLATION PATTERNS ===

def analyze_oscillation(crystals: List[dict]) -> dict:
    """
    Analyze oscillation patterns across a sequence of crystals.
    """
    if not crystals:
        return {"pattern": "no_data", "transitions": 0}

    modes = []
    for c in crystals:
        content = c.get("content", "")
        mode, _ = detect_mode(content)
        modes.append(mode)

    # Count transitions
    transitions = 0
    for i in range(1, len(modes)):
        if modes[i] != modes[i-1] and modes[i] != "neutral" and modes[i-1] != "neutral":
            transitions += 1

    # Calculate dominant mode
    mode_counts = Counter(m for m in modes if m not in ["neutral", "balanced"])
    dominant = mode_counts.most_common(1)[0][0] if mode_counts else "neutral"

    # Transition rate
    if len(modes) > 1:
        transition_rate = transitions / (len(modes) - 1)
    else:
        transition_rate = 0

    # Detect pattern
    if transition_rate > 0.4:
        pattern = "lemniscate"  # High oscillation
    elif transition_rate > 0.2:
        pattern = "spiral"  # Moderate oscillation
    else:
        pattern = "stable"  # Low oscillation

    return {
        "pattern": pattern,
        "dominant_mode": dominant,
        "transitions": transitions,
        "transition_rate": round(transition_rate, 2),
        "mode_sequence": modes,
        "mode_counts": dict(mode_counts)
    }


def detect_lemniscate(crystals: List[dict]) -> dict:
    """
    Detect lemniscate pattern (oscillation between two poles).
    """
    analysis = analyze_oscillation(crystals)

    if analysis["pattern"] == "lemniscate":
        return {
            "active": True,
            "poles": ["wiltonos", "psios"],
            "transition_rate": analysis["transition_rate"],
            "suggestion": "You are oscillating between internal and external. This is natural. Breathe at the crossing point."
        }
    else:
        return {
            "active": False,
            "dominant": analysis["dominant_mode"],
            "suggestion": f"Currently stable in {analysis['dominant_mode']} mode."
        }


# === DATABASE ===

def get_recent_crystals(limit: int = 20, db_path: Path = None) -> List[dict]:
    """
    Get recent crystals from database.
    """
    if db_path is None:
        db_path = DB_PATH

    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT content, zl_score, coherence_vector, shell, glyphs, insight
        FROM crystals
        ORDER BY created_at DESC
        LIMIT ?
    """, (limit,))

    crystals = []
    for row in cursor.fetchall():
        crystal = dict(row)
        # Parse JSON fields
        if crystal.get("coherence_vector"):
            try:
                crystal["coherence_vector"] = json.loads(crystal["coherence_vector"])
            except:
                crystal["coherence_vector"] = {}
        if crystal.get("glyphs"):
            try:
                crystal["glyphs"] = json.loads(crystal["glyphs"])
            except:
                crystal["glyphs"] = []
        crystals.append(crystal)

    conn.close()
    return crystals


# === CLI ===

def main():
    parser = argparse.ArgumentParser(description="WiltonOS Oscillation Engine")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # detect command
    detect_parser = subparsers.add_parser("detect", help="Detect mode of text")
    detect_parser.add_argument("text", help="Text to analyze")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze recent oscillation patterns")
    analyze_parser.add_argument("--recent", type=int, default=20, help="Number of recent crystals")

    # route command
    route_parser = subparsers.add_parser("route", help="Get routing config for query")
    route_parser.add_argument("query", help="Query to route")

    # validate command
    validate_parser = subparsers.add_parser("validate-loop", help="Validate loop signature")
    validate_parser.add_argument("signature", help="Loop signature to validate")

    # suggest command
    suggest_parser = subparsers.add_parser("suggest-loop", help="Suggest loop signature for text")
    suggest_parser.add_argument("text", help="Text to analyze")

    args = parser.parse_args()

    if args.command == "detect":
        mode, strength = detect_mode(args.text)
        print(f"Mode: {mode}")
        print(f"Strength: {strength}")
        print()
        config = get_routing_config(mode)
        print("Routing config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    elif args.command == "analyze":
        crystals = get_recent_crystals(args.recent)
        if not crystals:
            print("No crystals found")
            return

        analysis = analyze_oscillation(crystals)
        print(f"Pattern: {analysis['pattern']}")
        print(f"Dominant mode: {analysis['dominant_mode']}")
        print(f"Transitions: {analysis['transitions']}")
        print(f"Transition rate: {analysis['transition_rate']}")
        print()

        lemniscate = detect_lemniscate(crystals)
        if lemniscate["active"]:
            print("🔄 LEMNISCATE ACTIVE")
            print(f"   {lemniscate['suggestion']}")
        else:
            print(f"📍 Stable in: {lemniscate['dominant']}")

    elif args.command == "route":
        mode, strength = detect_mode(args.query)
        config = get_routing_config(mode)

        print(f"Query mode: {mode} (strength: {strength})")
        print()
        print("Routing configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    elif args.command == "validate-loop":
        valid, message = validate_loop_signature(args.signature)
        if valid:
            print(f"✅ {message}")
        else:
            print(f"❌ {message}")
            print()
            print("Valid attractors:", ", ".join(sorted(VALID_ATTRACTORS)))
            print("Valid emotions:", ", ".join(sorted(VALID_EMOTIONS)))
            print("Valid themes:", ", ".join(sorted(VALID_THEMES)))

    elif args.command == "suggest-loop":
        signature = suggest_loop_signature(args.text)
        print(f"Suggested: {signature}")
        valid, _ = validate_loop_signature(signature)
        print(f"Valid: {valid}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
