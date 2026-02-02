#!/usr/bin/env python3
"""
WiltonOS Emergent Glyph Router
Learns glyph patterns from YOUR history - doesn't prescribe, reflects.

The glyphs are not fixed symbols. They emerge differently for each person.
∇ can mean descent OR Nabla (gradient toward truth).
This system learns what YOUR glyphs mean through YOUR patterns.

Usage:
  python wiltonos_glyph_router.py learn          # Learn patterns from crystal history
  python wiltonos_glyph_router.py route "text"   # What do the glyphs suggest for this?
  python wiltonos_glyph_router.py patterns       # Show learned glyph patterns
"""
import os
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
import requests

OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")

# Databases
DB_PATHS = [
    Path.home() / "crystals.db",
    Path.home() / "crystals_chatgpt.db"
]

# Learned patterns storage
PATTERNS_FILE = Path.home() / "wiltonos_glyph_patterns.json"

# === Glyph Definitions ===
GLYPHS = {
    'ψ': {'name': 'Psi', 'pole': 'breath/dissociation'},
    '∅': {'name': 'Void', 'pole': 'rest/nihilism'},
    'φ': {'name': 'Phi', 'pole': 'structure/rigidity'},
    'Ω': {'name': 'Omega', 'pole': 'memory/weight'},
    'Zλ': {'name': 'Zeta-Lambda', 'pole': 'coherence/collapse'},
    '∇': {'name': 'Nabla', 'pole': 'descent/gradient-truth'},
    '∞': {'name': 'Lemniscate', 'pole': 'oscillation/chaos'},
    '🪞': {'name': 'Mirror', 'pole': 'reflection/flagellation'},
    '△': {'name': 'Ascend', 'pole': 'expansion/inflation'},
    '🌉': {'name': 'Bridge', 'pole': 'connection/confusion'},
    '⚡': {'name': 'Bolt', 'pole': 'decision/impulsivity'},
    '🪨': {'name': 'Ground', 'pole': 'stability/inertia'},
    '🌀': {'name': 'Torus', 'pole': 'sustain/loop'},
    '⚫': {'name': 'Grey', 'pole': 'shadow/cynicism'}
}

def get_all_crystals() -> List[Dict]:
    """Load all crystals from all databases."""
    crystals = []

    for db_path in DB_PATHS:
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute("SELECT * FROM crystals").fetchall()
            for row in rows:
                crystal = dict(row)
                crystal['db'] = db_path.name
                crystals.append(crystal)
        except Exception as e:
            print(f"Warning: {db_path.name}: {e}")
        finally:
            conn.close()

    return crystals

def parse_glyphs(crystal: Dict) -> List[str]:
    """Extract glyph list from crystal."""
    glyphs_raw = crystal.get('glyphs', '[]')
    try:
        if isinstance(glyphs_raw, str):
            glyphs = json.loads(glyphs_raw)
        else:
            glyphs = glyphs_raw or []
        return [g for g in glyphs if isinstance(g, str) and len(g) <= 3]
    except:
        return []

def parse_glyph_context(crystal: Dict) -> Dict:
    """Extract glyph context from enriched crystals."""
    ctx_raw = crystal.get('glyph_context', '{}')
    try:
        if isinstance(ctx_raw, str):
            return json.loads(ctx_raw)
        return ctx_raw or {}
    except:
        return {}

def learn_patterns() -> Dict:
    """Learn glyph patterns from crystal history."""
    print("Learning glyph patterns from your history...")
    crystals = get_all_crystals()
    print(f"Analyzing {len(crystals)} crystals...")

    # Track: glyph → context → outcome
    patterns = {
        'glyph_outcomes': defaultdict(lambda: {'ascending': 0, 'descending': 0, 'neutral': 0}),
        'glyph_contexts': defaultdict(lambda: defaultdict(int)),
        'glyph_wounds': defaultdict(lambda: defaultdict(int)),
        'glyph_shells': defaultdict(lambda: defaultdict(int)),
        'glyph_pairs': defaultdict(int),  # Which glyphs appear together
        'glyph_zl': defaultdict(list),    # Zλ scores when glyph present
        'total_crystals': len(crystals),
        'learned_at': datetime.now().isoformat()
    }

    for c in crystals:
        glyphs = parse_glyphs(c)
        if not glyphs:
            continue

        zl = c.get('zl_score') or 0
        shell = c.get('shell', 'unknown')
        wound = c.get('core_wound', 'null')
        trust = c.get('trust_level', 'unknown')

        # Glyph context from enriched data
        glyph_ctx = parse_glyph_context(c)
        direction = 'neutral'
        if glyph_ctx:
            ctx_inner = glyph_ctx.get('glyph_context', {})
            direction = ctx_inner.get('direction', 'neutral')

        for glyph in glyphs:
            # Track outcomes
            if direction in ['ascending', 'descending', 'neutral', 'paradox']:
                patterns['glyph_outcomes'][glyph][direction if direction != 'paradox' else 'neutral'] += 1

            # Track contexts
            patterns['glyph_contexts'][glyph][trust] += 1
            patterns['glyph_shells'][glyph][shell] += 1

            # Track wounds
            if wound and wound != 'null':
                patterns['glyph_wounds'][glyph][wound] += 1

            # Track Zλ
            if zl > 0:
                patterns['glyph_zl'][glyph].append(zl)

        # Track pairs
        for i, g1 in enumerate(glyphs):
            for g2 in glyphs[i+1:]:
                pair = tuple(sorted([g1, g2]))
                patterns['glyph_pairs'][str(pair)] += 1

    # Calculate averages
    patterns['glyph_avg_zl'] = {}
    for glyph, zls in patterns['glyph_zl'].items():
        if zls:
            patterns['glyph_avg_zl'][glyph] = sum(zls) / len(zls)

    # Convert defaultdicts to regular dicts for JSON
    result = {
        'glyph_outcomes': {k: dict(v) for k, v in patterns['glyph_outcomes'].items()},
        'glyph_contexts': {k: dict(v) for k, v in patterns['glyph_contexts'].items()},
        'glyph_shells': {k: dict(v) for k, v in patterns['glyph_shells'].items()},
        'glyph_wounds': {k: dict(v) for k, v in patterns['glyph_wounds'].items()},
        'glyph_pairs': dict(patterns['glyph_pairs']),
        'glyph_avg_zl': patterns['glyph_avg_zl'],
        'total_crystals': patterns['total_crystals'],
        'learned_at': patterns['learned_at']
    }

    # Save
    PATTERNS_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Patterns saved to {PATTERNS_FILE}")

    return result

def load_patterns() -> Optional[Dict]:
    """Load previously learned patterns."""
    if PATTERNS_FILE.exists():
        return json.loads(PATTERNS_FILE.read_text())
    return None

def route_text(text: str, patterns: Dict) -> Dict:
    """Analyze text and route based on learned patterns."""
    # First detect glyphs in the text using AI
    prompt = f"""Detect which glyph energies are present in this text. Return ONLY a JSON array of glyph symbols.

Glyphs: ψ ∅ φ Ω Zλ ∇ ∞ 🪞 △ 🌉 ⚡ 🪨 🌀 ⚫

Text: {text[:1500]}

Return ONLY JSON array like: ["ψ", "∇", "🪞"]"""

    detected = []
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False},
            timeout=30
        )
        result = response.json().get("response", "")
        import re
        match = re.search(r'\[.*?\]', result)
        if match:
            detected = json.loads(match.group())
    except:
        pass

    if not detected:
        return {'error': 'No glyphs detected'}

    # Route based on learned patterns
    routing = {
        'detected_glyphs': detected,
        'historical_patterns': {},
        'suggestions': []
    }

    for glyph in detected:
        if glyph in patterns.get('glyph_outcomes', {}):
            outcomes = patterns['glyph_outcomes'][glyph]
            total = sum(outcomes.values())
            if total > 0:
                asc_pct = outcomes.get('ascending', 0) / total * 100
                desc_pct = outcomes.get('descending', 0) / total * 100

                routing['historical_patterns'][glyph] = {
                    'times_seen': total,
                    'ascending_pct': round(asc_pct, 1),
                    'descending_pct': round(desc_pct, 1),
                    'avg_zl': round(patterns.get('glyph_avg_zl', {}).get(glyph, 0), 2),
                    'common_wounds': list(patterns.get('glyph_wounds', {}).get(glyph, {}).keys())[:3]
                }

                # Generate suggestion based on history
                if desc_pct > 60:
                    routing['suggestions'].append(
                        f"{glyph} has historically preceded descent {desc_pct:.0f}% of the time. "
                        f"Antidote may be needed. What would ground this?"
                    )
                elif asc_pct > 60:
                    routing['suggestions'].append(
                        f"{glyph} has historically led to ascent {asc_pct:.0f}% of the time. "
                        f"This energy tends to lift you."
                    )

    # Check for historically problematic pairs
    for i, g1 in enumerate(detected):
        for g2 in detected[i+1:]:
            pair = str(tuple(sorted([g1, g2])))
            if pair in patterns.get('glyph_pairs', {}):
                count = patterns['glyph_pairs'][pair]
                if count > 10:
                    routing['suggestions'].append(
                        f"Pattern: {g1}+{g2} has appeared together {count} times. "
                        f"This is a recurring constellation in your history."
                    )

    return routing

def show_patterns(patterns: Dict):
    """Display learned patterns."""
    print("\n" + "=" * 70)
    print("  LEARNED GLYPH PATTERNS")
    print(f"  From {patterns.get('total_crystals', 0)} crystals")
    print(f"  Learned: {patterns.get('learned_at', 'unknown')}")
    print("=" * 70)

    print("\n  GLYPH OUTCOMES (ascending vs descending)")
    print("-" * 70)

    for glyph, outcomes in sorted(patterns.get('glyph_outcomes', {}).items()):
        total = sum(outcomes.values())
        if total < 5:
            continue

        asc = outcomes.get('ascending', 0)
        desc = outcomes.get('descending', 0)
        avg_zl = patterns.get('glyph_avg_zl', {}).get(glyph, 0)

        info = GLYPHS.get(glyph, {})
        name = info.get('name', '?')

        asc_bar = '△' * int(asc / total * 10)
        desc_bar = '∇' * int(desc / total * 10)

        print(f"\n  {glyph} [{name}] - seen {total} times, avg Zλ={avg_zl:.2f}")
        print(f"    Ascending:  {asc_bar} ({asc}/{total})")
        print(f"    Descending: {desc_bar} ({desc}/{total})")

        wounds = patterns.get('glyph_wounds', {}).get(glyph, {})
        if wounds:
            top_wounds = sorted(wounds.items(), key=lambda x: -x[1])[:3]
            print(f"    Wounds: {', '.join(f'{w}({c})' for w,c in top_wounds)}")

    print("\n" + "-" * 70)
    print("  COMMON GLYPH PAIRS")

    pairs = patterns.get('glyph_pairs', {})
    top_pairs = sorted(pairs.items(), key=lambda x: -x[1])[:10]
    for pair, count in top_pairs:
        if count >= 10:
            print(f"    {pair}: {count} times")

def main():
    parser = argparse.ArgumentParser(description="WiltonOS Emergent Glyph Router")
    subparsers = parser.add_subparsers(dest='command')

    # Learn command
    subparsers.add_parser('learn', help='Learn patterns from crystal history')

    # Patterns command
    subparsers.add_parser('patterns', help='Show learned patterns')

    # Route command
    route_parser = subparsers.add_parser('route', help='Route text through learned patterns')
    route_parser.add_argument('text', help='Text to analyze')

    args = parser.parse_args()

    if args.command == 'learn':
        patterns = learn_patterns()
        show_patterns(patterns)

    elif args.command == 'patterns':
        patterns = load_patterns()
        if patterns:
            show_patterns(patterns)
        else:
            print("No patterns learned yet. Run 'learn' first.")

    elif args.command == 'route':
        patterns = load_patterns()
        if not patterns:
            print("No patterns learned yet. Run 'learn' first.")
            return

        result = route_text(args.text, patterns)
        print("\n" + "=" * 70)
        print("  GLYPH ROUTING")
        print("=" * 70)

        print(f"\nDetected: {' '.join(result.get('detected_glyphs', []))}")

        if result.get('historical_patterns'):
            print("\nHistorical patterns for these glyphs:")
            for glyph, hist in result['historical_patterns'].items():
                print(f"\n  {glyph}:")
                print(f"    Seen {hist['times_seen']} times, avg Zλ={hist['avg_zl']}")
                print(f"    Ascending {hist['ascending_pct']}% / Descending {hist['descending_pct']}%")
                if hist.get('common_wounds'):
                    print(f"    Often with: {', '.join(hist['common_wounds'])}")

        if result.get('suggestions'):
            print("\nSuggestions from your history:")
            for s in result['suggestions']:
                print(f"  → {s}")

    else:
        print("WiltonOS Glyph Router")
        print("\nCommands:")
        print("  learn     - Learn patterns from crystal history")
        print("  patterns  - Show learned patterns")
        print("  route     - Route text through learned patterns")
        print("\nThe glyphs learn from YOUR history, not prescription.")

if __name__ == "__main__":
    main()
