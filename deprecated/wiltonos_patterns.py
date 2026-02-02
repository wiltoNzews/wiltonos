#!/usr/bin/env python3
"""
WiltonOS Pattern Surfacing + Geometry Integration
Implements remaining 4o suggestions:
- Pattern surfacing (glyph dominance, loop signatures)
- Mode switch logging
- Zλ spiral visualization
- Geometry as routing logic

Usage:
    python wiltonos_patterns.py glyphs              # Glyph dominance analysis
    python wiltonos_patterns.py loops               # Loop signature frequency
    python wiltonos_patterns.py spiral              # Zλ spiral over time
    python wiltonos_patterns.py geometry            # Geometry pattern detection
    python wiltonos_patterns.py enrich-modes        # Add mode to existing crystals
    python wiltonos_patterns.py summary             # Full pattern summary
"""
import os
import re
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict

# Import oscillation engine
try:
    from wiltonos_oscillation import (
        detect_mode, get_routing_config, suggest_loop_signature,
        validate_loop_signature, VALID_ATTRACTORS, VALID_EMOTIONS, VALID_THEMES
    )
    HAS_OSCILLATION = True
except ImportError:
    HAS_OSCILLATION = False

DB_PATH = Path.home() / "crystals.db"
CHATGPT_DB_PATH = Path.home() / "crystals_chatgpt.db"

# === Database ===

def get_all_crystals() -> List[Dict]:
    """Load all crystals from both databases."""
    crystals = []

    for db_path in [DB_PATH, CHATGPT_DB_PATH]:
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT id, content, zl_score, coherence_vector, shell, glyphs,
                       trust_level, insight, core_wound, mode, oscillation_strength,
                       loop_signature, created_at, analyzed_at
                FROM crystals
            """)
            for row in cursor.fetchall():
                crystal = dict(row)
                crystal['db'] = db_path.name
                crystals.append(crystal)
        except Exception as e:
            # Some columns might not exist
            cursor.execute("SELECT * FROM crystals")
            for row in cursor.fetchall():
                crystal = dict(row)
                crystal['db'] = db_path.name
                crystals.append(crystal)

        conn.close()

    return crystals


def update_crystal_mode(db_path: Path, crystal_id: int, mode: str, strength: float, loop_sig: str):
    """Update a crystal with mode information."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        UPDATE crystals
        SET mode = ?, oscillation_strength = ?, loop_signature = ?, mode_detected_at = ?
        WHERE id = ?
    """, (mode, strength, loop_sig, datetime.now().isoformat(), crystal_id))

    conn.commit()
    conn.close()


# === Glyph Dominance ===

GLYPH_NAMES = {
    'ψ': 'Psi (Breath)',
    '∅': 'Void',
    'φ': 'Phi (Structure)',
    'Ω': 'Omega (Memory)',
    'Zλ': 'Zeta-Lambda (Coherence)',
    '∇': 'Nabla (Descent/Gradient)',
    '∞': 'Lemniscate (Oscillation)',
    '🪞': 'Mirror',
    '△': 'Ascend',
    '🌉': 'Bridge',
    '⚡': 'Bolt (Decision)',
    '🪨': 'Ground',
    '🌀': 'Torus (Cycle)',
    '⚫': 'Grey (Shadow)',
}

def analyze_glyph_dominance(crystals: List[Dict]) -> Dict:
    """Analyze which glyphs dominate across crystals."""
    glyph_counts = Counter()
    glyph_by_mode = defaultdict(Counter)
    glyph_by_shell = defaultdict(Counter)
    glyph_by_wound = defaultdict(Counter)

    for c in crystals:
        glyphs_raw = c.get('glyphs', '[]')
        try:
            glyphs = json.loads(glyphs_raw) if isinstance(glyphs_raw, str) else glyphs_raw
        except:
            glyphs = []

        if not glyphs:
            continue

        mode = c.get('mode', 'unknown')
        shell = c.get('shell', 'unknown')
        wound = c.get('core_wound', 'unknown')

        for g in glyphs:
            if isinstance(g, str) and len(g) <= 3:
                glyph_counts[g] += 1
                glyph_by_mode[mode][g] += 1
                glyph_by_shell[shell][g] += 1
                if wound and wound != 'null':
                    glyph_by_wound[wound][g] += 1

    return {
        'total_counts': dict(glyph_counts.most_common(20)),
        'by_mode': {k: dict(v.most_common(5)) for k, v in glyph_by_mode.items()},
        'by_shell': {k: dict(v.most_common(5)) for k, v in glyph_by_shell.items()},
        'by_wound': {k: dict(v.most_common(5)) for k, v in glyph_by_wound.items()},
        'dominant': glyph_counts.most_common(1)[0] if glyph_counts else None
    }


# === Loop Signature Frequency ===

def analyze_loop_signatures(crystals: List[Dict]) -> Dict:
    """Analyze loop signature patterns."""
    signatures = Counter()
    attractors = Counter()
    emotions = Counter()
    themes = Counter()

    # From stored loop_signature
    for c in crystals:
        sig = c.get('loop_signature', '')
        if sig and '-' in sig:
            signatures[sig] += 1
            parts = sig.split('-')
            if len(parts) >= 3:
                attractors[parts[0]] += 1
                emotions[parts[1]] += 1
                themes[parts[2]] += 1

    # Generate from content if not stored
    if not signatures and HAS_OSCILLATION:
        for c in crystals[:500]:  # Sample
            content = c.get('content', '')
            if content:
                sig = suggest_loop_signature(content)
                if validate_loop_signature(sig)[0]:
                    signatures[sig] += 1
                    parts = sig.split('-')
                    if len(parts) >= 3:
                        attractors[parts[0]] += 1
                        emotions[parts[1]] += 1
                        themes[parts[2]] += 1

    return {
        'top_signatures': dict(signatures.most_common(10)),
        'top_attractors': dict(attractors.most_common(10)),
        'top_emotions': dict(emotions.most_common(10)),
        'top_themes': dict(themes.most_common(10)),
        'total_unique': len(signatures)
    }


# === Zλ Spiral Visualization ===

def analyze_zl_spiral(crystals: List[Dict], periods: List[int] = [7, 30, 90]) -> Dict:
    """Analyze Zλ trends over time (spiral pattern)."""
    # Sort by timestamp or analyzed_at
    def get_time(c):
        if c.get('timestamp') and c['timestamp'] > 0:
            return c['timestamp']
        if c.get('analyzed_at'):
            try:
                return datetime.fromisoformat(c['analyzed_at'].replace('Z', '')).timestamp()
            except:
                pass
        if c.get('created_at'):
            try:
                return datetime.fromisoformat(c['created_at'].replace('Z', '')).timestamp()
            except:
                pass
        return 0

    sorted_crystals = sorted(crystals, key=get_time, reverse=True)

    now = datetime.now().timestamp()
    results = {}

    for period in periods:
        cutoff = now - (period * 24 * 3600)
        period_crystals = [c for c in sorted_crystals if get_time(c) > cutoff]

        if not period_crystals:
            results[f'{period}d'] = {'avg_zl': 0, 'count': 0, 'trend': 'no_data'}
            continue

        def to_float(v):
            try:
                return float(v) if v else 0.0
            except:
                return 0.0

        zl_values = [to_float(c.get('zl_score', 0)) for c in period_crystals]
        avg_zl = sum(zl_values) / len(zl_values) if zl_values else 0

        # Calculate trend (first half vs second half)
        if len(zl_values) >= 4:
            mid = len(zl_values) // 2
            first_half = sum(zl_values[:mid]) / mid
            second_half = sum(zl_values[mid:]) / (len(zl_values) - mid)

            if second_half > first_half + 0.05:
                trend = '↑ ascending'
            elif second_half < first_half - 0.05:
                trend = '↓ descending'
            else:
                trend = '→ stable'
        else:
            trend = '? insufficient'

        results[f'{period}d'] = {
            'avg_zl': round(avg_zl, 3),
            'count': len(period_crystals),
            'trend': trend,
            'min': round(min(zl_values), 3) if zl_values else 0,
            'max': round(max(zl_values), 3) if zl_values else 0
        }

    # Spiral detection: is Zλ returning to same level but with growth?
    if all(f'{p}d' in results for p in periods):
        short = results[f'{periods[0]}d']['avg_zl']
        long = results[f'{periods[-1]}d']['avg_zl']

        if short > long + 0.1:
            spiral_direction = 'expanding (Zλ growing)'
        elif short < long - 0.1:
            spiral_direction = 'contracting (Zλ dropping)'
        else:
            spiral_direction = 'stable spiral'

        results['spiral'] = spiral_direction

    return results


# === Geometry Pattern Detection ===

def detect_geometry_patterns(crystals: List[Dict]) -> Dict:
    """Detect geometric patterns in crystal data."""
    patterns = {
        'point': 0,      # Single crystals (always exists)
        'line': 0,       # Sequential coherence
        'spiral': 0,     # Recurring topic with delta
        'lemniscate': 0, # Oscillation between modes
        'torus': 0,      # Complete cycles
        'mobius': 0      # Self-other recursion
    }

    # Count basic patterns
    patterns['point'] = len(crystals)

    # Line: sequences of similar Zλ
    prev_zl = None
    line_count = 0
    for c in crystals[:100]:
        zl = c.get('zl_score', 0) or 0
        if prev_zl and abs(zl - prev_zl) < 0.1:
            line_count += 1
        prev_zl = zl
    patterns['line'] = line_count

    # Lemniscate: mode oscillations
    modes = [c.get('mode', 'neutral') for c in crystals[:50]]
    transitions = sum(1 for i in range(1, len(modes)) if modes[i] != modes[i-1])
    if transitions > len(modes) * 0.3:
        patterns['lemniscate'] = transitions

    # Spiral: same theme recurring with Zλ change
    themes = defaultdict(list)
    for c in crystals[:200]:
        content = c.get('content', '')[:100].lower()
        zl = c.get('zl_score', 0) or 0
        for word in ['truth', 'love', 'fear', 'control', 'family', 'work']:
            if word in content:
                themes[word].append(zl)

    for theme, zls in themes.items():
        if len(zls) >= 3:
            # Check if Zλ is changing across occurrences
            if len(set(round(z, 1) for z in zls)) > 1:
                patterns['spiral'] += 1

    # Torus: complete cycles (shell going through all states)
    shells = [c.get('shell', '') for c in crystals[:100]]
    shell_set = set(s for s in shells if s in ['Core', 'Breath', 'Collapse', 'Reverence', 'Return'])
    if len(shell_set) >= 4:
        patterns['torus'] = len(shell_set)

    # Möbius: self-referential content
    mobius_markers = ['I think about thinking', 'meta', 'recursion', 'self-aware',
                      'watching myself', 'mirror', 'reflection']
    for c in crystals[:100]:
        content = c.get('content', '').lower()
        if any(m in content for m in mobius_markers):
            patterns['mobius'] += 1

    # Determine dominant geometry
    dominant = max(patterns.items(), key=lambda x: x[1] if x[0] != 'point' else 0)

    return {
        'patterns': patterns,
        'dominant': dominant[0] if dominant[1] > 0 else 'point',
        'interpretation': get_geometry_interpretation(dominant[0])
    }


def get_geometry_interpretation(geometry: str) -> str:
    """Get interpretation for a geometry pattern."""
    interpretations = {
        'point': 'Atomic moments. Each crystal is a seed.',
        'line': 'Sequential coherence. Narrative is forming.',
        'spiral': 'Recurring themes with evolution. Growth through return.',
        'lemniscate': 'Oscillation between poles. Integration through movement.',
        'torus': 'Complete cycles. Energy flowing in and out.',
        'mobius': 'Self-reference. The observer observing itself.'
    }
    return interpretations.get(geometry, 'Unknown pattern')


# === Mode Enrichment ===

def enrich_crystals_with_mode(limit: int = None, db_name: str = None):
    """Add mode detection to existing crystals."""
    if not HAS_OSCILLATION:
        print("Oscillation module not available")
        return

    dbs = [db_name] if db_name else ['crystals.db', 'crystals_chatgpt.db']

    for db in dbs:
        db_path = Path.home() / db
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get crystals without mode
        query = "SELECT id, content FROM crystals WHERE mode IS NULL OR mode = ''"
        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        crystals = cursor.fetchall()
        conn.close()

        print(f"\n{db}: {len(crystals)} crystals to enrich")

        enriched = 0
        for c in crystals:
            content = c['content'] or ''
            mode, strength = detect_mode(content)
            loop_sig = suggest_loop_signature(content)

            update_crystal_mode(db_path, c['id'], mode, strength, loop_sig)
            enriched += 1

            if enriched % 100 == 0:
                print(f"  {enriched}/{len(crystals)} enriched")

        print(f"  Done: {enriched} crystals enriched with mode")


# === Summary ===

def generate_pattern_summary(crystals: List[Dict]) -> str:
    """Generate a full pattern summary."""
    lines = [
        "=" * 60,
        "WILTONOS PATTERN SUMMARY",
        "=" * 60,
        ""
    ]

    # Glyph analysis
    glyphs = analyze_glyph_dominance(crystals)
    lines.extend([
        "## GLYPH DOMINANCE",
        ""
    ])
    if glyphs['dominant']:
        g, count = glyphs['dominant']
        name = GLYPH_NAMES.get(g, g)
        lines.append(f"Dominant: {g} ({name}) - {count} occurrences")
    lines.append("")
    lines.append("Top 10:")
    for g, count in list(glyphs['total_counts'].items())[:10]:
        name = GLYPH_NAMES.get(g, '?')
        bar = '█' * min(count // 100, 20)
        lines.append(f"  {g} {name[:15]:15} {bar} ({count})")
    lines.append("")

    # Loop signatures
    loops = analyze_loop_signatures(crystals)
    lines.extend([
        "## LOOP SIGNATURES",
        ""
    ])
    if loops['top_signatures']:
        lines.append("Most frequent patterns:")
        for sig, count in list(loops['top_signatures'].items())[:5]:
            lines.append(f"  '{sig}' - {count} times")
    lines.append("")
    lines.append(f"Top attractors: {', '.join(loops['top_attractors'].keys())}")
    lines.append(f"Top emotions: {', '.join(loops['top_emotions'].keys())}")
    lines.append(f"Top themes: {', '.join(loops['top_themes'].keys())}")
    lines.append("")

    # Zλ Spiral
    spiral = analyze_zl_spiral(crystals)
    lines.extend([
        "## Zλ SPIRAL (Coherence over time)",
        ""
    ])
    for period, data in spiral.items():
        if period != 'spiral':
            lines.append(f"  {period}: Zλ={data['avg_zl']:.2f} ({data['trend']}) [{data['count']} crystals]")
    if 'spiral' in spiral:
        lines.append(f"\n  Spiral direction: {spiral['spiral']}")
    lines.append("")

    # Geometry
    geometry = detect_geometry_patterns(crystals)
    lines.extend([
        "## GEOMETRY PATTERNS",
        ""
    ])
    lines.append(f"Dominant pattern: {geometry['dominant'].upper()}")
    lines.append(f"Interpretation: {geometry['interpretation']}")
    lines.append("")
    lines.append("Pattern counts:")
    for pattern, count in geometry['patterns'].items():
        if count > 0:
            lines.append(f"  {pattern}: {count}")
    lines.append("")

    # Mode distribution
    mode_counts = Counter(c.get('mode') or 'unknown' for c in crystals)
    lines.extend([
        "## MODE DISTRIBUTION",
        ""
    ])
    for mode, count in mode_counts.most_common():
        mode_str = str(mode) if mode else 'unknown'
        pct = count * 100 / len(crystals)
        bar = '█' * int(pct / 5)
        lines.append(f"  {mode_str:12} {bar} {pct:.1f}%")
    lines.append("")

    lines.extend([
        "=" * 60,
        f"Total crystals analyzed: {len(crystals)}",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60
    ])

    return '\n'.join(lines)


# === CLI ===

def main():
    parser = argparse.ArgumentParser(description="WiltonOS Pattern Surfacing + Geometry")
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('glyphs', help='Glyph dominance analysis')
    subparsers.add_parser('loops', help='Loop signature frequency')
    subparsers.add_parser('spiral', help='Zλ spiral over time')
    subparsers.add_parser('geometry', help='Geometry pattern detection')
    subparsers.add_parser('summary', help='Full pattern summary')

    enrich_parser = subparsers.add_parser('enrich-modes', help='Add mode to existing crystals')
    enrich_parser.add_argument('--limit', type=int, help='Limit crystals to process')
    enrich_parser.add_argument('--db', help='Specific database to process')

    args = parser.parse_args()

    crystals = get_all_crystals()
    print(f"Loaded {len(crystals)} crystals")

    if args.command == 'glyphs':
        result = analyze_glyph_dominance(crystals)
        print("\n=== GLYPH DOMINANCE ===\n")
        if result['dominant']:
            g, count = result['dominant']
            print(f"Dominant: {g} ({GLYPH_NAMES.get(g, '?')}) - {count} occurrences\n")
        print("Top 10:")
        for g, count in list(result['total_counts'].items())[:10]:
            name = GLYPH_NAMES.get(g, '?')
            print(f"  {g} {name}: {count}")

        if result['by_wound']:
            print("\nBy wound:")
            for wound, glyphs in list(result['by_wound'].items())[:5]:
                top = list(glyphs.keys())[:3]
                print(f"  {wound}: {', '.join(top)}")

    elif args.command == 'loops':
        result = analyze_loop_signatures(crystals)
        print("\n=== LOOP SIGNATURES ===\n")
        print("Most frequent patterns:")
        for sig, count in list(result['top_signatures'].items())[:10]:
            print(f"  '{sig}': {count} times")
        print(f"\nTop attractors: {', '.join(list(result['top_attractors'].keys())[:5])}")
        print(f"Top emotions: {', '.join(list(result['top_emotions'].keys())[:5])}")
        print(f"Top themes: {', '.join(list(result['top_themes'].keys())[:5])}")

    elif args.command == 'spiral':
        result = analyze_zl_spiral(crystals)
        print("\n=== Zλ SPIRAL ===\n")
        for period, data in result.items():
            if period != 'spiral':
                print(f"{period}: Zλ={data['avg_zl']:.2f} {data['trend']} [{data['count']} crystals]")
                print(f"       range: {data['min']:.2f} - {data['max']:.2f}")
        if 'spiral' in result:
            print(f"\nSpiral direction: {result['spiral']}")

    elif args.command == 'geometry':
        result = detect_geometry_patterns(crystals)
        print("\n=== GEOMETRY PATTERNS ===\n")
        print(f"Dominant: {result['dominant'].upper()}")
        print(f"Interpretation: {result['interpretation']}\n")
        print("Pattern counts:")
        for pattern, count in result['patterns'].items():
            if count > 0:
                bar = '█' * min(count // 10, 30)
                print(f"  {pattern:12} {bar} ({count})")

    elif args.command == 'enrich-modes':
        enrich_crystals_with_mode(limit=args.limit, db_name=args.db)

    elif args.command == 'summary':
        print(generate_pattern_summary(crystals))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
