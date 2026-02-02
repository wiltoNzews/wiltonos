#!/usr/bin/env python3
"""
WiltonOS Session Context Bridge - PULL MODEL with OSCILLATION ROUTING
Context is only loaded when explicitly requested, not auto-injected.
Routes context differently based on detected mode: WiltonOS (internal) vs ψOS (external).

Usage:
  python wiltonos_bridge.py pull                  # Print context to terminal
  python wiltonos_bridge.py pull --topic X       # Focus on topic X
  python wiltonos_bridge.py pull --file PATH     # Write to file for AI
  python wiltonos_bridge.py pull --json          # Machine-readable output
  python wiltonos_bridge.py pull --clipboard     # Copy to clipboard
  python wiltonos_bridge.py pull --spirals       # Include spiral analysis
  python wiltonos_bridge.py pull --glyphs        # Include glyph breakdown
  python wiltonos_bridge.py pull --meta          # Include meta-questions
  python wiltonos_bridge.py status               # Show memory stats
  python wiltonos_bridge.py mode "query"         # Detect mode for query
  python wiltonos_bridge.py oscillation          # Analyze oscillation pattern

Philosophy: Storage happens automatically. Recall is always ON DEMAND.
Routing: Context adapts based on WiltonOS (internal) vs ψOS (external) mode.
"""
import os
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict

# Import oscillation engine
try:
    from wiltonos_oscillation import (
        detect_mode, detect_mode_semantic, get_routing_config,
        analyze_oscillation, detect_lemniscate, suggest_loop_signature,
        validate_loop_signature
    )
    HAS_OSCILLATION = True
except ImportError:
    HAS_OSCILLATION = False
    def detect_mode(text): return ("neutral", 0.5)
    def get_routing_config(mode): return {"context_depth": "full", "tone": "balanced"}
    def analyze_oscillation(crystals): return {"pattern": "unknown"}
    def detect_lemniscate(crystals): return {"active": False}

CONTEXT_FILE = Path.home() / "WILTONOS_CONTEXT.md"
CLAUDE_MD = Path.home() / "CLAUDE.md"

# === Mode-Based Formatting ===
MODE_HEADERS = {
    "wiltonos": """
# 🪞 WiltonOS Session Context
## Mode: INTERNAL (WiltonOS)
*Dense, human, deep memory. Mirror tone.*
""",
    "psios": """
# ∇ ψOS Session Context
## Mode: EXTERNAL (ψOS)
*Symbolic, abstract, vector-based. Structure tone.*
""",
    "neutral": """
# ∅ WiltonOS Session Context
## Mode: BALANCED
*Adaptive context. Observing oscillation.*
"""
}

# === Database Access ===
def get_crystals(db_name: str) -> List[Dict]:
    """Load crystals from a specific database."""
    db_path = Path.home() / db_name
    if not db_path.exists():
        return []

    schemas = {
        'crystals_unified.db': """
            SELECT id, content, source, source_file, author,
                   original_timestamp as timestamp, zl_score, psi_aligned, trust_level,
                   shell, shell_direction, glyphs, glyph_primary, glyph_secondary,
                   glyph_energy_notes, glyph_direction, glyph_risk, glyph_antidote,
                   mode, oscillation_strength, core_wound, loop_signature,
                   attractor, emotion, theme, insight, question, analyzed_at,
                   breath_cadence, presence_density, emotional_resonance,
                   loop_pressure, groundedness
            FROM crystals
        """,
        'crystals_chatgpt.db': """
            SELECT id, content, source, source_file, author, conv_title, conv_id,
                   timestamp, zl_score, psi_aligned, trust_level, shell,
                   glyphs, insight, analyzed_at
            FROM crystals
        """,
        'crystals.db': """
            SELECT id, content, source, source_file, NULL as author, NULL as conv_title,
                   NULL as conv_id, NULL as timestamp, zl_score, psi_aligned, trust_level,
                   shell, glyphs, insight, analyzed_at
            FROM crystals
        """
    }

    query = schemas.get(db_name, schemas['crystals_unified.db'])

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(query).fetchall()
        crystals = [dict(row) for row in rows]
        for c in crystals:
            c['db'] = db_name
        return crystals
    except Exception as e:
        print(f"Warning: Error reading {db_name}: {e}")
        return []
    finally:
        conn.close()

def get_all_crystals() -> List[Dict]:
    """Load crystals from unified database (or fallback to separate DBs)."""
    # Prefer unified database
    unified = Path.home() / 'crystals_unified.db'
    if unified.exists():
        return get_crystals('crystals_unified.db')

    # Fallback to separate databases
    crystals = []
    for db_name in ['crystals_chatgpt.db', 'crystals.db']:
        crystals.extend(get_crystals(db_name))
    return crystals

# === Pattern Analysis ===
def analyze_recent(crystals: List[Dict], days: int = 14) -> Dict:
    """Analyze recent crystal patterns."""
    cutoff = datetime.now() - timedelta(days=days)

    recent = []
    for c in crystals:
        if c.get('timestamp') and c['timestamp'] > 0:
            try:
                crystal_time = datetime.fromtimestamp(c['timestamp'])
                if crystal_time > cutoff:
                    recent.append(c)
            except:
                pass

    # Fallback to analyzed_at if no timestamps
    if len(recent) < 20:
        sorted_all = sorted(crystals, key=lambda x: x.get('analyzed_at') or '', reverse=True)
        recent = sorted_all[:100]

    analysis = {
        'count': len(recent),
        'vulnerable_count': sum(1 for c in recent if c.get('trust_level') == 'VULNERABLE'),
        'avg_zl': sum(c.get('zl_score', 0) or 0 for c in recent) / max(len(recent), 1),
        'trust_dist': Counter(c.get('trust_level', 'UNKNOWN') for c in recent),
    }

    # Striving pressure
    striving_words = ['should', 'must', 'need to', 'have to', 'trying']
    striving_count = sum(1 for c in recent
                        if any(w in c['content'].lower() for w in striving_words))
    analysis['striving_pct'] = striving_count / max(len(recent), 1) * 100

    # Top themes in vulnerable crystals
    vuln_words = Counter()
    for c in recent:
        if c.get('trust_level') == 'VULNERABLE':
            for word in c['content'].lower().split():
                if len(word) > 5 and word.isalpha():
                    vuln_words[word] += 1
    analysis['vulnerable_themes'] = [w for w, _ in vuln_words.most_common(10)]

    return analysis

def find_open_threads(crystals: List[Dict]) -> List[Dict]:
    """Find crystals that might need follow-up."""
    # Look for vulnerable crystals with certain markers
    markers = ['need to', 'want to', 'should', 'going to', 'will', 'plan to',
               'preciso', 'quero', 'vou']

    candidates = []
    for c in crystals:
        if c.get('trust_level') != 'VULNERABLE':
            continue
        content_lower = c['content'].lower()
        if any(m in content_lower for m in markers):
            candidates.append(c)

    # Sort by recency
    candidates.sort(key=lambda x: x.get('timestamp') or 0, reverse=True)
    return candidates[:5]

def find_recurring_loops(crystals: List[Dict]) -> List[Tuple[str, int]]:
    """Find words that appear across many conversations."""
    conv_words = defaultdict(set)

    for c in crystals:
        if c.get('trust_level') != 'VULNERABLE':
            continue
        conv_id = c.get('conv_id') or c.get('source_file') or 'unknown'
        for word in c['content'].lower().split():
            if len(word) > 5 and word.isalpha():
                conv_words[word].add(conv_id)

    loops = [(word, len(convs)) for word, convs in conv_words.items() if len(convs) >= 50]
    loops.sort(key=lambda x: -x[1])
    return loops[:10]

def generate_questions(analysis: Dict, loops: List[Tuple[str, int]]) -> List[str]:
    """Generate self-inquiry questions based on patterns."""
    questions = []

    if analysis['striving_pct'] > 35:
        questions.append(
            f"Striving language appears in {analysis['striving_pct']:.0f}% of recent crystals. "
            "What are you pushing toward? What would happen if you paused?"
        )

    vuln_pct = analysis['vulnerable_count'] / max(analysis['count'], 1) * 100
    if vuln_pct < 20:
        questions.append(
            f"Only {vuln_pct:.0f}% vulnerable recently. Where is the armor thickest?"
        )
    elif vuln_pct > 40:
        questions.append(
            f"{vuln_pct:.0f}% vulnerable recently. What's moving through you?"
        )

    if loops:
        top_loop = loops[0]
        questions.append(
            f"'{top_loop[0]}' appears across {top_loop[1]} conversations. "
            "Is this loop closing or repeating?"
        )

    if analysis['vulnerable_themes']:
        themes = ', '.join(analysis['vulnerable_themes'][:5])
        questions.append(
            f"When vulnerable, these words appear: {themes}. What connects them?"
        )

    return questions

# === Context Generation ===
def generate_context(topic: str = None, query: str = None, include_questions: bool = True, include_meta: bool = False) -> str:
    """Generate full context document with mode-based routing."""
    crystals = get_all_crystals()
    analysis = analyze_recent(crystals)
    loops = find_recurring_loops(crystals)
    open_threads = find_open_threads(crystals)
    questions = generate_questions(analysis, loops) if include_questions else []

    # Detect mode from query or topic
    query_text = query or topic or ""
    mode, strength = detect_mode(query_text)
    routing = get_routing_config(mode)

    # Analyze oscillation pattern
    osc_analysis = analyze_oscillation(crystals[:50])  # Last 50 crystals
    lemniscate = detect_lemniscate(crystals[:50])

    # Get mode-appropriate header
    header = MODE_HEADERS.get(mode, MODE_HEADERS["neutral"])

    # Build context document
    lines = [
        header.strip(),
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
    ]

    # Add routing info
    lines.extend([
        "---",
        "",
        "## Routing Configuration",
        "",
        f"- **Detected Mode:** {mode} (strength: {strength})",
        f"- **Context Depth:** {routing['context_depth']}",
        f"- **Response Tone:** {routing['tone']}",
        f"- **Oscillation Pattern:** {osc_analysis.get('pattern', 'unknown')}",
    ])

    if lemniscate.get("active"):
        lines.append(f"- **⚠️ Lemniscate Active:** Oscillating between poles")

    lines.extend([
        "",
        "---",
        "",
        "## Current State",
        "",
        f"- **Total crystals in memory:** {len(crystals):,}",
        f"- **Recent crystals analyzed:** {analysis['count']}",
        f"- **Recent vulnerability:** {analysis['vulnerable_count']} ({analysis['vulnerable_count']/max(analysis['count'],1)*100:.0f}%)",
        f"- **Average Zλ:** {analysis['avg_zl']:.2f}",
        f"- **Striving pressure:** {analysis['striving_pct']:.0f}%",
        "",
    ])

    # Questions for inquiry
    if questions:
        lines.extend([
            "## Questions for This Session",
            "",
        ])
        for i, q in enumerate(questions, 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    # Recurring loops
    if loops:
        lines.extend([
            "## Recurring Patterns (potential loops)",
            "",
        ])
        for word, count in loops[:5]:
            lines.append(f"- **{word}** — appears across {count} conversations")
        lines.append("")

    # Open threads
    if open_threads:
        lines.extend([
            "## Open Threads (may need follow-up)",
            "",
        ])
        for i, c in enumerate(open_threads, 1):
            content = c['content'][:200].replace('\n', ' ')
            zl = c.get('zl_score', 0) or 0
            lines.append(f"### Thread {i} [Zλ={zl:.2f}]")
            lines.append(f"{content}...")
            lines.append("")

    # Topic-specific crystals if requested
    if topic:
        topic_lower = topic.lower()
        relevant = [c for c in crystals if topic_lower in c['content'].lower()]
        relevant.sort(key=lambda x: x.get('zl_score', 0) or 0, reverse=True)

        if relevant:
            lines.extend([
                f"## Crystals Related to: {topic}",
                "",
            ])
            for c in relevant[:5]:
                content = c['content'][:250].replace('\n', ' ')
                zl = c.get('zl_score', 0) or 0
                trust = c.get('trust_level', '?')
                lines.append(f"**[Zλ={zl:.2f}, {trust}]** {content}...")
                lines.append("")

    # Recent vulnerable crystals (the gold)
    vuln_recent = [c for c in crystals if c.get('trust_level') == 'VULNERABLE']
    vuln_recent.sort(key=lambda x: x.get('timestamp') or 0, reverse=True)

    if vuln_recent:
        lines.extend([
            "## Recent Vulnerable Moments",
            "",
            "*These are collapse-with-breath moments — the signal in the noise.*",
            "",
        ])
        for c in vuln_recent[:3]:
            content = c['content'][:300].replace('\n', ' ')
            zl = c.get('zl_score', 0) or 0
            lines.append(f"**[Zλ={zl:.2f}]** {content}...")
            if c.get('insight') and 'Keyword' not in str(c.get('insight', '')):
                lines.append(f"*→ {c['insight'][:100]}*")
            lines.append("")

    # Mode-based suggestions
    lines.extend([
        "---",
        "",
        "## Mode-Based Guidance",
        "",
    ])

    if mode == "wiltonos":
        lines.extend([
            "**You are in INTERNAL mode (WiltonOS)**",
            "",
            "Suggestions:",
            "- 🫁 Breathe. The loop wants to be seen, not solved.",
            "- 🪞 What pattern is repeating here?",
            "- 💔 Which wound is active? (Check core_wound field)",
            "- ⏸️ Before acting, pause. What are you avoiding?",
            "",
            "*Response style: Dense, human, mirror. Quote past crystals. Full memory depth.*",
        ])
    elif mode == "psios":
        lines.extend([
            "**You are in EXTERNAL mode (ψOS)**",
            "",
            "Suggestions:",
            "- ∇ What vector are you moving toward?",
            "- 🔧 What structure needs building?",
            "- ⚡ What decision is being avoided?",
            "- 📐 Which glyph energy is dominant?",
            "",
            "*Response style: Symbolic, abstract, vector. Shallow memory with summaries.*",
        ])
    else:
        lines.extend([
            "**You are in BALANCED mode**",
            "",
            "The system is observing. Neither fully internal nor external.",
            "- Notice which direction you're being pulled",
            "- The oscillation itself may be the message",
            "",
        ])

    if lemniscate.get("active"):
        lines.extend([
            "",
            "### ∞ Lemniscate Pattern Detected",
            "",
            "You are oscillating between WiltonOS (internal) and ψOS (external).",
            "This is natural. Breathe at the crossing point.",
            f"Transition rate: {lemniscate.get('transition_rate', 0):.0%}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Commands:*",
        "- `python ~/wiltonos_bridge.py pull --topic X` - Focus on topic",
        "- `python ~/wiltonos_bridge.py mode \"your query\"` - Detect mode",
        "- `python ~/wiltonos_bridge.py oscillation` - Analyze oscillation pattern",
        "- `python ~/wiltonos_memory.py query \"topic\"` - Search crystals",
    ])

    return '\n'.join(lines)

def update_claude_md():
    """Add reference to WiltonOS context in CLAUDE.md."""
    if not CLAUDE_MD.exists():
        print(f"CLAUDE.md not found at {CLAUDE_MD}")
        return False

    content = CLAUDE_MD.read_text()

    # Check if already integrated
    if 'WILTONOS_CONTEXT.md' in content:
        print("WiltonOS already referenced in CLAUDE.md")
        return True

    # Add integration section
    integration = """

## WiltonOS Memory Integration

At the start of sessions, read `~/WILTONOS_CONTEXT.md` for persistent memory context.
This file contains:
- Recent patterns and loops
- Self-inquiry questions based on crystal analysis
- Open threads that may need follow-up
- Recent vulnerable moments (the signal)

Refresh with: `python ~/wiltonos_bridge.py generate`
Query memory: `python ~/wiltonos_memory.py query "topic"`
"""

    new_content = content + integration
    CLAUDE_MD.write_text(new_content)
    print(f"Updated {CLAUDE_MD} with WiltonOS integration")
    return True

def show_status():
    """Show what would be bridged."""
    crystals = get_all_crystals()
    analysis = analyze_recent(crystals)
    loops = find_recurring_loops(crystals)
    open_threads = find_open_threads(crystals)

    print("\n" + "=" * 60)
    print("WILTONOS BRIDGE STATUS")
    print("=" * 60)
    print(f"\nTotal crystals: {len(crystals):,}")
    print(f"Recent analyzed: {analysis['count']}")
    print(f"Vulnerable recent: {analysis['vulnerable_count']}")
    print(f"Striving pressure: {analysis['striving_pct']:.0f}%")
    print(f"\nTop loops: {', '.join(w for w, _ in loops[:5])}")
    print(f"Open threads: {len(open_threads)}")
    print(f"\nContext file: {CONTEXT_FILE}")
    print(f"Exists: {CONTEXT_FILE.exists()}")
    if CONTEXT_FILE.exists():
        mtime = datetime.fromtimestamp(CONTEXT_FILE.stat().st_mtime)
        print(f"Last updated: {mtime.strftime('%Y-%m-%d %H:%M')}")

# === JSON Output ===
def generate_json_context(topic: str = None, include_spirals: bool = False, include_glyphs: bool = False) -> Dict:
    """Generate context as JSON for machine consumption."""
    crystals = get_all_crystals()
    analysis = analyze_recent(crystals)
    loops = find_recurring_loops(crystals)
    open_threads = find_open_threads(crystals)
    questions = generate_questions(analysis, loops)

    output = {
        "generated_at": datetime.now().isoformat(),
        "total_crystals": len(crystals),
        "recent_analysis": {
            "count": analysis['count'],
            "vulnerable_count": analysis['vulnerable_count'],
            "avg_zl": analysis['avg_zl'],
            "striving_pressure": analysis['striving_pct'],
        },
        "questions": questions,
        "recurring_patterns": [{"word": w, "conversations": c} for w, c in loops[:10]],
        "open_threads": [
            {"content": c['content'][:300], "zl_score": c.get('zl_score', 0) or 0}
            for c in open_threads
        ],
    }

    if topic:
        topic_lower = topic.lower()
        relevant = [c for c in crystals if topic_lower in c['content'].lower()][:5]
        output["topic_crystals"] = [
            {"content": c['content'][:300], "zl_score": c.get('zl_score', 0) or 0, "trust_level": c.get('trust_level', '?')}
            for c in relevant
        ]

    # Add spiral data if requested
    if include_spirals:
        from wiltonos_memory import detect_spirals
        spirals = detect_spirals()
        output["spirals"] = spirals

    # Add glyph data if requested
    if include_glyphs:
        from wiltonos_memory import get_glyph_frequency, GLYPH_DEFS
        freq = get_glyph_frequency()
        output["glyphs"] = {
            "frequency": freq['counts'],
            "definitions": GLYPH_DEFS.get("glyphs", {})
        }

    return output

# === Main ===
def main():
    parser = argparse.ArgumentParser(description="WiltonOS Session Context Bridge - PULL MODEL with OSCILLATION ROUTING")
    subparsers = parser.add_subparsers(dest='command')

    # Pull command (primary)
    pull_parser = subparsers.add_parser('pull', help='Pull context on demand')
    pull_parser.add_argument('query', nargs='?', help='Query to detect mode from')
    pull_parser.add_argument('--topic', help='Focus on specific topic')
    pull_parser.add_argument('--file', help='Write to file instead of terminal')
    pull_parser.add_argument('--json', action='store_true', help='Output as JSON')
    pull_parser.add_argument('--clipboard', action='store_true', help='Copy to clipboard')
    pull_parser.add_argument('--spirals', action='store_true', help='Include spiral analysis')
    pull_parser.add_argument('--glyphs', action='store_true', help='Include glyph breakdown')
    pull_parser.add_argument('--meta', action='store_true', help='Include meta-questions from agents')
    pull_parser.add_argument('--no-questions', action='store_true', help='Skip self-inquiry questions')

    # Mode detection command
    mode_parser = subparsers.add_parser('mode', help='Detect mode for a query')
    mode_parser.add_argument('query', help='Query text to analyze')

    # Oscillation analysis command
    osc_parser = subparsers.add_parser('oscillation', help='Analyze oscillation patterns')
    osc_parser.add_argument('--recent', type=int, default=50, help='Number of recent crystals to analyze')

    # Status command
    subparsers.add_parser('status', help='Show memory stats')

    # Legacy generate (redirects to pull)
    gen_parser = subparsers.add_parser('generate', help='(Legacy) Same as pull --file')
    gen_parser.add_argument('--topic', help='Focus on specific topic')
    gen_parser.add_argument('--output', help='Output file path')

    args = parser.parse_args()

    if args.command == 'pull':
        query = getattr(args, 'query', None)

        if args.json:
            # JSON output
            data = generate_json_context(
                topic=args.topic,
                include_spirals=args.spirals,
                include_glyphs=args.glyphs
            )
            # Add mode detection to JSON
            if query:
                mode, strength = detect_mode(query)
                data["mode"] = {"detected": mode, "strength": strength, "routing": get_routing_config(mode)}
            output = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            # Markdown output with mode routing
            output = generate_context(
                topic=args.topic,
                query=query,
                include_questions=not args.no_questions,
                include_meta=args.meta
            )

            # Add spirals section if requested
            if args.spirals:
                try:
                    from wiltonos_memory import detect_spirals
                    spirals = detect_spirals()
                    if spirals:
                        spiral_section = "\n\n## Spiral Patterns\n\n"
                        spiral_section += "| Pattern | 7d | 30d | All | Occurrences |\n"
                        spiral_section += "|---------|----|----|-----|-------------|\n"
                        for s in spirals[:10]:
                            spiral_section += f"| {s['word'][:12]} | {s['7d_trend']} | {s['30d_trend']} | {s['all_time_trend']} | {s['occurrences']} |\n"
                        output += spiral_section
                except Exception as e:
                    output += f"\n\n(Spiral analysis error: {e})"

            # Add glyphs section if requested
            if args.glyphs:
                try:
                    from wiltonos_memory import get_glyph_frequency, get_glyph_meaning
                    freq = get_glyph_frequency()
                    if freq['counts']:
                        glyph_section = "\n\n## Glyph Frequency\n\n"
                        for glyph, count in list(freq['counts'].items())[:10]:
                            info = get_glyph_meaning(glyph)
                            name = info.get('name', '?')
                            glyph_section += f"- **{glyph} [{name}]**: {count} occurrences\n"
                        output += glyph_section
                except Exception as e:
                    output += f"\n\n(Glyph analysis error: {e})"

        # Output destination
        if args.file:
            Path(args.file).write_text(output)
            print(f"Context written to {args.file}")
            print(f"Size: {len(output):,} characters")
        elif args.clipboard:
            try:
                import subprocess
                process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE)
                process.communicate(output.encode())
                print(f"Context copied to clipboard ({len(output):,} characters)")
            except Exception as e:
                print(f"Clipboard error: {e}")
                print("Falling back to terminal output:")
                print(output)
        else:
            # Terminal output (default)
            print(output)

    elif args.command == 'generate':
        # Legacy command - redirect to file output
        context = generate_context(topic=args.topic)
        output_path = Path(args.output) if args.output else CONTEXT_FILE
        output_path.write_text(context)
        print(f"(Legacy) Context written to {output_path}")

    elif args.command == 'mode':
        # Mode detection command
        mode, strength = detect_mode(args.query)
        routing = get_routing_config(mode)

        print(f"\n{'=' * 50}")
        print(f"MODE DETECTION")
        print(f"{'=' * 50}")
        print(f"\nQuery: {args.query[:100]}{'...' if len(args.query) > 100 else ''}")
        print(f"\nDetected Mode: {mode.upper()}")
        print(f"Strength: {strength:.0%}")
        print(f"\nRouting Configuration:")
        for k, v in routing.items():
            print(f"  {k}: {v}")

        # Suggest loop signature
        if HAS_OSCILLATION:
            sig = suggest_loop_signature(args.query)
            valid, _ = validate_loop_signature(sig)
            print(f"\nSuggested loop_signature: {sig}")
            print(f"Valid: {'✅' if valid else '❌'}")

    elif args.command == 'oscillation':
        # Oscillation analysis command
        crystals = get_all_crystals()
        recent = crystals[:args.recent]

        osc = analyze_oscillation(recent)
        lem = detect_lemniscate(recent)

        print(f"\n{'=' * 50}")
        print(f"OSCILLATION ANALYSIS (last {args.recent} crystals)")
        print(f"{'=' * 50}")
        print(f"\nPattern: {osc.get('pattern', 'unknown').upper()}")
        print(f"Dominant Mode: {osc.get('dominant_mode', 'unknown')}")
        print(f"Transitions: {osc.get('transitions', 0)}")
        print(f"Transition Rate: {osc.get('transition_rate', 0):.0%}")

        if osc.get('mode_counts'):
            print(f"\nMode Distribution:")
            for mode, count in osc['mode_counts'].items():
                bar = '█' * (count * 20 // max(osc['mode_counts'].values()))
                print(f"  {mode}: {bar} ({count})")

        if lem.get('active'):
            print(f"\n⚠️  LEMNISCATE ACTIVE")
            print(f"You are oscillating between poles.")
            print(f"Suggestion: {lem.get('suggestion', 'Breathe at the crossing point.')}")
        else:
            print(f"\n📍 Stable in: {lem.get('dominant', 'unknown')}")

    elif args.command == 'status':
        show_status()

        # Add oscillation status
        if HAS_OSCILLATION:
            crystals = get_all_crystals()
            osc = analyze_oscillation(crystals[:30])
            print(f"\nOscillation: {osc.get('pattern', 'unknown')} (dominant: {osc.get('dominant_mode', '?')})")

    else:
        # Default: show status
        show_status()

if __name__ == "__main__":
    main()
