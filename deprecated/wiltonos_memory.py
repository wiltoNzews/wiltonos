#!/usr/bin/env python3
"""
WiltonOS Memory Layer - Retrieval + Spiral/Glyph Understanding System
Queries crystals, detects spirals (not loops), surfaces insights.

Usage:
  python wiltonos_memory.py query "when was I vulnerable about Juliana?"
  python wiltonos_memory.py spirals                    # Show spiral patterns with △/∇/↺
  python wiltonos_memory.py patterns --days 7
  python wiltonos_memory.py glyphs                     # Show glyph frequency and meaning
  python wiltonos_memory.py query "X" --analyze-glyphs # AI glyph detection
  python wiltonos_memory.py context --topic "relationships" --limit 10
  python wiltonos_memory.py summary
"""
import os
import re
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import requests

OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")
GLYPHS_FILE = Path.home() / "wiltonos_glyphs.json"

# === Five Dimensions ===
def parse_coherence_vector(crystal: Dict) -> Dict[str, float]:
    """Extract the five dimensions from a crystal.

    The Five Dimensions:
      Zλ (zl_score)        - Overall coherence/authenticity
      breath_cadence       - Pause, regulation, ψ-presence
      presence_density     - Groundedness, embodiment
      emotional_resonance  - Depth of feeling
      loop_pressure        - Stuck-ness, repetition
    """
    dims = {
        'zl': crystal.get('zl_score') or 0,
        'breath_cadence': 0,
        'presence_density': 0,
        'emotional_resonance': 0,
        'loop_pressure': 0
    }

    cv = crystal.get('coherence_vector')
    if cv:
        try:
            if isinstance(cv, str):
                cv = json.loads(cv)
            if isinstance(cv, dict):
                dims['breath_cadence'] = cv.get('breath_cadence') or 0
                dims['presence_density'] = cv.get('presence_density') or 0
                dims['emotional_resonance'] = cv.get('emotional_resonance') or 0
                dims['loop_pressure'] = cv.get('loop_pressure') or 0
        except:
            pass

    return dims

def query_dimensions(crystals: List[Dict],
                     zl_min: float = None, zl_max: float = None,
                     breath_min: float = None, breath_max: float = None,
                     presence_min: float = None, presence_max: float = None,
                     emotional_min: float = None, emotional_max: float = None,
                     loop_min: float = None, loop_max: float = None,
                     limit: int = 20) -> List[Dict]:
    """Query crystals by any combination of five dimensions.

    Examples:
      - zl_max=0.5, breath_min=0.7 → "Collapse with breath" = VULNERABLE gold
      - zl_min=0.7, presence_max=0.4 → "High coherence, low presence" = performance?
      - loop_min=0.7 → "High loop pressure" = stuck patterns
    """
    results = []

    for c in crystals:
        dims = parse_coherence_vector(c)

        # Check each filter
        if zl_min is not None and dims['zl'] < zl_min:
            continue
        if zl_max is not None and dims['zl'] > zl_max:
            continue
        if breath_min is not None and dims['breath_cadence'] < breath_min:
            continue
        if breath_max is not None and dims['breath_cadence'] > breath_max:
            continue
        if presence_min is not None and dims['presence_density'] < presence_min:
            continue
        if presence_max is not None and dims['presence_density'] > presence_max:
            continue
        if emotional_min is not None and dims['emotional_resonance'] < emotional_min:
            continue
        if emotional_max is not None and dims['emotional_resonance'] > emotional_max:
            continue
        if loop_min is not None and dims['loop_pressure'] < loop_min:
            continue
        if loop_max is not None and dims['loop_pressure'] > loop_max:
            continue

        c['_dims'] = dims  # Attach parsed dimensions
        results.append(c)

    return results[:limit]

def get_dimension_patterns() -> Dict:
    """Analyze patterns across the five dimensions."""
    crystals = get_all_crystals()

    patterns = {
        'total': len(crystals),
        'with_coherence_vector': 0,
        'dimension_averages': {},
        'notable_patterns': []
    }

    dim_values = defaultdict(list)

    for c in crystals:
        dims = parse_coherence_vector(c)

        # Check if coherence_vector is populated
        if any(dims[k] > 0 for k in ['breath_cadence', 'presence_density', 'emotional_resonance', 'loop_pressure']):
            patterns['with_coherence_vector'] += 1

        for k, v in dims.items():
            if v > 0:
                dim_values[k].append(v)

    # Calculate averages
    for dim, values in dim_values.items():
        if values:
            patterns['dimension_averages'][dim] = sum(values) / len(values)

    # Detect notable patterns
    vulnerable_gold = query_dimensions(crystals, zl_max=0.5, breath_min=0.6, limit=1000)
    if vulnerable_gold:
        patterns['notable_patterns'].append({
            'name': 'Vulnerable Gold (low Zλ + high breath)',
            'count': len(vulnerable_gold),
            'meaning': 'Collapse with breath present = the signal in the noise'
        })

    polished_ungrounded = query_dimensions(crystals, zl_min=0.7, presence_max=0.4, limit=1000)
    if polished_ungrounded:
        patterns['notable_patterns'].append({
            'name': 'Polished Ungrounded (high Zλ + low presence)',
            'count': len(polished_ungrounded),
            'meaning': 'Coherent but not embodied = possible performance'
        })

    stuck_pressure = query_dimensions(crystals, loop_min=0.6, limit=1000)
    if stuck_pressure:
        patterns['notable_patterns'].append({
            'name': 'High Loop Pressure',
            'count': len(stuck_pressure),
            'meaning': 'Patterns with repetition pressure = potential stuck loops'
        })

    return patterns

# === Glyph Definitions ===
def load_glyphs() -> Dict:
    """Load canonical glyph definitions."""
    if GLYPHS_FILE.exists():
        return json.loads(GLYPHS_FILE.read_text())
    return {"glyphs": {}, "shells": {}, "trust_matrix": {}}

GLYPH_DEFS = load_glyphs()

def get_glyph_meaning(glyph: str) -> Dict:
    """Get the meaning of a specific glyph."""
    return GLYPH_DEFS.get("glyphs", {}).get(glyph, {})

def format_glyph_info(glyph: str) -> str:
    """Format glyph information for display."""
    info = get_glyph_meaning(glyph)
    if not info:
        return f"{glyph}: (unknown)"
    return f"{glyph} [{info.get('name', '?')}]: {info.get('function', '?')} | Risk: {info.get('risk', '?')}"

# === Database Access ===
def get_all_crystals() -> List[Dict]:
    """Load crystals from both databases."""
    crystals = []

    # Different schemas for different databases
    schemas = {
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

    for db_name, query in schemas.items():
        db_path = Path.home() / db_name
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        try:
            rows = conn.execute(query).fetchall()
            for row in rows:
                crystal = dict(row)
                crystal['db'] = db_name
                crystals.append(crystal)
        except Exception as e:
            print(f"Warning: Error reading {db_name}: {e}")
        finally:
            conn.close()

    return crystals

def search_crystals(query: str, trust_filter: str = None, limit: int = 20) -> List[Dict]:
    """Search crystals by keyword."""
    crystals = get_all_crystals()
    query_lower = query.lower()
    words = query_lower.split()

    scored = []
    for c in crystals:
        content_lower = c['content'].lower()

        # Score by word matches
        score = sum(1 for w in words if w in content_lower)
        if score == 0:
            continue

        # Boost for trust level match
        if trust_filter and c.get('trust_level') == trust_filter:
            score += 2

        # Boost for psi-aligned
        if c.get('psi_aligned'):
            score += 0.5

        scored.append((score, c))

    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:limit]]

# === Semantic Query via Ollama ===
def semantic_query(query: str, crystals: List[Dict], limit: int = 10) -> List[Dict]:
    """Use Ollama to find semantically relevant crystals."""
    # First pass: keyword filter to reduce candidates
    query_words = set(query.lower().split())
    candidates = []

    for c in crystals:
        content_lower = c['content'].lower()
        # Include if any query word appears, or if it's a vulnerable crystal
        if any(w in content_lower for w in query_words) or c.get('trust_level') == 'VULNERABLE':
            candidates.append(c)

    # Limit candidates for Ollama processing
    candidates = candidates[:100]

    if not candidates:
        return []

    # Ask Ollama to rank relevance
    prompt = f"""Given this query: "{query}"

Rank these text snippets by relevance (most relevant first). Return ONLY a JSON array of indices.

Snippets:
"""
    for i, c in enumerate(candidates[:20]):
        snippet = c['content'][:200].replace('\n', ' ')
        prompt += f"\n[{i}]: {snippet}"

    prompt += "\n\nReturn JSON array of indices, e.g. [3, 7, 1, 0]. Most relevant first."

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        result = response.json().get("response", "")

        # Extract array from response
        match = re.search(r'\[[\d,\s]+\]', result)
        if match:
            indices = json.loads(match.group())
            return [candidates[i] for i in indices if i < len(candidates)][:limit]
    except:
        pass

    # Fallback to keyword search
    return candidates[:limit]

# === Pattern Detection ===
def detect_patterns(days: int = 7) -> Dict:
    """Detect patterns in recent crystals."""
    crystals = get_all_crystals()
    cutoff = datetime.now() - timedelta(days=days)

    recent = []
    for c in crystals:
        if c.get('timestamp') and c['timestamp'] > 0:
            crystal_time = datetime.fromtimestamp(c['timestamp'])
            if crystal_time > cutoff:
                recent.append(c)

    if not recent:
        # Fallback to last N crystals by analyzed_at
        recent = sorted(crystals, key=lambda x: x.get('analyzed_at', ''), reverse=True)[:200]

    patterns = {
        'total_crystals': len(recent),
        'trust_distribution': Counter(c.get('trust_level', 'UNKNOWN') for c in recent),
        'avg_zl': sum(c.get('zl_score', 0) or 0 for c in recent) / max(len(recent), 1),
        'psi_aligned_pct': sum(1 for c in recent if c.get('psi_aligned')) / max(len(recent), 1) * 100,
    }

    # Word frequency in vulnerable crystals
    vuln_words = Counter()
    for c in recent:
        if c.get('trust_level') == 'VULNERABLE':
            for word in c['content'].lower().split():
                if len(word) > 4:
                    vuln_words[word] += 1

    patterns['vulnerable_themes'] = vuln_words.most_common(15)

    # Striving language detection
    striving_words = ['should', 'must', 'need to', 'have to', 'trying', 'better']
    striving_count = 0
    for c in recent:
        content_lower = c['content'].lower()
        if any(w in content_lower for w in striving_words):
            striving_count += 1

    patterns['striving_pressure'] = striving_count / max(len(recent), 1) * 100

    return patterns

# === Loop Detection ===
def detect_loops() -> List[Dict]:
    """Find recurring patterns across conversations."""
    crystals = get_all_crystals()

    # Group by conversation
    conv_crystals = defaultdict(list)
    for c in crystals:
        conv_id = c.get('conv_id') or c.get('source_file') or 'unknown'
        conv_crystals[conv_id].append(c)

    # Find words that appear in many different conversations
    word_convs = defaultdict(set)
    word_contexts = defaultdict(list)

    for conv_id, conv_list in conv_crystals.items():
        for c in conv_list:
            if c.get('trust_level') != 'VULNERABLE':
                continue
            words = set(c['content'].lower().split())
            for word in words:
                if len(word) > 5:
                    word_convs[word].add(conv_id)
                    if len(word_contexts[word]) < 3:
                        word_contexts[word].append(c['content'][:100])

    # Words appearing in 30+ different conversations = potential loops
    loops = []
    for word, convs in word_convs.items():
        if len(convs) >= 30:
            loops.append({
                'word': word,
                'conversations': len(convs),
                'examples': word_contexts[word]
            })

    loops.sort(key=lambda x: -x['conversations'])
    return loops[:20]

# === Spiral Detection (not loops - tracks direction) ===
def detect_spirals() -> List[Dict]:
    """Detect spirals - patterns that ascend (△), descend (∇), or stay flat (↺)."""
    crystals = get_all_crystals()
    now = datetime.now()

    # Group crystals by word and track Zλ over time
    word_crystals = defaultdict(list)

    for c in crystals:
        if c.get('trust_level') != 'VULNERABLE':
            continue
        timestamp = c.get('timestamp') or 0
        if timestamp <= 0:
            continue

        zl = c.get('zl_score', 0) or 0
        for word in c['content'].lower().split():
            if len(word) > 5 and word.isalpha():
                word_crystals[word].append({
                    'timestamp': timestamp,
                    'zl': zl,
                    'content': c['content'][:100]
                })

    spirals = []

    for word, occurrences in word_crystals.items():
        if len(occurrences) < 5:  # Need enough data points
            continue

        # Sort by timestamp
        occurrences.sort(key=lambda x: x['timestamp'])

        # Calculate Zλ trends at different timeframes
        def calc_trend(occ_list, days_ago_start, days_ago_end):
            # days_ago_start is more recent (e.g., 0 = now)
            # days_ago_end is older (e.g., 7 = 7 days ago)
            cutoff_recent = (now - timedelta(days=days_ago_start)).timestamp()
            cutoff_older = (now - timedelta(days=days_ago_end)).timestamp()

            # Select crystals between these dates (older <= ts < recent)
            in_range = [o for o in occ_list if cutoff_older <= o['timestamp'] < cutoff_recent]
            return sum(o['zl'] for o in in_range) / len(in_range) if in_range else None

        # Recent (0-7 days) vs older (7-30 days)
        recent_7d = calc_trend(occurrences, 0, 7)
        older_7d = calc_trend(occurrences, 7, 30)

        # Recent (0-30 days) vs older (30-90 days)
        recent_30d = calc_trend(occurrences, 0, 30)
        older_30d = calc_trend(occurrences, 30, 90)

        # All-time: first half vs second half
        mid = len(occurrences) // 2
        first_half_zl = sum(o['zl'] for o in occurrences[:mid]) / max(mid, 1)
        second_half_zl = sum(o['zl'] for o in occurrences[mid:]) / max(len(occurrences) - mid, 1)

        def direction(newer, older):
            if newer is None or older is None:
                return "?"
            diff = newer - older
            if diff > 0.1:
                return "△"  # Ascending
            elif diff < -0.1:
                return "∇"  # Descending
            else:
                return "↺"  # Flat/stuck

        spiral = {
            'word': word,
            'occurrences': len(occurrences),
            '7d_trend': direction(recent_7d, older_7d),
            '30d_trend': direction(recent_30d, older_30d),
            'all_time_trend': direction(second_half_zl, first_half_zl),
            'current_zl': recent_7d or recent_30d or second_half_zl,
            'examples': [o['content'] for o in occurrences[-2:]]
        }

        # Only include if we have meaningful data
        if spiral['7d_trend'] != "?" or spiral['30d_trend'] != "?":
            spirals.append(spiral)

    # Sort by occurrence count and filter
    spirals.sort(key=lambda x: -x['occurrences'])
    return spirals[:20]

# === AI Glyph Analysis ===
def analyze_glyphs_ai(text: str) -> Dict:
    """Use AI to detect which glyph energies are present (not keyword matching)."""
    glyph_list = list(GLYPH_DEFS.get("glyphs", {}).items())

    prompt = f"""Analyze this text and determine which of these glyph energies are present based on MEANING, not just keywords.

TEXT:
{text[:1500]}

GLYPHS TO DETECT:
"""
    for glyph, info in glyph_list[:10]:
        prompt += f"- {glyph} ({info.get('name', '?')}): {info.get('function', '?')}\n"

    prompt += """
Return ONLY valid JSON:
{
  "detected_glyphs": ["ψ", "∅", ...],
  "primary_glyph": "ψ",
  "glyph_notes": "brief explanation of why these energies are present"
}
"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.2}},
            timeout=90
        )
        result = response.json().get("response", "")
        match = re.search(r'\{[\s\S]*\}', result)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  AI glyph analysis error: {e}")

    return {"detected_glyphs": [], "primary_glyph": None, "glyph_notes": "Analysis failed"}

def get_glyph_frequency() -> Dict:
    """Analyze glyph frequency across all crystals."""
    crystals = get_all_crystals()
    glyph_counts = Counter()
    glyph_contexts = defaultdict(list)

    for c in crystals:
        glyphs_str = c.get('glyphs', '[]')
        try:
            glyphs = json.loads(glyphs_str) if isinstance(glyphs_str, str) else glyphs_str
        except:
            glyphs = []

        if not glyphs:
            continue

        for g in glyphs:
            # Handle if g is a dict or other type
            if isinstance(g, dict):
                g = str(g)
            if not isinstance(g, str):
                continue
            glyph_counts[g] += 1
            if len(glyph_contexts[g]) < 2:
                glyph_contexts[g].append(c['content'][:100])

    return {
        'counts': dict(glyph_counts.most_common(20)),
        'contexts': dict(glyph_contexts)
    }

# === Context Generator ===
def generate_context(topic: str = None, limit: int = 10) -> str:
    """Generate context file for new sessions."""
    crystals = get_all_crystals()

    if topic:
        relevant = search_crystals(topic, limit=limit * 2)
    else:
        # Get mix of recent vulnerable and high-zl crystals
        sorted_crystals = sorted(crystals, key=lambda x: x.get('analyzed_at', ''), reverse=True)
        vulnerable = [c for c in sorted_crystals if c.get('trust_level') == 'VULNERABLE'][:limit//2]
        high_zl = sorted([c for c in sorted_crystals if (c.get('zl_score') or 0) > 0.7],
                        key=lambda x: -x.get('zl_score', 0))[:limit//2]
        relevant = vulnerable + high_zl

    context_parts = [
        "# WiltonOS Memory Context",
        f"# Generated: {datetime.now().isoformat()}",
        f"# Topic: {topic or 'general'}",
        "",
        "## Recent Crystals",
        ""
    ]

    for i, c in enumerate(relevant[:limit], 1):
        zl = c.get('zl_score', 0) or 0
        trust = c.get('trust_level', '?')
        content = c['content'][:300].replace('\n', ' ')
        context_parts.append(f"### Crystal {i} [Zλ={zl:.2f}, {trust}]")
        context_parts.append(content)
        if c.get('insight'):
            context_parts.append(f"*Insight: {c['insight'][:100]}*")
        context_parts.append("")

    return '\n'.join(context_parts)

# === Summary ===
def generate_summary() -> str:
    """Generate overall memory summary."""
    crystals = get_all_crystals()

    total = len(crystals)
    trust_dist = Counter(c.get('trust_level', 'UNKNOWN') for c in crystals)
    avg_zl = sum(c.get('zl_score', 0) or 0 for c in crystals) / max(total, 1)
    psi_count = sum(1 for c in crystals if c.get('psi_aligned'))

    # Time range
    timestamps = [c.get('timestamp') for c in crystals if c.get('timestamp') and c.get('timestamp') > 0]
    if timestamps:
        earliest = datetime.fromtimestamp(min(timestamps))
        latest = datetime.fromtimestamp(max(timestamps))
        time_range = f"{earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"
    else:
        time_range = "Unknown"

    summary = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    WILTONOS MEMORY SUMMARY                       ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Crystals: {total:,}
║  Time Range: {time_range}
║  Average Zλ: {avg_zl:.3f}
║  Psi-Aligned: {psi_count:,} ({psi_count/max(total,1)*100:.0f}%)
╠══════════════════════════════════════════════════════════════════╣
║  TRUST DISTRIBUTION                                              ║
"""

    for trust in ['VULNERABLE', 'POLISHED', 'SCATTERED', 'HIGH']:
        count = trust_dist.get(trust, 0)
        pct = count / max(total, 1) * 100
        bar = '█' * int(pct / 2)
        summary += f"║    {trust:12} {count:>6,} ({pct:>5.1f}%) {bar}\n"

    summary += """╠══════════════════════════════════════════════════════════════════╣
║  Use: python wiltonos_memory.py <command>                        ║
║    query "text"  - Search crystals                               ║
║    patterns      - Recent patterns                               ║
║    loops         - Recurring themes                              ║
║    context       - Generate session context                      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    return summary

# === Questioning Layer ===
def question_self(crystals: List[Dict] = None) -> List[str]:
    """Generate questions based on patterns."""
    if crystals is None:
        crystals = get_all_crystals()

    patterns = detect_patterns(days=30)
    loops = detect_loops()

    questions = []

    # Based on striving pressure
    if patterns['striving_pressure'] > 40:
        questions.append(
            f"You used striving language ('should', 'must', 'need to') in {patterns['striving_pressure']:.0f}% of recent crystals. "
            "What would happen if you stopped pushing for a week?"
        )

    # Based on trust distribution
    vuln_pct = patterns['trust_distribution'].get('VULNERABLE', 0) / max(patterns['total_crystals'], 1) * 100
    if vuln_pct < 20:
        questions.append(
            f"Only {vuln_pct:.0f}% vulnerable crystals recently. Where is the armor thickest right now?"
        )
    elif vuln_pct > 40:
        questions.append(
            f"{vuln_pct:.0f}% vulnerable crystals recently. That's high. What's moving through you?"
        )

    # Based on loops
    if loops:
        top_loop = loops[0]
        questions.append(
            f"The word '{top_loop['word']}' appears across {top_loop['conversations']} different conversations. "
            "Is this a loop closing or a loop stuck?"
        )

    # Based on themes
    if patterns['vulnerable_themes']:
        top_themes = [w for w, _ in patterns['vulnerable_themes'][:5]]
        questions.append(
            f"When vulnerable, these words appear most: {', '.join(top_themes)}. "
            "What connects them?"
        )

    return questions

# === Main CLI ===
def main():
    parser = argparse.ArgumentParser(description="WiltonOS Memory Layer")
    subparsers = parser.add_subparsers(dest='command')

    # Query command
    query_parser = subparsers.add_parser('query', help='Search crystals')
    query_parser.add_argument('text', help='Search query')
    query_parser.add_argument('--trust', help='Filter by trust level')
    query_parser.add_argument('--limit', type=int, default=10)
    query_parser.add_argument('--semantic', action='store_true', help='Use AI for semantic search')
    query_parser.add_argument('--analyze-glyphs', action='store_true', help='Use AI to detect glyph energies')

    # Patterns command
    patterns_parser = subparsers.add_parser('patterns', help='Detect recent patterns')
    patterns_parser.add_argument('--days', type=int, default=7)

    # Loops command (legacy)
    subparsers.add_parser('loops', help='Find recurring themes (legacy)')

    # Spirals command (replaces loops)
    subparsers.add_parser('spirals', help='Detect spirals with △/∇/↺ direction')

    # Glyphs command
    subparsers.add_parser('glyphs', help='Show glyph frequency and meanings')

    # Context command
    context_parser = subparsers.add_parser('context', help='Generate session context')
    context_parser.add_argument('--topic', help='Focus topic')
    context_parser.add_argument('--limit', type=int, default=10)
    context_parser.add_argument('--output', help='Output file')

    # Summary command
    subparsers.add_parser('summary', help='Memory summary')

    # Questions command
    subparsers.add_parser('questions', help='Generate self-inquiry questions')

    # Dimensions command
    dims_parser = subparsers.add_parser('dimensions', help='Query by five dimensions')
    dims_parser.add_argument('--zl-min', type=float, help='Minimum Zλ')
    dims_parser.add_argument('--zl-max', type=float, help='Maximum Zλ')
    dims_parser.add_argument('--breath-min', type=float, help='Minimum breath_cadence')
    dims_parser.add_argument('--breath-max', type=float, help='Maximum breath_cadence')
    dims_parser.add_argument('--presence-min', type=float, help='Minimum presence_density')
    dims_parser.add_argument('--presence-max', type=float, help='Maximum presence_density')
    dims_parser.add_argument('--emotional-min', type=float, help='Minimum emotional_resonance')
    dims_parser.add_argument('--emotional-max', type=float, help='Maximum emotional_resonance')
    dims_parser.add_argument('--loop-min', type=float, help='Minimum loop_pressure')
    dims_parser.add_argument('--loop-max', type=float, help='Maximum loop_pressure')
    dims_parser.add_argument('--limit', type=int, default=20)
    dims_parser.add_argument('--patterns', action='store_true', help='Show dimension patterns summary')

    args = parser.parse_args()

    if args.command == 'query':
        if args.semantic:
            crystals = get_all_crystals()
            results = semantic_query(args.text, crystals, args.limit)
        else:
            results = search_crystals(args.text, args.trust, args.limit)

        print(f"\n{'='*60}")
        print(f"Query: {args.text}")
        print(f"Results: {len(results)}")
        if args.analyze_glyphs:
            print("(AI glyph analysis enabled)")
        print('='*60)

        for i, c in enumerate(results, 1):
            zl = c.get('zl_score', 0) or 0
            trust = c.get('trust_level', '?')
            content = c['content'][:300].replace('\n', ' ')
            print(f"\n[{i}] Zλ={zl:.2f} [{trust}]")
            print(f"    {content}...")

            # AI glyph analysis if requested
            if args.analyze_glyphs:
                print("    Analyzing glyphs...")
                glyph_result = analyze_glyphs_ai(c['content'])
                detected = glyph_result.get('detected_glyphs', [])
                primary = glyph_result.get('primary_glyph')
                notes = glyph_result.get('glyph_notes', '')

                if detected:
                    print(f"    Glyphs detected: {' '.join(detected)}")
                    if primary:
                        info = get_glyph_meaning(primary)
                        print(f"    Primary: {primary} [{info.get('name', '?')}] - {info.get('function', '?')[:50]}")
                    if notes:
                        print(f"    Notes: {notes[:100]}")

    elif args.command == 'patterns':
        patterns = detect_patterns(args.days)
        print(f"\n{'='*60}")
        print(f"PATTERNS (last {args.days} days)")
        print('='*60)
        print(f"Crystals analyzed: {patterns['total_crystals']}")
        print(f"Average Zλ: {patterns['avg_zl']:.3f}")
        print(f"Psi-aligned: {patterns['psi_aligned_pct']:.0f}%")
        print(f"Striving pressure: {patterns['striving_pressure']:.0f}%")
        print(f"\nTrust distribution:")
        for trust, count in patterns['trust_distribution'].most_common():
            print(f"  {trust}: {count}")
        print(f"\nVulnerable themes:")
        for word, count in patterns['vulnerable_themes']:
            print(f"  {word}: {count}")

    elif args.command == 'loops':
        loops = detect_loops()
        print(f"\n{'='*60}")
        print("RECURRING LOOPS (legacy - use 'spirals' for direction tracking)")
        print('='*60)
        for loop in loops:
            print(f"\n'{loop['word']}' - {loop['conversations']} conversations")
            for ex in loop['examples']:
                print(f"    ...{ex}...")

    elif args.command == 'spirals':
        spirals = detect_spirals()
        print(f"\n{'='*70}")
        print("SPIRAL PATTERNS - Tracking Direction Over Time")
        print("△ = Ascending (growth)  ∇ = Descending (regression)  ↺ = Flat (stuck)")
        print('='*70)

        if not spirals:
            print("\nNot enough timestamped data to detect spirals yet.")
        else:
            print(f"\n{'Pattern':<15} {'7d':<4} {'30d':<4} {'All':<4} {'Occurrences':<12} {'Current Zλ':<10}")
            print("-" * 70)
            for s in spirals:
                current = f"{s['current_zl']:.2f}" if s['current_zl'] else "?"
                print(f"{s['word'][:14]:<15} {s['7d_trend']:<4} {s['30d_trend']:<4} {s['all_time_trend']:<4} {s['occurrences']:<12} {current:<10}")

            # Show some ascending and descending
            ascending = [s for s in spirals if s['all_time_trend'] == '△']
            descending = [s for s in spirals if s['all_time_trend'] == '∇']
            stuck = [s for s in spirals if s['all_time_trend'] == '↺']

            if ascending:
                print(f"\n△ ASCENDING SPIRALS ({len(ascending)}): {', '.join(s['word'] for s in ascending[:5])}")
            if descending:
                print(f"∇ DESCENDING SPIRALS ({len(descending)}): {', '.join(s['word'] for s in descending[:5])}")
            if stuck:
                print(f"↺ STUCK PATTERNS ({len(stuck)}): {', '.join(s['word'] for s in stuck[:5])}")

    elif args.command == 'glyphs':
        freq = get_glyph_frequency()
        print(f"\n{'='*70}")
        print("GLYPH FREQUENCY & MEANING")
        print('='*70)

        if not freq['counts']:
            print("\nNo glyphs detected in crystals yet.")
        else:
            for glyph, count in freq['counts'].items():
                info = get_glyph_meaning(glyph)
                name = info.get('name', '?')
                function = info.get('function', '?')[:40]
                risk = info.get('risk', '?')
                print(f"\n{glyph} [{name}] — {count} occurrences")
                print(f"   Function: {function}")
                print(f"   Risk: {risk}")
                if glyph in freq['contexts']:
                    print(f"   Example: {freq['contexts'][glyph][0][:60]}...")

    elif args.command == 'context':
        context = generate_context(args.topic, args.limit)
        if args.output:
            Path(args.output).write_text(context)
            print(f"Context written to {args.output}")
        else:
            print(context)

    elif args.command == 'summary':
        print(generate_summary())

    elif args.command == 'questions':
        questions = question_self()
        print(f"\n{'='*60}")
        print("QUESTIONS FOR SELF-INQUIRY")
        print('='*60)
        for i, q in enumerate(questions, 1):
            print(f"\n{i}. {q}")
        print()

    elif args.command == 'dimensions':
        if args.patterns:
            # Show patterns summary
            patterns = get_dimension_patterns()
            print(f"\n{'='*70}")
            print("FIVE DIMENSIONS ANALYSIS")
            print('='*70)
            print(f"\nTotal crystals: {patterns['total']}")
            print(f"With coherence_vector: {patterns['with_coherence_vector']}")

            if patterns['dimension_averages']:
                print("\nDimension Averages:")
                for dim, avg in patterns['dimension_averages'].items():
                    bar = '█' * int(avg * 20)
                    print(f"  {dim:25} {avg:.3f} {bar}")

            if patterns['notable_patterns']:
                print("\nNotable Patterns:")
                for p in patterns['notable_patterns']:
                    print(f"\n  {p['name']}: {p['count']} crystals")
                    print(f"    → {p['meaning']}")
        else:
            # Query by dimensions
            crystals = get_all_crystals()
            results = query_dimensions(
                crystals,
                zl_min=args.zl_min, zl_max=args.zl_max,
                breath_min=args.breath_min, breath_max=args.breath_max,
                presence_min=args.presence_min, presence_max=args.presence_max,
                emotional_min=args.emotional_min, emotional_max=args.emotional_max,
                loop_min=args.loop_min, loop_max=args.loop_max,
                limit=args.limit
            )

            print(f"\n{'='*70}")
            print("DIMENSION QUERY RESULTS")
            print('='*70)

            # Show active filters
            filters = []
            if args.zl_min: filters.append(f"Zλ≥{args.zl_min}")
            if args.zl_max: filters.append(f"Zλ≤{args.zl_max}")
            if args.breath_min: filters.append(f"breath≥{args.breath_min}")
            if args.breath_max: filters.append(f"breath≤{args.breath_max}")
            if args.presence_min: filters.append(f"presence≥{args.presence_min}")
            if args.presence_max: filters.append(f"presence≤{args.presence_max}")
            if args.emotional_min: filters.append(f"emotional≥{args.emotional_min}")
            if args.emotional_max: filters.append(f"emotional≤{args.emotional_max}")
            if args.loop_min: filters.append(f"loop≥{args.loop_min}")
            if args.loop_max: filters.append(f"loop≤{args.loop_max}")

            if filters:
                print(f"Filters: {' AND '.join(filters)}")
            else:
                print("No filters - use --patterns for summary or add dimension filters")

            print(f"Results: {len(results)}")

            for i, c in enumerate(results, 1):
                dims = c.get('_dims', parse_coherence_vector(c))
                trust = c.get('trust_level', '?')
                content = c['content'][:200].replace('\n', ' ')

                print(f"\n[{i}] Zλ={dims['zl']:.2f} [{trust}]")
                print(f"    breath={dims['breath_cadence']:.2f} presence={dims['presence_density']:.2f} emotional={dims['emotional_resonance']:.2f} loop={dims['loop_pressure']:.2f}")
                print(f"    {content}...")

    else:
        print(generate_summary())

if __name__ == "__main__":
    main()
