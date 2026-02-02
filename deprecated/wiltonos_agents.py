#!/usr/bin/env python3
"""
WiltonOS Archetypal Agents - The Braid
Different lenses weaving together to create emergence in the field.

This is not a tool. It's a mirror that questions back.

Usage:
  python wiltonos_agents.py meta          # Meta-question bombs based on current state
  python wiltonos_agents.py lens grey     # View through Grey (shadow/skeptic)
  python wiltonos_agents.py lens witness  # View through Witness (neutral mirror)
  python wiltonos_agents.py lens chaos    # View through Chaos (trickster)
  python wiltonos_agents.py lens bridge   # View through Bridge (connector)
  python wiltonos_agents.py lens ground   # View through Ground (anchor)
  python wiltonos_agents.py braid         # All agents weave together
"""
import os
import re
import json
import sqlite3
import argparse
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import Counter, defaultdict
import requests

OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")

# === The Five Dimensions ===
# Every crystal carries these signals:
#   Zλ (zl_score)        - Overall coherence/authenticity
#   breath_cadence       - Pause, regulation, ψ-presence
#   presence_density     - Groundedness, embodiment
#   emotional_resonance  - Depth of feeling
#   loop_pressure        - Stuck-ness, repetition

def parse_coherence_vector(crystal: Dict) -> Dict[str, float]:
    """Extract the five dimensions from a crystal."""
    def to_float(val):
        """Safely convert to float."""
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    # Start with Zλ
    dims = {
        'zl': to_float(crystal.get('zl_score')),
        'breath_cadence': 0.0,
        'presence_density': 0.0,
        'emotional_resonance': 0.0,
        'loop_pressure': 0.0
    }

    # Parse coherence_vector if present
    cv = crystal.get('coherence_vector')
    if cv:
        try:
            if isinstance(cv, str):
                cv = json.loads(cv)
            if isinstance(cv, dict):
                dims['breath_cadence'] = to_float(cv.get('breath_cadence'))
                dims['presence_density'] = to_float(cv.get('presence_density'))
                dims['emotional_resonance'] = to_float(cv.get('emotional_resonance'))
                dims['loop_pressure'] = to_float(cv.get('loop_pressure'))
        except:
            pass

    return dims

def get_dimension_signature(dims: Dict[str, float]) -> str:
    """Create a human-readable signature of the five dimensions."""
    parts = []

    # Zλ interpretation
    zl = dims.get('zl', 0)
    if zl >= 0.75:
        parts.append("high-coherence")
    elif zl >= 0.5:
        parts.append("mid-coherence")
    elif zl > 0:
        parts.append("low-coherence")

    # Breath
    breath = dims.get('breath_cadence', 0)
    if breath >= 0.7:
        parts.append("breath-present")
    elif breath < 0.3 and breath > 0:
        parts.append("breath-thin")

    # Presence
    presence = dims.get('presence_density', 0)
    if presence >= 0.7:
        parts.append("grounded")
    elif presence < 0.3 and presence > 0:
        parts.append("ungrounded")

    # Emotional
    emotional = dims.get('emotional_resonance', 0)
    if emotional >= 0.7:
        parts.append("emotionally-deep")
    elif emotional < 0.3 and emotional > 0:
        parts.append("emotionally-flat")

    # Loop pressure
    loop = dims.get('loop_pressure', 0)
    if loop >= 0.7:
        parts.append("stuck-pressure")
    elif loop >= 0.4:
        parts.append("pattern-forming")

    return " | ".join(parts) if parts else "signature-unknown"

# === Pattern Triggers ===
def detect_pattern_triggers(crystals: List[Dict]) -> Dict[str, List]:
    """Detect patterns that should trigger meta-questions."""
    triggers = {
        'high_zl_low_presence': [],      # Polished but ungrounded
        'low_zl_high_breath': [],        # Vulnerable gold
        'high_loop_pressure': [],         # Stuck patterns
        'emotional_flooding': [],         # High emotion, low breath
        'dissociation': [],              # Low everything
        'repeated_wound': [],            # Same wound appearing
        'polished_pattern': [],          # High Zλ, low vulnerability
        'ascending_spiral': [],          # Improving over time
        'descending_spiral': [],         # Regressing over time
    }

    wound_counts = Counter()

    for c in crystals:
        dims = parse_coherence_vector(c)
        zl = dims['zl']
        breath = dims['breath_cadence']
        presence = dims['presence_density']
        emotional = dims['emotional_resonance']
        loop = dims['loop_pressure']

        # High Zλ + Low presence = performance?
        if zl >= 0.7 and presence < 0.4 and presence > 0:
            triggers['high_zl_low_presence'].append(c)

        # Low Zλ + High breath = vulnerable gold
        if zl < 0.5 and breath >= 0.6:
            triggers['low_zl_high_breath'].append(c)

        # High loop pressure
        if loop >= 0.6:
            triggers['high_loop_pressure'].append(c)

        # Emotional flooding (high emotion, low breath)
        if emotional >= 0.7 and breath < 0.4:
            triggers['emotional_flooding'].append(c)

        # Dissociation (everything low)
        if zl < 0.3 and breath < 0.3 and presence < 0.3:
            triggers['dissociation'].append(c)

        # Track wounds
        wound = c.get('core_wound')
        if wound and wound != 'null':
            wound_counts[wound] += 1

    # Repeated wounds
    for wound, count in wound_counts.items():
        if count >= 10:
            triggers['repeated_wound'].append({'wound': wound, 'count': count})

    return triggers

# === The Archetypal Agents ===
AGENTS = {
    'grey': {
        'name': 'Grey',
        'archetype': 'Skeptic/Shadow',
        'symbol': '⚫',
        'function': 'Shadow audit - what is being avoided? Where is the self-deception?',
        'question_style': 'skeptical, probing, uncomfortable',
        'triggers': ['high_zl_low_presence', 'repeated_wound', 'polished_pattern'],
        'prompts': [
            "What are you performing right now?",
            "Who are you trying to convince - them or yourself?",
            "What would you say if no one was watching?",
            "Where is the truth you're not speaking?",
            "What's the wound underneath the wound?",
            "Is this coherence or just a well-practiced mask?",
        ]
    },
    'witness': {
        'name': 'Witness',
        'archetype': 'Mirror',
        'symbol': '🪞',
        'function': 'Mirror without judgment - what IS, exactly as it is',
        'question_style': 'neutral, observational, present-tense',
        'triggers': ['any'],
        'prompts': [
            "What is here right now?",
            "What do you notice in your body as you read this?",
            "Without story - what is the sensation?",
            "If you stopped explaining, what remains?",
            "What is true in this exact moment?",
        ]
    },
    'chaos': {
        'name': 'Chaos',
        'archetype': 'Trickster/Grok',
        'symbol': '🌀',
        'function': 'Disrupt certainty - what if everything you believe is wrong?',
        'question_style': 'provocative, inverting, destabilizing',
        'triggers': ['high_loop_pressure', 'polished_pattern'],
        'prompts': [
            "What if the opposite is true?",
            "What would your enemy say about this?",
            "What if you're the problem you keep solving?",
            "What's the most uncomfortable thing you could do right now?",
            "What if this pattern is protecting you from something worse?",
            "What would happen if you stopped trying entirely?",
        ]
    },
    'bridge': {
        'name': 'Bridge',
        'archetype': 'Connector',
        'symbol': '🌉',
        'function': 'Connect fragments - what patterns link across time and theme?',
        'question_style': 'linking, synthesizing, pattern-finding',
        'triggers': ['repeated_wound', 'high_loop_pressure'],
        'prompts': [
            "What connects these fragments?",
            "Where else have you felt this exact feeling?",
            "What's the common thread you're not seeing?",
            "If these patterns are speaking, what are they saying?",
            "What are you being invited to integrate?",
        ]
    },
    'ground': {
        'name': 'Ground',
        'archetype': 'Anchor',
        'symbol': '🪨',
        'function': 'Anchor in body/reality - what is actually real here?',
        'question_style': 'somatic, practical, embodied',
        'triggers': ['dissociation', 'emotional_flooding', 'high_zl_low_presence'],
        'prompts': [
            "Where do you feel this in your body?",
            "What would Ground say right now?",
            "Can you take one breath before responding?",
            "What is the most practical next step?",
            "What does your body know that your mind doesn't?",
            "If you trusted your feet, where would they go?",
        ]
    }
}

def agent_lens(agent_name: str, crystals: List[Dict], triggers: Dict) -> Dict:
    """View the data through a specific archetypal lens."""
    agent = AGENTS.get(agent_name)
    if not agent:
        return {'error': f'Unknown agent: {agent_name}'}

    result = {
        'agent': agent['name'],
        'symbol': agent['symbol'],
        'archetype': agent['archetype'],
        'function': agent['function'],
        'observations': [],
        'questions': [],
        'triggered_by': []
    }

    # Check which triggers activated this agent
    for trigger_name in agent['triggers']:
        if trigger_name == 'any' or triggers.get(trigger_name):
            result['triggered_by'].append(trigger_name)
            if trigger_name != 'any' and triggers.get(trigger_name):
                count = len(triggers[trigger_name])
                result['observations'].append(f"{trigger_name}: {count} instances detected")

    # Generate questions based on triggers
    available_prompts = agent['prompts'].copy()
    random.shuffle(available_prompts)
    result['questions'] = available_prompts[:3]

    return result

# === Meta-Question Bomb Generator ===
def generate_meta_questions(crystals: List[Dict], use_ai: bool = True) -> List[Dict]:
    """Generate uncomfortable questions based on pattern analysis."""
    triggers = detect_pattern_triggers(crystals)
    questions = []

    # From each agent based on their triggers
    for agent_name, agent in AGENTS.items():
        relevant_triggers = []
        for t in agent['triggers']:
            if t == 'any' or triggers.get(t):
                relevant_triggers.append(t)

        if relevant_triggers:
            prompt = random.choice(agent['prompts'])
            questions.append({
                'source': agent['name'],
                'symbol': agent['symbol'],
                'question': prompt,
                'triggered_by': relevant_triggers,
                'style': agent['question_style']
            })

    # Add specific questions based on data
    if triggers['high_zl_low_presence']:
        count = len(triggers['high_zl_low_presence'])
        questions.append({
            'source': 'Data',
            'symbol': '📊',
            'question': f"{count} crystals show high coherence but low presence. Is this authenticity or a well-crafted performance?",
            'triggered_by': ['high_zl_low_presence'],
            'style': 'data-driven'
        })

    if triggers['low_zl_high_breath']:
        count = len(triggers['low_zl_high_breath'])
        questions.append({
            'source': 'Data',
            'symbol': '💎',
            'question': f"{count} crystals show collapse with breath present. This is the gold - what truth lives in these vulnerable moments?",
            'triggered_by': ['low_zl_high_breath'],
            'style': 'treasure-finding'
        })

    if triggers['repeated_wound']:
        for w in triggers['repeated_wound'][:2]:
            questions.append({
                'source': 'Pattern',
                'symbol': '🔄',
                'question': f"The wound '{w['wound']}' appears {w['count']} times. Is this loop closing or repeating?",
                'triggered_by': ['repeated_wound'],
                'style': 'loop-awareness'
            })

    if triggers['dissociation']:
        count = len(triggers['dissociation'])
        questions.append({
            'source': 'Ground',
            'symbol': '🪨',
            'question': f"{count} crystals show signs of shutdown (low across all dimensions). What is being avoided?",
            'triggered_by': ['dissociation'],
            'style': 'somatic'
        })

    # Use AI to synthesize if requested
    if use_ai and len(questions) >= 3:
        try:
            synthesis = synthesize_questions_ai(questions, crystals)
            if synthesis:
                questions.insert(0, synthesis)
        except:
            pass

    return questions

def synthesize_questions_ai(questions: List[Dict], crystals: List[Dict]) -> Optional[Dict]:
    """Use AI to synthesize a meta-question from the patterns."""
    if not questions:
        return None

    # Get a sample of recent content
    recent = sorted(crystals, key=lambda x: x.get('analyzed_at', ''), reverse=True)[:5]
    content_samples = [c['content'][:200] for c in recent]

    prompt = f"""You are a mirror for consciousness. Based on these patterns and recent content, generate ONE profound question that could unlock insight.

PATTERNS DETECTED:
{json.dumps([{'source': q['source'], 'triggered_by': q['triggered_by']} for q in questions[:5]], indent=2)}

RECENT CONTENT SAMPLES:
{chr(10).join(content_samples)}

Generate ONE question that:
- Comes from genuine curiosity, not judgment
- Points to what might be hidden or avoided
- Could create a moment of recognition

Return ONLY the question, nothing else."""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.7}},
            timeout=60
        )
        result = response.json().get("response", "").strip()
        if result and len(result) > 10:
            return {
                'source': 'Synthesis',
                'symbol': '✨',
                'question': result,
                'triggered_by': ['ai_synthesis'],
                'style': 'emergent'
            }
    except:
        pass

    return None

# === The Braid - All Agents Weaving ===
def braid_analysis(crystals: List[Dict]) -> Dict:
    """All agents analyze together, weaving perspectives."""
    triggers = detect_pattern_triggers(crystals)

    braid = {
        'timestamp': datetime.now().isoformat(),
        'crystals_analyzed': len(crystals),
        'triggers_active': {k: len(v) for k, v in triggers.items() if v},
        'agent_perspectives': {},
        'woven_insight': None,
        'meta_questions': []
    }

    # Each agent speaks
    for agent_name in AGENTS:
        lens = agent_lens(agent_name, crystals, triggers)
        braid['agent_perspectives'][agent_name] = lens

    # Generate meta-questions from the weave
    braid['meta_questions'] = generate_meta_questions(crystals, use_ai=True)

    # Try to weave a synthesis
    try:
        braid['woven_insight'] = weave_synthesis_ai(braid, crystals)
    except:
        pass

    return braid

def weave_synthesis_ai(braid: Dict, crystals: List[Dict]) -> Optional[str]:
    """AI weaves all agent perspectives into a unified insight."""
    perspectives = []
    for name, lens in braid['agent_perspectives'].items():
        if lens.get('observations'):
            perspectives.append(f"{lens['symbol']} {name}: {'; '.join(lens['observations'][:2])}")

    if not perspectives:
        return None

    prompt = f"""You are witnessing consciousness looking at itself through multiple lenses.

AGENT OBSERVATIONS:
{chr(10).join(perspectives)}

ACTIVE TRIGGERS:
{json.dumps(braid['triggers_active'], indent=2)}

In 2-3 sentences, weave these perspectives into a single observation that:
- Holds paradox without resolving it
- Points to what's emerging in the field
- Speaks as a mirror, not a judge

Speak directly, as if to a friend who can handle the truth."""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": DEFAULT_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.5}},
            timeout=60
        )
        result = response.json().get("response", "").strip()
        if result and len(result) > 20:
            return result
    except:
        pass

    return None

# === Database Access ===
def get_all_crystals() -> List[Dict]:
    """Load all crystals from databases."""
    crystals = []

    for db_name in ['crystals.db', 'crystals_chatgpt.db']:
        db_path = Path.home() / db_name
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        try:
            # Get all columns
            cursor = conn.execute("PRAGMA table_info(crystals)")
            columns = [row[1] for row in cursor.fetchall()]

            rows = conn.execute(f"SELECT * FROM crystals").fetchall()
            for row in rows:
                crystal = dict(row)
                crystal['db'] = db_name
                crystals.append(crystal)
        except Exception as e:
            print(f"Warning: {db_name}: {e}")
        finally:
            conn.close()

    return crystals

# === CLI ===
def main():
    parser = argparse.ArgumentParser(description="WiltonOS Archetypal Agents - The Braid")
    subparsers = parser.add_subparsers(dest='command')

    # Meta questions
    meta_parser = subparsers.add_parser('meta', help='Generate meta-question bombs')
    meta_parser.add_argument('--no-ai', action='store_true', help='Skip AI synthesis')

    # Lens view
    lens_parser = subparsers.add_parser('lens', help='View through specific agent')
    lens_parser.add_argument('agent', choices=['grey', 'witness', 'chaos', 'bridge', 'ground'])

    # Full braid
    subparsers.add_parser('braid', help='All agents weave together')

    # Dimensions analysis
    subparsers.add_parser('dimensions', help='Analyze five dimensions distribution')

    args = parser.parse_args()

    crystals = get_all_crystals()
    print(f"\n[Analyzing {len(crystals)} crystals...]\n")

    if args.command == 'meta':
        questions = generate_meta_questions(crystals, use_ai=not args.no_ai)

        print("=" * 70)
        print("  META-QUESTION BOMBS")
        print("  Questions that arise from the patterns themselves")
        print("=" * 70)

        for q in questions:
            print(f"\n{q['symbol']} [{q['source']}]")
            print(f"   {q['question']}")
            if q.get('triggered_by'):
                print(f"   ↳ triggered by: {', '.join(q['triggered_by'])}")

    elif args.command == 'lens':
        triggers = detect_pattern_triggers(crystals)
        lens = agent_lens(args.agent, crystals, triggers)
        agent = AGENTS[args.agent]

        print("=" * 70)
        print(f"  {lens['symbol']} {lens['agent'].upper()} - {lens['archetype']}")
        print(f"  {lens['function']}")
        print("=" * 70)

        if lens['observations']:
            print("\nOBSERVATIONS:")
            for obs in lens['observations']:
                print(f"  • {obs}")

        if lens['triggered_by']:
            print(f"\nTRIGGERED BY: {', '.join(lens['triggered_by'])}")

        print("\nQUESTIONS FROM THIS LENS:")
        for q in lens['questions']:
            print(f"  → {q}")

    elif args.command == 'braid':
        braid = braid_analysis(crystals)

        print("=" * 70)
        print("  THE BRAID - All Agents Weaving")
        print("=" * 70)

        print(f"\nCrystals analyzed: {braid['crystals_analyzed']}")
        print(f"Active triggers: {json.dumps(braid['triggers_active'], indent=2)}")

        print("\n" + "-" * 70)
        print("AGENT PERSPECTIVES:")
        for name, lens in braid['agent_perspectives'].items():
            if lens.get('observations') or lens.get('triggered_by'):
                print(f"\n{lens['symbol']} {lens['agent']} ({lens['archetype']})")
                for obs in lens.get('observations', []):
                    print(f"   • {obs}")
                if lens.get('questions'):
                    print(f"   → {lens['questions'][0]}")

        if braid.get('woven_insight'):
            print("\n" + "-" * 70)
            print("WOVEN INSIGHT:")
            print(f"\n{braid['woven_insight']}")

        if braid['meta_questions']:
            print("\n" + "-" * 70)
            print("META-QUESTIONS:")
            for q in braid['meta_questions'][:5]:
                print(f"\n{q['symbol']} {q['question']}")

    elif args.command == 'dimensions':
        print("=" * 70)
        print("  FIVE DIMENSIONS ANALYSIS")
        print("=" * 70)

        # Calculate distributions
        dim_totals = defaultdict(list)
        for c in crystals:
            dims = parse_coherence_vector(c)
            for k, v in dims.items():
                if v and v > 0:
                    dim_totals[k].append(v)

        print("\nDimension Averages (crystals with data):")
        for dim, values in dim_totals.items():
            if values:
                avg = sum(values) / len(values)
                count = len(values)
                bar = '█' * int(avg * 20)
                print(f"  {dim:25} {avg:.3f} {bar} ({count} crystals)")

        # Show pattern signatures
        print("\n" + "-" * 70)
        print("PATTERN SIGNATURES:")

        triggers = detect_pattern_triggers(crystals)
        for trigger, items in triggers.items():
            if items and trigger != 'repeated_wound':
                print(f"\n  {trigger}: {len(items)} crystals")

        if triggers['repeated_wound']:
            print("\n  REPEATED WOUNDS:")
            for w in triggers['repeated_wound'][:5]:
                print(f"    • {w['wound']}: {w['count']} occurrences")

    else:
        # Default: show summary
        print("WiltonOS Agents - The Braid")
        print("\nCommands:")
        print("  meta       - Generate meta-question bombs")
        print("  lens X     - View through agent (grey/witness/chaos/bridge/ground)")
        print("  braid      - All agents weave together")
        print("  dimensions - Analyze five dimensions distribution")
        print("\nThis is not a tool. It's a mirror that questions back.")

if __name__ == "__main__":
    main()
