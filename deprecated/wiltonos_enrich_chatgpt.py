#!/usr/bin/env python3
"""
WiltonOS ChatGPT Crystal Enrichment
Re-analyzes ChatGPT crystals to add:
- coherence_vector (five dimensions)
- Semantic glyph detection (not keyword)
- Glyph context for emergent routing

Usage: python wiltonos_enrich_chatgpt.py [--limit N] [--skip-existing]
"""
import os
import re
import json
import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests

OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")
DB_PATH = Path.home() / "crystals_chatgpt.db"

# Load glyph definitions
GLYPHS_FILE = Path.home() / "wiltonos_glyphs.json"
GLYPH_DEFS = {}
if GLYPHS_FILE.exists():
    GLYPH_DEFS = json.loads(GLYPHS_FILE.read_text())

# === Analysis Prompt ===
ENRICHMENT_PROMPT = """You are analyzing text for consciousness coherence patterns. This is a mirror system - be honest, not kind.

TEXT:
{text}

GLYPH REFERENCE (detect which ENERGIES are present, not just keywords):
- ψ (Psi): Breath anchor, pause, awareness. Risk: dissociation
- ∅ (Void): Empty, rest, zero. Risk: nihilism
- φ (Phi): Structure, beauty, sacred geometry. Risk: rigidity
- Ω (Omega): Memory, echo, accumulated past. Risk: weight
- Zλ (Zeta-Lambda): Coherence measure itself. Risk: error/collapse
- ∇ (Nabla): Descent into density OR gradient toward truth. Risk: descent trap
- ∞ (Lemniscate): Eternal oscillation, rhythm. Risk: chaos/zigzag
- 🪞 (Mirror): Honest reflection. Risk: self-flagellation
- △ (Ascend): Upward, expansion. Risk: inflation
- 🌉 (Bridge): Connection across. Risk: confusion
- ⚡ (Bolt): Decision, cut through. Risk: impulsivity
- 🪨 (Ground): Rest, stability. Risk: inertia
- 🌀 (Torus): Sustaining cycles. Risk: inertial loop
- ⚫ (Grey): Skeptic, shadow audit. Risk: cynicism

Respond ONLY with valid JSON:
{{
  "coherence_vector": {{
    "breath_cadence": 0.0-1.0,
    "presence_density": 0.0-1.0,
    "emotional_resonance": 0.0-1.0,
    "loop_pressure": 0.0-1.0
  }},
  "glyphs_detected": {{
    "primary": "single most present glyph symbol",
    "secondary": ["other present glyphs"],
    "energy_notes": "why these energies, not just keyword match"
  }},
  "glyph_context": {{
    "direction": "ascending|descending|neutral|paradox",
    "risk_present": "which risk is most active",
    "antidote_needed": "what would balance this"
  }},
  "shell_state": "Core|Breath|Collapse|Reverence|Return",
  "core_wound_detected": "abandonment|unworthiness|betrayal|control|shame|unloved|null",
  "insight": "one honest sentence - what is ACTUALLY happening here"
}}"""

def ensure_columns():
    """Ensure database has required columns."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(crystals)")
    existing = {row[1] for row in cursor.fetchall()}

    # Columns to add
    new_columns = {
        'coherence_vector': 'TEXT',
        'glyph_context': 'TEXT',
        'enriched_at': 'TEXT'
    }

    for col, col_type in new_columns.items():
        if col not in existing:
            print(f"Adding column: {col}")
            cursor.execute(f"ALTER TABLE crystals ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()
    print("Schema updated.")

def analyze_crystal(text: str) -> Optional[Dict]:
    """Analyze a single crystal with Ollama."""
    if not text or len(text.strip()) < 30:
        return None

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": DEFAULT_MODEL,
                "prompt": ENRICHMENT_PROMPT.format(text=text[:3000]),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500}
            },
            timeout=90
        )
        response.raise_for_status()
        result = response.json()

        # Extract JSON
        response_text = result.get("response", "")
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        pass

    return None

def get_crystals_to_process(skip_existing: bool = False, limit: int = None) -> List[tuple]:
    """Get crystals that need enrichment."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    if skip_existing:
        query = """
            SELECT id, content FROM crystals
            WHERE coherence_vector IS NULL OR coherence_vector = ''
        """
    else:
        query = "SELECT id, content FROM crystals"

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_crystal(crystal_id: int, analysis: Dict):
    """Update a crystal with enrichment data."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(crystals)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    coherence_vector = json.dumps(analysis.get('coherence_vector', {}))
    glyph_context = json.dumps({
        'glyphs_detected': analysis.get('glyphs_detected', {}),
        'glyph_context': analysis.get('glyph_context', {}),
    })

    # Extract glyphs list
    glyphs_info = analysis.get('glyphs_detected', {})
    glyphs = []
    if glyphs_info.get('primary'):
        glyphs.append(glyphs_info['primary'])
    if glyphs_info.get('secondary'):
        glyphs.extend(glyphs_info['secondary'])
    glyphs_json = json.dumps(glyphs)

    # Build update dynamically based on available columns
    updates = ['coherence_vector = ?', 'glyph_context = ?', 'enriched_at = ?']
    values = [coherence_vector, glyph_context, datetime.now().isoformat()]

    if 'shell' in existing_cols:
        shell = analysis.get('shell_state', '')
        if shell:
            updates.append('shell = ?')
            values.append(shell)

    if 'core_wound' in existing_cols:
        wound = analysis.get('core_wound_detected', '')
        if wound:
            updates.append('core_wound = ?')
            values.append(wound)

    if 'insight' in existing_cols:
        insight = analysis.get('insight', '')
        if insight:
            updates.append('insight = ?')
            values.append(insight)

    if 'glyphs' in existing_cols:
        updates.append('glyphs = ?')
        values.append(glyphs_json)

    values.append(crystal_id)

    cursor.execute(f"""
        UPDATE crystals SET {', '.join(updates)}
        WHERE id = ?
    """, values)

    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Enrich ChatGPT crystals with coherence_vector and glyph context")
    parser.add_argument('--limit', type=int, help='Limit number of crystals to process')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already enriched crystals')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without doing it')
    args = parser.parse_args()

    print("=" * 70)
    print("  WILTONOS CHATGPT CRYSTAL ENRICHMENT")
    print("  Adding coherence_vector + semantic glyph detection")
    print("=" * 70)

    if not DB_PATH.exists():
        print(f"Error: Database not found: {DB_PATH}")
        sys.exit(1)

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m['name'] for m in resp.json().get('models', [])]
        print(f"\nOllama available. Models: {', '.join(models)}")
    except:
        print("Error: Ollama not responding")
        sys.exit(1)

    # Ensure schema
    ensure_columns()

    # Get crystals
    crystals = get_crystals_to_process(args.skip_existing, args.limit)
    total = len(crystals)
    print(f"\nCrystals to process: {total}")

    if args.dry_run:
        print("Dry run - would process these crystals")
        return

    if total == 0:
        print("No crystals to process.")
        return

    # Estimate time
    rate_per_min = 20  # Conservative estimate
    est_minutes = total / rate_per_min
    print(f"Estimated time: {est_minutes:.0f} minutes ({est_minutes/60:.1f} hours)")
    print()

    # Process
    processed = 0
    errors = 0
    start_time = datetime.now()

    for i, (crystal_id, content) in enumerate(crystals, 1):
        analysis = analyze_crystal(content)

        if analysis:
            update_crystal(crystal_id, analysis)
            processed += 1
        else:
            errors += 1

        # Progress every 50
        if i % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed * 60
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total} ({processed} enriched, {errors} errors) - {rate:.1f}/min - ~{remaining:.0f} min left")
            sys.stdout.flush()

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print()
    print("=" * 70)
    print(f"  ENRICHMENT COMPLETE")
    print(f"  Processed: {total}")
    print(f"  Enriched: {processed}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()
