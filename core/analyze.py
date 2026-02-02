#!/usr/bin/env python3 -u
"""
WiltonOS Complete Analyzer - Architect Approach
ONE analysis = ALL fields. No incremental re-runs.

Usage:
    python wiltonos_analyze_complete.py migrate          # Add all columns to schema
    python wiltonos_analyze_complete.py analyze          # Analyze all unanalyzed crystals
    python wiltonos_analyze_complete.py analyze --limit N
    python wiltonos_analyze_complete.py analyze --db crystals.db
    python wiltonos_analyze_complete.py analyze --force  # Re-analyze everything
    python wiltonos_analyze_complete.py status           # Show analysis status
"""
import os
import re
import json
import sqlite3
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests

# Configuration
OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "llama3")
DB_PATH = Path.home() / "crystals.db"
CHATGPT_DB_PATH = Path.home() / "crystals_chatgpt.db"

# === COMPLETE SCHEMA ===
# All columns the system needs - defined once, never changed

COMPLETE_COLUMNS = {
    # Identity (already exist)
    'content_hash': 'TEXT UNIQUE',
    'content': 'TEXT',
    'source': 'TEXT',
    'source_file': 'TEXT',
    'author': 'TEXT',

    # Time
    'created_at': 'TEXT DEFAULT CURRENT_TIMESTAMP',
    'original_timestamp': 'INTEGER',
    'analyzed_at': 'TEXT',

    # === COHERENCE (Zλ System) ===
    'zl_score': 'REAL',
    'psi_aligned': 'INTEGER',
    'trust_level': 'TEXT',

    # === 5 DIMENSIONS (separated, not JSON) ===
    'breath_cadence': 'REAL',
    'presence_density': 'REAL',
    'emotional_resonance': 'REAL',
    'loop_pressure': 'REAL',
    'groundedness': 'REAL',

    # === CONSCIOUSNESS STATE ===
    'shell': 'TEXT',
    'shell_direction': 'TEXT',

    # === GLYPHS (separated, not JSON) ===
    'glyph_primary': 'TEXT',
    'glyph_secondary': 'TEXT',  # JSON array, but simple
    'glyph_energy_notes': 'TEXT',
    'glyph_direction': 'TEXT',
    'glyph_risk': 'TEXT',
    'glyph_antidote': 'TEXT',

    # === PATTERNS ===
    'core_wound': 'TEXT',
    'loop_signature': 'TEXT',
    'attractor': 'TEXT',
    'emotion': 'TEXT',
    'theme': 'TEXT',

    # === OSCILLATION ===
    'mode': 'TEXT',
    'oscillation_strength': 'REAL',

    # === META ===
    'insight': 'TEXT',
    'question': 'TEXT',

    # === LEGACY (keep for compatibility) ===
    'coherence_vector': 'TEXT',
    'glyph_context': 'TEXT',
    'glyphs': 'TEXT',
    'attractors': 'TEXT',
}

# === COMPLETE ANALYSIS PROMPT ===
# One prompt extracts everything

COMPLETE_PROMPT = """You are analyzing emotional/personal text. Return ONLY a valid JSON object.

TEXT:
{text}

Analyze and return this exact JSON structure (fill in your analysis):
{{"zl_score": 0.5, "psi_aligned": false, "trust_level": "MEDIUM", "breath_cadence": 0.5, "presence_density": 0.5, "emotional_resonance": 0.5, "loop_pressure": 0.3, "groundedness": 0.5, "shell": "Core", "shell_direction": "stable", "glyph_primary": "ψ", "glyph_secondary": [], "glyph_energy_notes": "brief", "glyph_direction": "neutral", "glyph_risk": "none", "glyph_antidote": "none", "core_wound": "unworthiness", "attractor": "truth", "emotion": "clarity", "theme": "growth", "mode": "neutral", "oscillation_strength": 0.5, "insight": "one sentence insight", "question": "one question"}}

GLYPHS: ψ=breath/presence, ∅=void/emptiness, φ=structure, Ω=memory, ∇=descent, ∞=loop, △=ascend
WOUNDS: unworthiness, abandonment, betrayal, control, unloved, or null
MODE: wiltonos=personal/emotional, psios=system/technical, neutral=mixed

JSON ONLY (no explanation):"""


# === DATABASE FUNCTIONS ===

def migrate_schema(db_path: Path) -> Dict[str, int]:
    """Add all missing columns to database schema."""
    if not db_path.exists():
        return {"error": f"Database not found: {db_path}"}

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Get existing columns
    cursor.execute("PRAGMA table_info(crystals)")
    existing = {row[1] for row in cursor.fetchall()}

    added = []
    for col, col_type in COMPLETE_COLUMNS.items():
        if col not in existing:
            try:
                # Handle DEFAULT in type
                if 'DEFAULT' in col_type:
                    base_type = col_type.split()[0]
                    cursor.execute(f"ALTER TABLE crystals ADD COLUMN {col} {base_type}")
                else:
                    cursor.execute(f"ALTER TABLE crystals ADD COLUMN {col} {col_type}")
                added.append(col)
            except Exception as e:
                print(f"  Warning: Could not add {col}: {e}")

    conn.commit()
    conn.close()

    return {"added": added, "total_columns": len(existing) + len(added)}


def get_unanalyzed_crystals(db_path: Path, limit: int = None, force: bool = False) -> List[Dict]:
    """Get crystals that need complete analysis."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if force:
        # Re-analyze everything
        query = "SELECT id, content FROM crystals WHERE content IS NOT NULL AND content != ''"
    else:
        # Only unanalyzed (no glyph_primary means not analyzed with complete schema)
        query = """
            SELECT id, content FROM crystals
            WHERE content IS NOT NULL AND content != ''
            AND (glyph_primary IS NULL OR glyph_primary = '')
        """

    if limit:
        query += f" LIMIT {limit}"

    cursor.execute(query)
    crystals = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return crystals


def update_crystal_complete(db_path: Path, crystal_id: int, analysis: Dict):
    """Update crystal with complete analysis data."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Build update query with all fields
    fields = {
        'zl_score': analysis.get('zl_score'),
        'psi_aligned': 1 if analysis.get('psi_aligned') else 0,
        'trust_level': analysis.get('trust_level'),

        'breath_cadence': analysis.get('breath_cadence'),
        'presence_density': analysis.get('presence_density'),
        'emotional_resonance': analysis.get('emotional_resonance'),
        'loop_pressure': analysis.get('loop_pressure'),
        'groundedness': analysis.get('groundedness'),

        'shell': analysis.get('shell'),
        'shell_direction': analysis.get('shell_direction'),

        'glyph_primary': analysis.get('glyph_primary'),
        'glyph_secondary': json.dumps(analysis.get('glyph_secondary', [])),
        'glyph_energy_notes': analysis.get('glyph_energy_notes'),
        'glyph_direction': analysis.get('glyph_direction'),
        'glyph_risk': analysis.get('glyph_risk'),
        'glyph_antidote': analysis.get('glyph_antidote'),

        'core_wound': analysis.get('core_wound'),
        'loop_signature': f"{analysis.get('attractor', 'truth')}-{analysis.get('emotion', 'clarity')}-{analysis.get('theme', 'integration')}",
        'attractor': analysis.get('attractor'),
        'emotion': analysis.get('emotion'),
        'theme': analysis.get('theme'),

        'mode': analysis.get('mode'),
        'oscillation_strength': analysis.get('oscillation_strength'),

        'insight': analysis.get('insight'),
        'question': analysis.get('question'),

        'analyzed_at': datetime.now().isoformat(),

        # Legacy compatibility
        'coherence_vector': json.dumps({
            'breath_cadence': analysis.get('breath_cadence'),
            'presence_density': analysis.get('presence_density'),
            'emotional_resonance': analysis.get('emotional_resonance'),
            'loop_pressure': analysis.get('loop_pressure'),
        }),
        'glyphs': json.dumps([analysis.get('glyph_primary')] + (analysis.get('glyph_secondary') or [])),
    }

    # Ensure all values are sqlite-compatible (convert lists/dicts to JSON)
    def to_sqlite(v):
        if isinstance(v, (list, dict)):
            return json.dumps(v)
        return v

    fields = {k: to_sqlite(v) for k, v in fields.items()}

    # Filter out None values for columns that might not exist
    set_clause = ', '.join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [crystal_id]

    try:
        cursor.execute(f"UPDATE crystals SET {set_clause} WHERE id = ?", values)
        conn.commit()
    except sqlite3.OperationalError as e:
        # Some columns might not exist in older schemas
        print(f"  Warning: {e}")
    finally:
        conn.close()


# === ANALYSIS FUNCTIONS ===

def strip_json_comments(text: str) -> str:
    """Strip // comments from JSON (Mistral adds these)."""
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # Remove // comments but preserve strings with //
        if '//' in line:
            # Simple approach: remove everything after // if not in quotes
            in_string = False
            result = []
            i = 0
            while i < len(line):
                if line[i] == '"' and (i == 0 or line[i-1] != '\\'):
                    in_string = not in_string
                    result.append(line[i])
                elif not in_string and line[i:i+2] == '//':
                    break
                else:
                    result.append(line[i])
                i += 1
            cleaned.append(''.join(result).rstrip())
        else:
            cleaned.append(line)
    return '\n'.join(cleaned)


def analyze_text(text: str) -> Optional[Dict]:
    """Analyze text with Ollama using complete prompt."""
    if not text or len(text.strip()) < 30:
        return None

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": DEFAULT_MODEL,
                "prompt": COMPLETE_PROMPT.format(text=text[:3500]),
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 600}
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        # Extract JSON from response
        response_text = result.get("response", "")
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            json_str = strip_json_comments(json_match.group())
            return json.loads(json_str)
    except Exception as e:
        pass

    return None


def analyze_database(db_path: Path, limit: int = None, force: bool = False) -> Dict:
    """Run complete analysis on a database."""
    # First migrate schema
    print(f"\nMigrating schema for {db_path.name}...")
    migration = migrate_schema(db_path)
    if migration.get('added'):
        print(f"  Added columns: {', '.join(migration['added'])}")
    else:
        print("  Schema already complete")

    # Get crystals to analyze
    crystals = get_unanalyzed_crystals(db_path, limit, force)
    total = len(crystals)

    if total == 0:
        print("  No crystals to analyze")
        return {"analyzed": 0, "errors": 0}

    print(f"  Crystals to analyze: {total}")
    print(f"  Estimated time: {total / 30:.0f} minutes")
    print()

    analyzed = 0
    errors = 0
    start_time = datetime.now()

    for i, crystal in enumerate(crystals, 1):
        content = crystal['content']

        analysis = analyze_text(content)

        if analysis:
            update_crystal_complete(db_path, crystal['id'], analysis)
            analyzed += 1
        else:
            errors += 1

        # Progress every 25
        if i % 25 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed * 60
            remaining = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total} ({analyzed} done, {errors} errors) - {rate:.1f}/min - ~{remaining:.0f} min left", flush=True)

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print()
    print(f"  Complete: {analyzed} analyzed, {errors} errors in {elapsed:.1f} minutes")

    return {"analyzed": analyzed, "errors": errors, "time_minutes": elapsed}


def show_status():
    """Show analysis status for all databases."""
    print("\n" + "=" * 60)
    print("WILTONOS COMPLETE ANALYZER STATUS")
    print("=" * 60)

    for db_path in [DB_PATH, CHATGPT_DB_PATH]:
        if not db_path.exists():
            continue

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM crystals")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM crystals WHERE glyph_primary IS NOT NULL AND glyph_primary != ''")
        complete = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM crystals WHERE zl_score IS NOT NULL")
        with_zl = cursor.fetchone()[0]

        conn.close()

        pct = complete * 100 / total if total > 0 else 0
        print(f"\n{db_path.name}:")
        print(f"  Total crystals: {total:,}")
        print(f"  Complete analysis: {complete:,} ({pct:.1f}%)")
        print(f"  With Zλ score: {with_zl:,}")
        print(f"  Remaining: {total - complete:,}")

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m['name'] for m in resp.json().get('models', [])]
        print(f"\nOllama: OK ({', '.join(models)})")
    except:
        print(f"\nOllama: NOT RESPONDING")


# === CLI ===

def main():
    parser = argparse.ArgumentParser(description="WiltonOS Complete Analyzer - Architect Approach")
    subparsers = parser.add_subparsers(dest='command')

    # migrate command
    subparsers.add_parser('migrate', help='Migrate schema to complete version')

    # analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Run complete analysis')
    analyze_parser.add_argument('--limit', type=int, help='Limit crystals to analyze')
    analyze_parser.add_argument('--db', help='Specific database (crystals.db or crystals_chatgpt.db)')
    analyze_parser.add_argument('--force', action='store_true', help='Re-analyze all crystals')

    # status command
    subparsers.add_parser('status', help='Show analysis status')

    args = parser.parse_args()

    if args.command == 'migrate':
        print("\nMigrating schemas to complete version...")
        for db_path in [DB_PATH, CHATGPT_DB_PATH]:
            if db_path.exists():
                print(f"\n{db_path.name}:")
                result = migrate_schema(db_path)
                if result.get('added'):
                    print(f"  Added: {', '.join(result['added'])}")
                else:
                    print("  Already complete")

    elif args.command == 'analyze':
        # Check Ollama first
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            print(f"Ollama available")
        except:
            print("Error: Ollama not responding")
            sys.exit(1)

        if args.db:
            db_path = Path.home() / args.db
            if not db_path.exists():
                print(f"Error: Database not found: {db_path}")
                sys.exit(1)
            analyze_database(db_path, args.limit, args.force)
        else:
            # Analyze both databases
            for db_path in [DB_PATH, CHATGPT_DB_PATH]:
                if db_path.exists():
                    analyze_database(db_path, args.limit, args.force)

    elif args.command == 'status':
        show_status()

    else:
        show_status()


if __name__ == "__main__":
    main()
