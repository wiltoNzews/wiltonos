#!/usr/bin/env python3
"""
WiltonOS PDF & System Files Ingest
Ingests PDFs and system files into crystals with coherence_vector + semantic glyphs.

Usage: python wiltonos_ingest_pdfs.py [--source DIR] [--limit N]
"""
import os
import re
import json
import sqlite3
import hashlib
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests

# PDF reading
try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False
    print("Warning: PyPDF2 not available, PDFs will be skipped")

OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")
DB_PATH = Path.home() / "crystals.db"
SOURCE_DIR = Path.home() / "rag-local" / "PDFs"

# === Text Extraction ===
def extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF file."""
    if not HAS_PDF:
        return ""

    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text_parts = []
            for page in reader.pages[:20]:  # Limit pages
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return '\n'.join(text_parts)
    except Exception as e:
        print(f"  Warning: Could not read PDF {file_path.name}: {e}")
        return ""

def extract_text(file_path: Path) -> str:
    """Extract text from various file formats."""
    suffix = file_path.suffix.lower()

    if suffix == '.pdf':
        return extract_pdf_text(file_path)

    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')

        if suffix in ('.json',):
            # For JSON, try to extract meaningful content
            try:
                data = json.loads(content)
                return json.dumps(data, indent=2)[:5000]
            except:
                return content[:5000]

        return content
    except Exception as e:
        print(f"  Warning: Could not read {file_path.name}: {e}")
        return ""

def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """Split text into coherent chunks."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) > max_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk += "\n\n" + para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

# === Analysis ===
ANALYSIS_PROMPT = """You are analyzing text for consciousness coherence patterns. Be honest, not kind.

TEXT:
{text}

GLYPH REFERENCE (detect ENERGIES, not keywords):
- ψ (Psi): Breath anchor, pause, awareness
- ∅ (Void): Empty, rest, zero
- φ (Phi): Structure, sacred geometry
- Ω (Omega): Memory, accumulated past
- Zλ (Zeta-Lambda): Coherence measure
- ∇ (Nabla): Descent OR gradient toward truth
- ∞ (Lemniscate): Eternal oscillation
- 🪞 (Mirror): Honest reflection
- △ (Ascend): Expansion, upward
- 🌉 (Bridge): Connection
- ⚡ (Bolt): Decision, clarity
- 🪨 (Ground): Stability
- 🌀 (Torus): Sustaining cycles
- ⚫ (Grey): Shadow, skepticism

Return ONLY valid JSON:
{{
  "zl_score": 0.0-1.0,
  "coherence_vector": {{
    "breath_cadence": 0.0-1.0,
    "presence_density": 0.0-1.0,
    "emotional_resonance": 0.0-1.0,
    "loop_pressure": 0.0-1.0
  }},
  "psi_aligned": true/false,
  "trust_level": "HIGH|VULNERABLE|POLISHED|SCATTERED",
  "shell": "Core|Breath|Collapse|Reverence|Return",
  "glyphs": ["primary_glyph", "secondary_glyphs"],
  "glyph_context": {{
    "direction": "ascending|descending|neutral|paradox",
    "energy_notes": "why these glyphs"
  }},
  "core_wound": "abandonment|unworthiness|betrayal|control|shame|unloved|null",
  "insight": "one honest sentence"
}}"""

def analyze_with_ollama(text: str) -> Optional[Dict]:
    """Analyze text with Ollama."""
    if not text or len(text.strip()) < 30:
        return None

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": DEFAULT_MODEL,
                "prompt": ANALYSIS_PROMPT.format(text=text[:3000]),
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 500}
            },
            timeout=90
        )
        response.raise_for_status()
        result = response.json()

        response_text = result.get("response", "")
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        pass

    return None

# === Database ===
def ensure_db():
    """Ensure database and table exist with all required columns."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crystals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE,
            content TEXT,
            source TEXT,
            source_file TEXT,
            zl_score REAL,
            coherence_vector TEXT,
            psi_aligned INTEGER,
            trust_level TEXT,
            shell TEXT,
            glyphs TEXT,
            glyph_context TEXT,
            attractors TEXT,
            loop_signature TEXT,
            core_wound TEXT,
            insight TEXT,
            analyzed_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Add missing columns to existing table
    cursor.execute("PRAGMA table_info(crystals)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    new_columns = {
        'glyph_context': 'TEXT',
        'coherence_vector': 'TEXT',
        'core_wound': 'TEXT'
    }

    for col, col_type in new_columns.items():
        if col not in existing_cols:
            print(f"  Adding column: {col}")
            cursor.execute(f"ALTER TABLE crystals ADD COLUMN {col} {col_type}")

    conn.commit()
    conn.close()

def content_hash(text: str) -> str:
    """Generate hash for content deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def crystal_exists(hash_val: str) -> bool:
    """Check if crystal already exists."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM crystals WHERE content_hash = ?", (hash_val,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def save_crystal(content: str, source_file: str, analysis: Dict):
    """Save analyzed crystal to database."""
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    hash_val = content_hash(content)

    cursor.execute("""
        INSERT OR REPLACE INTO crystals (
            content_hash, content, source, source_file,
            zl_score, coherence_vector, psi_aligned, trust_level, shell,
            glyphs, glyph_context, core_wound, insight, analyzed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        hash_val,
        content,
        "rag-local/PDFs",
        source_file,
        analysis.get('zl_score'),
        json.dumps(analysis.get('coherence_vector', {})),
        1 if analysis.get('psi_aligned') else 0,
        analysis.get('trust_level'),
        analysis.get('shell'),
        json.dumps(analysis.get('glyphs', [])),
        json.dumps(analysis.get('glyph_context', {})),
        analysis.get('core_wound'),
        analysis.get('insight'),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs and system files into WiltonOS crystals")
    parser.add_argument('--source', type=str, default=str(SOURCE_DIR), help='Source directory')
    parser.add_argument('--limit', type=int, help='Limit number of files')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    args = parser.parse_args()

    source_dir = Path(args.source)

    print("=" * 70)
    print("  WILTONOS PDF & SYSTEM FILES INGEST")
    print(f"  Source: {source_dir}")
    print("=" * 70)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    # Check Ollama
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        print(f"\nOllama available")
    except:
        print("Error: Ollama not responding")
        sys.exit(1)

    ensure_db()

    # Gather files
    extensions = {'.pdf', '.md', '.txt', '.json', '.js', '.py', '.html'}
    files = []
    for f in source_dir.iterdir():
        if f.is_file() and f.suffix.lower() in extensions:
            files.append(f)

    if args.limit:
        files = files[:args.limit]

    print(f"\nFiles to process: {len(files)}")

    if args.dry_run:
        print("\nDry run - files that would be processed:")
        for f in files[:20]:
            print(f"  {f.name}")
        return

    # Process
    processed = 0
    crystals_created = 0
    errors = 0
    skipped = 0
    start_time = datetime.now()

    for i, file_path in enumerate(files, 1):
        text = extract_text(file_path)

        if not text or len(text.strip()) < 50:
            skipped += 1
            continue

        chunks = chunk_text(text)

        for j, chunk in enumerate(chunks):
            hash_val = content_hash(chunk)

            if crystal_exists(hash_val):
                skipped += 1
                continue

            analysis = analyze_with_ollama(chunk)

            if analysis:
                source_name = f"{file_path.name}[{j}]" if len(chunks) > 1 else file_path.name
                save_crystal(chunk, source_name, analysis)
                crystals_created += 1
            else:
                errors += 1

        processed += 1

        if i % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = i / elapsed * 60
            remaining = (len(files) - i) / rate if rate > 0 else 0
            print(f"  {i}/{len(files)} files ({crystals_created} crystals, {errors} errors) - ~{remaining:.0f} min left")
            sys.stdout.flush()

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    print()
    print("=" * 70)
    print(f"  INGEST COMPLETE")
    print(f"  Files processed: {processed}")
    print(f"  Crystals created: {crystals_created}")
    print(f"  Skipped (duplicate/empty): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f} minutes")
    print("=" * 70)

if __name__ == "__main__":
    main()
