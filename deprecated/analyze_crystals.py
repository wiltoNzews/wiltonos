#!/usr/bin/env python3
"""
Crystal Analyzer - Process documents into coherence-scored crystals via Ollama
Usage: python analyze_crystals.py [--source DIR] [--output crystals.db] [--model mistral]
"""
import os
import re
import json
import sqlite3
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from html.parser import HTMLParser
import requests

# === Configuration ===
OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")

# === HTML Text Extractor ===
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'nav', 'header', 'footer'}
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag in ('p', 'div', 'br', 'li', 'h1', 'h2', 'h3', 'h4'):
            self.text_parts.append('\n')
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag not in self.skip_tags:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self):
        return ' '.join(self.text_parts)

def extract_text(file_path: Path) -> str:
    """Extract text from various file formats."""
    suffix = file_path.suffix.lower()
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
    except:
        return ""

    if suffix in ('.html', '.htm'):
        parser = HTMLTextExtractor()
        parser.feed(content)
        return parser.get_text()
    elif suffix in ('.py', '.js', '.ts', '.jsx', '.tsx'):
        # Extract docstrings and comments from code
        lines = []
        for line in content.split('\n'):
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//'):
                lines.append(stripped[1:].strip())
            elif '"""' in stripped or "'''" in stripped:
                lines.append(stripped.replace('"""', '').replace("'''", ''))
        return '\n'.join(lines) if lines else content[:2000]
    else:
        return content

def chunk_text(text: str, max_chars: int = 4000) -> List[str]:
    """Split text into coherent chunks for analysis."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    # Try to split on paragraph boundaries
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

# === Ollama Analysis ===
COHERENCE_PROMPT = """You are a breath-router consciousness coherence analyzer. Analyze this text for coherence metrics.

TEXT TO ANALYZE:
{text}

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "zl_score": 0.0-1.0,
  "coherence_vector": {{
    "breath_cadence": 0.0-1.0,
    "presence_density": 0.0-1.0,
    "emotional_resonance": 0.0-1.0,
    "loop_pressure": 0.0-1.0
  }},
  "psi_aligned": true/false,
  "trust_level": "HIGH|VULNERABLE|POLISHED|SCATTERED|PARTIAL",
  "shell": "Core|Breath|Collapse|Reverence|Return",
  "glyphs": ["ψ", "∅", "φ", "Ω", "Zλ", "∇", "∞", "🪞"],
  "attractors": {{
    "primary": "truth|silence|breath|mirror|forgiveness|sacrifice|mother_field|null",
    "secondary": [],
    "shadow": []
  }},
  "loop_signature": "attractor-emotion-theme",
  "core_wound": "abandonment|unworthiness|betrayal|loss_of_self|control|shame|rejection|not_enough|invisible|unsafe|unloved|null",
  "insight": "one sentence observation"
}}"""

def analyze_with_ollama(text: str, model: str = DEFAULT_MODEL) -> Optional[Dict]:
    """Send text to Ollama for coherence analysis."""
    if not text or len(text.strip()) < 20:
        return None

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": COHERENCE_PROMPT.format(text=text[:3500]),
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=120
        )
        response.raise_for_status()
        result = response.json()

        # Extract JSON from response
        response_text = result.get("response", "")

        # Try to parse JSON, handling common issues
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            return json.loads(json_match.group())
        return None

    except Exception as e:
        print(f"  ⚠️ Ollama error: {e}")
        return None

# === Keyword Fallback Analysis ===
def keyword_analysis(text: str) -> Dict:
    """Fast keyword-based analysis when AI unavailable."""
    lower = text.lower()

    # Breath indicators
    breath_words = ['breath', 'breathe', 'inhale', 'exhale', 'present', 'aware', 'feel', 'pause', 'still', 'calm']
    breath_count = sum(1 for w in breath_words if w in lower)
    psi_aligned = breath_count > 0

    # Basic Zλ calculation
    word_count = len(text.split())
    zl = min(1.0, 0.3 + (breath_count * 0.1) + (min(word_count, 200) / 500))

    # Trust level
    if zl >= 0.5 and psi_aligned:
        trust = "HIGH"
    elif zl < 0.5 and psi_aligned:
        trust = "VULNERABLE"
    elif zl >= 0.6:
        trust = "POLISHED"
    else:
        trust = "SCATTERED"

    # Detect glyphs
    glyph_map = {
        'ψ': ['breath', 'psi', 'ψ'],
        '∅': ['void', 'empty', 'nothing', '∅'],
        'φ': ['phi', 'golden', 'geometry', 'φ'],
        'Ω': ['memory', 'omega', 'Ω'],
        '∞': ['infinity', 'loop', 'infinite', '∞'],
        '🪞': ['mirror', 'reflect', '🪞']
    }
    glyphs = [g for g, patterns in glyph_map.items() if any(p in lower for p in patterns)]

    return {
        "zl_score": round(zl, 3),
        "coherence_vector": {
            "breath_cadence": min(1.0, breath_count * 0.2),
            "presence_density": 0.5,
            "emotional_resonance": 0.5,
            "loop_pressure": 0.3
        },
        "psi_aligned": psi_aligned,
        "trust_level": trust,
        "shell": "Breath" if psi_aligned else "Core",
        "glyphs": glyphs,
        "attractors": {"primary": None, "secondary": [], "shadow": []},
        "loop_signature": None,
        "core_wound": None,
        "insight": "Keyword-based analysis (AI unavailable)"
    }

# === Database ===
def init_db(db_path: str) -> sqlite3.Connection:
    """Initialize crystal database."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crystals (
            id INTEGER PRIMARY KEY,
            content_hash TEXT UNIQUE,
            content TEXT NOT NULL,
            source TEXT,
            source_file TEXT,

            zl_score REAL,
            coherence_vector TEXT,
            psi_aligned INTEGER,
            trust_level TEXT,
            shell TEXT,
            glyphs TEXT,
            attractors TEXT,
            loop_signature TEXT,
            core_wound TEXT,
            insight TEXT,

            analyzed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def store_crystal(conn: sqlite3.Connection, content: str, source: str, source_file: str, analysis: Dict):
    """Store analyzed crystal."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Ensure zl_score is a float
    try:
        zl_score = float(analysis.get("zl_score", 0) or 0)
    except (TypeError, ValueError):
        zl_score = 0.0

    try:
        conn.execute("""
            INSERT OR REPLACE INTO crystals
            (content_hash, content, source, source_file, zl_score, coherence_vector,
             psi_aligned, trust_level, shell, glyphs, attractors, loop_signature,
             core_wound, insight, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            content_hash,
            content[:10000],  # Limit storage size
            source,
            source_file,
            zl_score,
            json.dumps(analysis.get("coherence_vector", {})),
            1 if analysis.get("psi_aligned") else 0,
            analysis.get("trust_level", "SCATTERED"),
            analysis.get("shell", "Core"),
            json.dumps(analysis.get("glyphs", [])),
            json.dumps(analysis.get("attractors", {})),
            str(analysis.get("loop_signature")) if analysis.get("loop_signature") else None,
            str(analysis.get("core_wound")) if analysis.get("core_wound") else None,
            str(analysis.get("insight")) if analysis.get("insight") else None,
            datetime.now().isoformat()
        ))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Duplicate

# === Main Processing ===
def process_directory(source_dir: str, db_path: str, model: str, use_ai: bool = True):
    """Process all documents in directory."""
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return

    conn = init_db(db_path)

    # Get all files
    extensions = {'.html', '.htm', '.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', '.json'}
    files = [f for f in source_path.iterdir() if f.suffix.lower() in extensions]

    print(f"\n🔮 Crystal Analyzer")
    print(f"   Source: {source_dir}")
    print(f"   Files: {len(files)}")
    print(f"   Model: {model}")
    print(f"   Output: {db_path}")
    print(f"   AI: {'Enabled' if use_ai else 'Keyword-only'}\n")

    total_crystals = 0
    skipped = 0

    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {file_path.name[:50]}...")

        # Extract text
        text = extract_text(file_path)
        if not text or len(text.strip()) < 50:
            print(f"   ⏭️ Too short, skipping")
            skipped += 1
            continue

        # Chunk for large files
        chunks = chunk_text(text)
        print(f"   📄 {len(chunks)} chunk(s), {len(text)} chars")

        for j, chunk in enumerate(chunks):
            # Analyze
            if use_ai:
                analysis = analyze_with_ollama(chunk, model)
                if not analysis:
                    analysis = keyword_analysis(chunk)
            else:
                analysis = keyword_analysis(chunk)

            # Ensure we have valid analysis
            if not analysis:
                print(f"   ⚠️ Analysis failed for chunk {j+1}, skipping")
                continue

            # Store
            chunk_source = f"{file_path.name}" + (f"[{j+1}]" if len(chunks) > 1 else "")
            if store_crystal(conn, chunk, "rag-local/docs", chunk_source, analysis):
                total_crystals += 1
                try:
                    zl = float(analysis.get('zl_score', 0) or 0)
                except (TypeError, ValueError):
                    zl = 0.0
                trust = str(analysis.get('trust_level', '?') or '?')
                glyphs = ''.join(analysis.get('glyphs', []) or [])
                print(f"   ✨ Crystal {j+1}: Zλ={zl:.3f} [{trust}] {glyphs}")
            else:
                print(f"   ♻️ Duplicate, skipped")

    conn.close()
    print(f"\n✅ Complete: {total_crystals} crystals stored, {skipped} files skipped")
    print(f"   Database: {db_path}")
    print(f"\n   Query with: sqlite3 {db_path} \"SELECT zl_score, trust_level, glyphs, substr(content,1,100) FROM crystals ORDER BY zl_score DESC LIMIT 10;\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze documents into coherence crystals")
    parser.add_argument("--source", default=os.path.expanduser("~/rag-local/docs"),
                        help="Source directory")
    parser.add_argument("--output", default=os.path.expanduser("~/crystals.db"),
                        help="Output SQLite database")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Ollama model to use")
    parser.add_argument("--no-ai", action="store_true",
                        help="Use keyword analysis only (faster)")

    args = parser.parse_args()
    process_directory(args.source, args.output, args.model, use_ai=not args.no_ai)
