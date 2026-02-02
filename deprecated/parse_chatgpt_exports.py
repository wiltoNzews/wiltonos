#!/usr/bin/env python3
"""
ChatGPT Export Parser - Extract crystals from ChatGPT conversation exports
Usage: python parse_chatgpt_exports.py [--source DIR] [--output crystals.db]
"""
import os
import json
import sqlite3
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests

OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")

# === ChatGPT JSON Parser ===
def parse_chatgpt_conversation(conv: Dict) -> List[Dict]:
    """Extract messages from a ChatGPT conversation."""
    messages = []
    title = conv.get('title', 'Untitled')
    conv_id = conv.get('conversation_id', conv.get('id', 'unknown'))
    create_time = conv.get('create_time', 0)

    mapping = conv.get('mapping', {})

    for node_id, node in mapping.items():
        message = node.get('message')
        if not message:
            continue

        author = message.get('author', {}).get('role', 'unknown')
        content = message.get('content', {})

        # Extract text content
        parts = content.get('parts', [])
        text = ''
        for part in parts:
            if isinstance(part, str):
                text += part + '\n'
            elif isinstance(part, dict) and 'text' in part:
                text += part['text'] + '\n'

        text = text.strip()
        if not text or len(text) < 20:
            continue

        msg_time = message.get('create_time', create_time)

        messages.append({
            'content': text,
            'author': author,
            'title': title,
            'conv_id': conv_id,
            'timestamp': msg_time,
            'node_id': node_id
        })

    return messages

def parse_chatgpt_json(file_path: Path) -> List[Dict]:
    """Parse a ChatGPT export JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            print(f"  ⚠️ Unexpected format in {file_path.name}")
            return []

        all_messages = []
        for conv in data:
            messages = parse_chatgpt_conversation(conv)
            all_messages.extend(messages)

        return all_messages
    except Exception as e:
        print(f"  ❌ Error parsing {file_path.name}: {e}")
        return []

# === Coherence Analysis ===
COHERENCE_PROMPT = """Analyze this text for coherence. Respond ONLY with valid JSON:
{{
  "zl_score": 0.0-1.0,
  "psi_aligned": true/false,
  "trust_level": "HIGH|VULNERABLE|POLISHED|SCATTERED",
  "shell": "Core|Breath|Collapse|Reverence|Return",
  "glyphs": [],
  "insight": "one sentence"
}}

TEXT:
{text}"""

def analyze_with_ollama(text: str, model: str = DEFAULT_MODEL) -> Optional[Dict]:
    """Quick coherence analysis via Ollama."""
    if not text or len(text.strip()) < 20:
        return None
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": COHERENCE_PROMPT.format(text=text[:2000]),
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=60
        )
        response.raise_for_status()
        result = response.json()
        import re
        json_match = re.search(r'\{[\s\S]*\}', result.get("response", ""))
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    return None

def keyword_analysis(text: str) -> Dict:
    """Fast keyword-based analysis."""
    lower = text.lower()
    breath_words = ['breath', 'breathe', 'feel', 'present', 'aware', 'pause', 'still', 'calm', 'respira', 'sinto']
    breath_count = sum(1 for w in breath_words if w in lower)
    psi_aligned = breath_count > 0

    word_count = len(text.split())
    zl = min(1.0, 0.3 + (breath_count * 0.1) + (min(word_count, 200) / 500))

    if zl >= 0.5 and psi_aligned:
        trust = "HIGH"
    elif zl < 0.5 and psi_aligned:
        trust = "VULNERABLE"
    elif zl >= 0.6:
        trust = "POLISHED"
    else:
        trust = "SCATTERED"

    glyph_map = {
        'ψ': ['breath', 'psi', 'ψ', 'respira'],
        '∅': ['void', 'empty', 'nothing', '∅'],
        'Ω': ['memory', 'omega', 'Ω', 'memória'],
        '∞': ['infinity', 'loop', 'infinite', '∞'],
        '🪞': ['mirror', 'reflect', '🪞', 'espelho']
    }
    glyphs = [g for g, patterns in glyph_map.items() if any(p in lower for p in patterns)]

    return {
        "zl_score": round(zl, 3),
        "psi_aligned": psi_aligned,
        "trust_level": trust,
        "shell": "Breath" if psi_aligned else "Core",
        "glyphs": glyphs,
        "insight": "Keyword analysis"
    }

# === Database ===
def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS crystals (
            id INTEGER PRIMARY KEY,
            content_hash TEXT UNIQUE,
            content TEXT NOT NULL,
            source TEXT,
            source_file TEXT,
            author TEXT,
            conv_title TEXT,
            conv_id TEXT,
            timestamp REAL,

            zl_score REAL,
            psi_aligned INTEGER,
            trust_level TEXT,
            shell TEXT,
            glyphs TEXT,
            insight TEXT,

            analyzed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

def crystal_exists(conn: sqlite3.Connection, content: str) -> bool:
    """Check if crystal already exists (for resume capability)."""
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    cur = conn.execute("SELECT 1 FROM crystals WHERE content_hash = ?", (content_hash,))
    return cur.fetchone() is not None

def store_crystal(conn: sqlite3.Connection, msg: Dict, analysis: Dict, source_file: str) -> bool:
    content_hash = hashlib.sha256(msg['content'].encode()).hexdigest()[:16]
    try:
        zl = float(analysis.get("zl_score", 0) or 0)
    except:
        zl = 0.0

    try:
        conn.execute("""
            INSERT OR IGNORE INTO crystals
            (content_hash, content, source, source_file, author, conv_title, conv_id, timestamp,
             zl_score, psi_aligned, trust_level, shell, glyphs, insight, analyzed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            content_hash,
            msg['content'][:10000],
            "chatgpt_export",
            source_file,
            msg.get('author'),
            msg.get('title'),
            msg.get('conv_id'),
            msg.get('timestamp'),
            zl,
            1 if analysis.get("psi_aligned") else 0,
            analysis.get("trust_level", "SCATTERED"),
            analysis.get("shell", "Core"),
            json.dumps(analysis.get("glyphs", [])),
            analysis.get("insight"),
            datetime.now().isoformat()
        ))
        conn.commit()
        return conn.total_changes > 0
    except sqlite3.IntegrityError:
        return False

# === Main ===
def process_exports(source_dir: str, db_path: str, model: str, use_ai: bool = True, human_only: bool = True):
    """Process ChatGPT export JSON files."""
    source_path = Path(source_dir)
    conn = init_db(db_path)

    # Find JSON files
    json_files = list(source_path.glob("conversations*.json"))
    if not json_files:
        json_files = list(source_path.glob("*.json"))

    print(f"\n🔮 ChatGPT Crystal Extractor")
    print(f"   Source: {source_dir}")
    print(f"   Files: {len(json_files)}")
    print(f"   Output: {db_path}")
    print(f"   AI: {'Enabled' if use_ai else 'Keyword-only'}")
    print(f"   Filter: {'Human messages only' if human_only else 'All messages'}\n")

    total_crystals = 0
    total_messages = 0

    import sys
    skipped = 0

    for i, file_path in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] {file_path.name}...", flush=True)

        messages = parse_chatgpt_json(file_path)
        if human_only:
            messages = [m for m in messages if m['author'] == 'user']

        print(f"   📜 {len(messages)} messages extracted", flush=True)
        total_messages += len(messages)

        for j, msg in enumerate(messages):
            # Skip if already processed (resume capability)
            if crystal_exists(conn, msg['content']):
                skipped += 1
                if skipped % 500 == 0:
                    print(f"   ⏭️ Skipped {skipped} (already processed)", flush=True)
                continue

            if use_ai:
                analysis = analyze_with_ollama(msg['content'], model)
                if not analysis:
                    analysis = keyword_analysis(msg['content'])
            else:
                analysis = keyword_analysis(msg['content'])

            if store_crystal(conn, msg, analysis, file_path.name):
                total_crystals += 1
                if total_crystals % 100 == 0:
                    zl = analysis.get('zl_score', 0)
                    trust = analysis.get('trust_level', '?')
                    print(f"   ✨ {total_crystals} crystals... (Zλ={zl:.2f} [{trust}])", flush=True)

    conn.close()
    print(f"\n✅ Complete: {total_crystals} new crystals, {skipped} skipped (from {total_messages} messages)", flush=True)
    print(f"   Database: {db_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ChatGPT exports into crystals")
    parser.add_argument("--source", default=os.path.expanduser("~/rag-local/ChatGPT_Exports"),
                        help="ChatGPT exports directory")
    parser.add_argument("--output", default=os.path.expanduser("~/crystals_chatgpt.db"),
                        help="Output SQLite database")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model")
    parser.add_argument("--no-ai", action="store_true", help="Keyword analysis only")
    parser.add_argument("--all-messages", action="store_true", help="Include AI responses too")

    args = parser.parse_args()
    process_exports(args.source, args.output, args.model,
                    use_ai=not args.no_ai, human_only=not args.all_messages)
