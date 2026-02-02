#!/usr/bin/env python3
"""
ChatGPT Export Parser - PARALLEL VERSION
Runs multiple Ollama calls concurrently for faster processing.
Usage: python parse_chatgpt_parallel.py [--workers 4]
"""
import os
import json
import sqlite3
import hashlib
import argparse
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import queue

OLLAMA_URL = os.environ.get("AI_OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("AI_OLLAMA_MODEL", "mistral")

# Thread-safe database lock
db_lock = threading.Lock()
progress_lock = threading.Lock()
stats = {"processed": 0, "skipped": 0, "failed": 0}

# === ChatGPT JSON Parser ===
def parse_chatgpt_conversation(conv: Dict) -> List[Dict]:
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
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        all_messages = []
        for conv in data:
            messages = parse_chatgpt_conversation(conv)
            all_messages.extend(messages)
        return all_messages
    except Exception as e:
        print(f"  ❌ Error parsing {file_path.name}: {e}", flush=True)
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
            timeout=90
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
    conn = sqlite3.connect(db_path, check_same_thread=False)
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
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    with db_lock:
        cur = conn.execute("SELECT 1 FROM crystals WHERE content_hash = ?", (content_hash,))
        return cur.fetchone() is not None

def store_crystal(conn: sqlite3.Connection, msg: Dict, analysis: Dict, source_file: str) -> bool:
    content_hash = hashlib.sha256(msg['content'].encode()).hexdigest()[:16]
    try:
        zl = float(analysis.get("zl_score", 0) or 0)
    except:
        zl = 0.0
    try:
        with db_lock:
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
                str(analysis.get("insight")) if analysis.get("insight") else None,
                datetime.now().isoformat()
            ))
            conn.commit()
            return conn.total_changes > 0
    except:
        return False

# === Worker Function ===
def process_message(args):
    msg, model, conn, source_file = args

    # Skip if exists
    if crystal_exists(conn, msg['content']):
        with progress_lock:
            stats["skipped"] += 1
        return None

    # Analyze
    analysis = analyze_with_ollama(msg['content'], model)
    if not analysis:
        analysis = keyword_analysis(msg['content'])

    # Store
    if store_crystal(conn, msg, analysis, source_file):
        with progress_lock:
            stats["processed"] += 1
        return analysis
    else:
        with progress_lock:
            stats["failed"] += 1
        return None

# === Main ===
def process_exports(source_dir: str, db_path: str, model: str, workers: int = 4, human_only: bool = True):
    source_path = Path(source_dir)
    conn = init_db(db_path)

    json_files = sorted(source_path.glob("conversations*.json"))
    if not json_files:
        json_files = list(source_path.glob("*.json"))

    print(f"\n🔮 ChatGPT Crystal Extractor (PARALLEL)", flush=True)
    print(f"   Source: {source_dir}", flush=True)
    print(f"   Files: {len(json_files)}", flush=True)
    print(f"   Workers: {workers}", flush=True)
    print(f"   Output: {db_path}", flush=True)
    print(f"   Filter: {'Human only' if human_only else 'All'}\n", flush=True)

    total_messages = 0

    for i, file_path in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] {file_path.name}...", flush=True)

        messages = parse_chatgpt_json(file_path)
        if human_only:
            messages = [m for m in messages if m['author'] == 'user']

        print(f"   📜 {len(messages)} messages to process", flush=True)
        total_messages += len(messages)

        # Prepare work items
        work_items = [(msg, model, conn, file_path.name) for msg in messages]

        # Process in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_message, item): item for item in work_items}

            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                if done_count % 100 == 0:
                    with progress_lock:
                        p, s = stats["processed"], stats["skipped"]
                    print(f"   ✨ {p} stored, {s} skipped... ({done_count}/{len(messages)})", flush=True)

        with progress_lock:
            print(f"   ✅ File done: {stats['processed']} total stored, {stats['skipped']} skipped", flush=True)

    conn.close()
    print(f"\n✅ COMPLETE", flush=True)
    print(f"   Processed: {stats['processed']}", flush=True)
    print(f"   Skipped: {stats['skipped']}", flush=True)
    print(f"   Total messages: {total_messages}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse ChatGPT exports (parallel)")
    parser.add_argument("--source", default=os.path.expanduser("~/rag-local/ChatGPT_Exports"))
    parser.add_argument("--output", default=os.path.expanduser("~/crystals_chatgpt.db"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    parser.add_argument("--all-messages", action="store_true")

    args = parser.parse_args()
    process_exports(args.source, args.output, args.model, args.workers, not args.all_messages)
