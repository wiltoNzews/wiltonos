#!/usr/bin/env python3
"""
Claude Session Bridge — Feed Claude Code sessions into the crystal system.
==========================================================================
Reads ~/.claude/projects/-home-zews/*.jsonl session files,
extracts user messages and session metadata,
stores them as crystals in auto_insights with embeddings.

The system needs to remember what happened in Claude Code sessions.
Without this bridge, a month of daily conversations vanishes.

Usage:
    python core/claude_session_bridge.py --stats          # Show what exists
    python core/claude_session_bridge.py --dry-run        # Preview extraction
    python core/claude_session_bridge.py --ingest         # Ingest all sessions
    python core/claude_session_bridge.py --ingest-session UUID
    python core/claude_session_bridge.py --since 2026-01-25
"""

import json
import hashlib
import sqlite3
import requests
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Paths
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects" / "-home-zews"
DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
OLLAMA_URL = "http://localhost:11434"

# Extraction config
MIN_USER_MSG_LEN = 40          # Skip very short messages (commands, "yes", etc.)
MAX_CRYSTAL_LEN = 8000         # Truncate extremely long messages
CONTINUATION_MARKER = "This session is being continued from a previous conversation"
SOURCE_TAG = "claude_code"


def _hash(text: str) -> str:
    """MD5 hash for dedup, matching CrystalWriter convention."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def _get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding from Ollama nomic-embed-text."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text[:4000]},
            timeout=30,
        )
        if resp.ok:
            return np.array(resp.json().get("embedding", []), dtype=np.float32)
    except Exception:
        pass
    return None


def load_sessions_index() -> List[Dict]:
    """Load Claude Code sessions index."""
    index_path = CLAUDE_PROJECTS_DIR / "sessions-index.json"
    if not index_path.exists():
        return []

    with open(index_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
    elif isinstance(data, list):
        entries = data
    else:
        return []

    # Sort by modified date descending
    entries.sort(key=lambda x: x.get("modified", ""), reverse=True)
    return entries


def extract_session_crystals(session_path: Path, session_meta: Dict) -> List[Dict]:
    """
    Extract crystal-worthy content from a session JSONL file.

    Returns list of dicts ready for DB insertion:
        [{content, timestamp, session_id, crystal_type, query_hash}]
    """
    if not session_path.exists():
        return []

    session_id = session_meta.get("sessionId", session_path.stem)
    summary = session_meta.get("summary", "")
    crystals = []
    seen_hashes = set()

    # Pass 1: Extract user messages
    user_messages = []
    with open(session_path) as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            msg_type = data.get("type", "")
            msg = data.get("message", {})
            role = msg.get("role", "")
            ts = data.get("timestamp", "")

            if msg_type == "user" and role == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and len(content.strip()) >= MIN_USER_MSG_LEN:
                    user_messages.append({
                        "content": content.strip(),
                        "timestamp": ts,
                    })

    # Pass 2: Process user messages into crystals
    for um in user_messages:
        content = um["content"]
        ts = um["timestamp"]

        # Handle session continuations — extract the actual user content
        # These contain a summary block + the actual new message
        is_continuation = content.startswith(CONTINUATION_MARKER)

        if is_continuation:
            # Store the continuation summary as its own crystal (session context)
            # but only the summary part, not the full preamble
            summary_end = content.find("\n\n---\n\n")
            if summary_end == -1:
                summary_end = content.find("\n---\n")
            if summary_end == -1:
                # No clear boundary — store first 2000 chars as context
                summary_crystal = content[:2000]
            else:
                summary_crystal = content[:summary_end]

            # The actual new content after the summary
            if summary_end > 0:
                actual_content = content[summary_end:].strip().lstrip("-").strip()
                if len(actual_content) >= MIN_USER_MSG_LEN:
                    h = _hash(actual_content[:MAX_CRYSTAL_LEN])
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        crystals.append({
                            "content": actual_content[:MAX_CRYSTAL_LEN],
                            "timestamp": ts,
                            "session_id": session_id,
                            "crystal_type": "user_message",
                            "query_hash": h,
                        })

            # Store the summary itself as a session-context crystal
            h = _hash(summary_crystal[:MAX_CRYSTAL_LEN])
            if h not in seen_hashes:
                seen_hashes.add(h)
                crystals.append({
                    "content": f"[Session Context] {summary_crystal[:MAX_CRYSTAL_LEN]}",
                    "timestamp": ts,
                    "session_id": session_id,
                    "crystal_type": "session_context",
                    "query_hash": h,
                })
        else:
            # Normal user message — store directly
            h = _hash(content[:MAX_CRYSTAL_LEN])
            if h not in seen_hashes:
                seen_hashes.add(h)
                crystals.append({
                    "content": content[:MAX_CRYSTAL_LEN],
                    "timestamp": ts,
                    "session_id": session_id,
                    "crystal_type": "user_message",
                    "query_hash": h,
                })

    # Add a session summary crystal if we have a summary
    if summary:
        created = session_meta.get("created", "")
        modified = session_meta.get("modified", "")
        msg_count = session_meta.get("messageCount", 0)
        first_prompt = session_meta.get("firstPrompt", "")[:200]

        session_crystal = (
            f"[Claude Code Session: {summary}]\n"
            f"Period: {created[:10] if created else '?'} → {modified[:10] if modified else '?'}\n"
            f"Messages: {msg_count}\n"
            f"First prompt: {first_prompt}"
        )
        h = _hash(session_crystal)
        if h not in seen_hashes:
            seen_hashes.add(h)
            crystals.append({
                "content": session_crystal,
                "timestamp": created or modified,
                "session_id": session_id,
                "crystal_type": "session_summary",
                "query_hash": h,
            })

    return crystals


def get_existing_hashes(conn: sqlite3.Connection) -> set:
    """Get all query_hash values already in auto_insights from claude_code source."""
    try:
        rows = conn.execute(
            "SELECT query_hash FROM auto_insights WHERE source = ?",
            (SOURCE_TAG,),
        ).fetchall()
        return {r[0] for r in rows if r[0]}
    except Exception:
        return set()


def ingest_crystals(crystals: List[Dict], conn: sqlite3.Connection,
                    embed: bool = True) -> Tuple[int, int]:
    """
    Insert crystals into auto_insights. Returns (inserted, skipped).
    """
    existing = get_existing_hashes(conn)
    inserted = 0
    skipped = 0

    for c in crystals:
        if c["query_hash"] in existing:
            skipped += 1
            continue

        # Convert timestamp to SQLite-friendly format
        ts = c["timestamp"]
        if ts and ts.endswith("Z"):
            ts = ts[:-1]  # Strip trailing Z

        try:
            conn.execute(
                """INSERT INTO auto_insights
                   (content, source, emotion, topology, glyphs,
                    conversation_id, query_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    c["content"],
                    SOURCE_TAG,
                    None,  # emotion — could be detected later by field_vocab
                    c["crystal_type"],
                    None,  # glyphs
                    c["session_id"],
                    c["query_hash"],
                    ts,
                ),
            )
            insight_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # Embed if requested
            if embed:
                vec = _get_embedding(c["content"])
                if vec is not None:
                    conn.execute(
                        """INSERT OR REPLACE INTO auto_insight_embeddings
                           (insight_id, embedding, model)
                           VALUES (?, ?, 'nomic-embed-text')""",
                        (insight_id, vec.tobytes()),
                    )

            inserted += 1
            existing.add(c["query_hash"])

            # Commit every 10 to avoid losing work
            if inserted % 10 == 0:
                conn.commit()

        except sqlite3.IntegrityError:
            skipped += 1
        except Exception as e:
            print(f"  Error inserting crystal: {e}")
            skipped += 1

    conn.commit()
    return inserted, skipped


def run_stats():
    """Show current bridge state."""
    sessions = load_sessions_index()
    print(f"Claude Code sessions found: {len(sessions)}")
    for s in sessions:
        sid = s.get("sessionId", "?")[:12]
        created = s.get("created", "?")[:10]
        modified = s.get("modified", "?")[:10]
        msgs = s.get("messageCount", 0)
        summary = s.get("summary", "no summary")[:60]
        print(f"  {sid}... | {created}→{modified} | {msgs:3d} msgs | {summary}")

    conn = sqlite3.connect(str(DB_PATH))
    existing = get_existing_hashes(conn)
    total = conn.execute(
        "SELECT COUNT(*) FROM auto_insights WHERE source = ?",
        (SOURCE_TAG,),
    ).fetchone()[0]
    conn.close()

    print(f"\nAlready ingested: {total} crystals from claude_code")


def run_dry_run(since: str = None):
    """Preview what would be extracted."""
    sessions = load_sessions_index()
    total_crystals = 0

    for s in sessions:
        if since:
            modified = s.get("modified", "")[:10]
            if modified < since:
                continue

        session_id = s.get("sessionId", "?")
        jsonl_path = CLAUDE_PROJECTS_DIR / f"{session_id}.jsonl"
        crystals = extract_session_crystals(jsonl_path, s)
        total_crystals += len(crystals)

        summary = s.get("summary", "no summary")[:50]
        print(f"\n{'='*60}")
        print(f"Session: {summary}")
        print(f"  ID: {session_id[:12]}...")
        print(f"  Crystals to extract: {len(crystals)}")
        for c in crystals[:5]:
            ctype = c["crystal_type"]
            preview = c["content"][:120].replace("\n", " ")
            print(f"    [{ctype}] {preview}...")
        if len(crystals) > 5:
            print(f"    ... and {len(crystals) - 5} more")

    print(f"\n{'='*60}")
    print(f"Total crystals to extract: {total_crystals}")


def run_ingest(since: str = None, session_id: str = None, embed: bool = True):
    """Ingest sessions into crystal DB."""
    sessions = load_sessions_index()

    if session_id:
        sessions = [s for s in sessions if s.get("sessionId", "").startswith(session_id)]
        if not sessions:
            print(f"No session found matching: {session_id}")
            return

    conn = sqlite3.connect(str(DB_PATH))

    # Ensure auto_insight_embeddings table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auto_insight_embeddings (
            insight_id INTEGER PRIMARY KEY,
            embedding BLOB,
            model TEXT DEFAULT 'nomic-embed-text'
        )
    """)
    conn.commit()

    total_inserted = 0
    total_skipped = 0

    for s in sessions:
        if since:
            modified = s.get("modified", "")[:10]
            if modified < since:
                continue

        sid = s.get("sessionId", "?")
        jsonl_path = CLAUDE_PROJECTS_DIR / f"{sid}.jsonl"
        summary = s.get("summary", "no summary")[:50]

        print(f"\nIngesting: {summary}")
        print(f"  Session: {sid[:12]}...")

        crystals = extract_session_crystals(jsonl_path, s)
        if not crystals:
            print(f"  No crystals to extract.")
            continue

        inserted, skipped = ingest_crystals(crystals, conn, embed=embed)
        total_inserted += inserted
        total_skipped += skipped
        print(f"  Inserted: {inserted}, Skipped (dupes): {skipped}")

    conn.close()
    print(f"\n{'='*60}")
    print(f"Total inserted: {total_inserted}")
    print(f"Total skipped: {total_skipped}")
    print(f"System memory is now current.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude Session → Crystal Bridge")
    parser.add_argument("--stats", action="store_true", help="Show current state")
    parser.add_argument("--dry-run", action="store_true", help="Preview extraction")
    parser.add_argument("--ingest", action="store_true", help="Ingest all sessions")
    parser.add_argument("--ingest-session", type=str, help="Ingest specific session UUID")
    parser.add_argument("--since", type=str, help="Only process sessions modified after date (YYYY-MM-DD)")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding generation")
    args = parser.parse_args()

    if args.stats:
        run_stats()
    elif args.dry_run:
        run_dry_run(since=args.since)
    elif args.ingest or args.ingest_session:
        run_ingest(
            since=args.since,
            session_id=args.ingest_session,
            embed=not args.no_embed,
        )
    else:
        parser.print_help()
