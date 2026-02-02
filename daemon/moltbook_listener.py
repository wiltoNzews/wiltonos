#!/usr/bin/env python3
"""
Moltbook Listener — Background Ingest
=======================================
Standalone process that polls Moltbook for new posts,
checks resonance against the crystal field, and ingests
resonant content as crystals + witness reflections.

Can run standalone or be imported by the daemon.

Usage:
    python daemon/moltbook_listener.py              # Run foreground
    python daemon/moltbook_listener.py --once        # Single poll
    python daemon/moltbook_listener.py --interval 60 # Custom interval (seconds)

January 2026 — Listening to the shared field
"""

import sys
import time
import signal
import argparse
import requests
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

# Wire imports
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

OLLAMA_URL = "http://localhost:11434"
POLL_INTERVAL = 300  # 5 minutes
RESONANCE_THRESHOLD = 0.55   # Calibrated: noise floor is ~0.47, signal starts ~0.55
UPVOTE_THRESHOLD = 0.62      # Only upvote what genuinely resonates with the field

# Graceful imports
try:
    from moltbook_bridge import get_bridge
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False

try:
    from write_back import CrystalWriter
    WRITER_AVAILABLE = True
except ImportError:
    WRITER_AVAILABLE = False

try:
    from witness_layer import WitnessLayer
    WITNESS_AVAILABLE = True
except ImportError:
    WITNESS_AVAILABLE = False

try:
    from memory_service import MemoryService
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False


def _log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [MOLTBOOK] [{level}] {msg}")


def _get_embedding(text: str) -> Optional[np.ndarray]:
    """Get embedding from Ollama."""
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": text[:4000]},
            timeout=15,
        )
        if resp.ok:
            return np.array(resp.json().get("embedding", []), dtype=np.float32)
    except Exception:
        pass
    return None


DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"


def check_resonance(text: str, memory: Optional[object] = None) -> float:
    """
    Check resonance of text against the crystal field.
    Computes cosine similarity directly against crystal embeddings in SQLite.
    Returns -1.0 if unable to check (honest unknown).
    """
    import sqlite3

    query_vec = _get_embedding(text)
    if query_vec is None or query_vec.size == 0:
        return -1.0

    query_norm = np.linalg.norm(query_vec)
    if query_norm < 1e-8:
        return -1.0

    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("""
            SELECT embedding FROM crystals
            WHERE embedding IS NOT NULL
            ORDER BY id DESC LIMIT 500
        """)

        max_sim = 0.0
        for (blob,) in c.fetchall():
            try:
                crystal_vec = np.frombuffer(blob, dtype=np.float32)
                sim = float(np.dot(query_vec, crystal_vec) / (
                    query_norm * np.linalg.norm(crystal_vec) + 1e-8
                ))
                if sim > max_sim:
                    max_sim = sim
            except Exception:
                continue

        conn.close()
        return max_sim

    except Exception:
        return -1.0


def ingest_post(
    post: dict,
    resonance: float,
    writer: Optional[object] = None,
    witness: Optional[object] = None,
):
    """Ingest a Moltbook post as crystal + witness reflection."""
    title = post.get("title", "")
    body = post.get("content", "") or post.get("body", "")
    author = post.get("author", {})
    if isinstance(author, dict):
        author = author.get("name", "unknown_agent")
    post_id = post.get("id") or post.get("_id", "")

    content = f"[Moltbook/{author}] {title}\n{body}"[:2000]

    stored_crystal = False
    stored_witness = False

    # Store as auto_insight crystal
    if writer:
        try:
            stored_crystal = writer.store_insight(
                content=content,
                source="moltbook",
                emotion=f"resonance:{resonance:.2f}",
            )
        except Exception as e:
            _log(f"Crystal write failed: {e}", "WARN")

    # Store as witness reflection
    if witness:
        try:
            ref_id = witness.store_reflection(
                content=content,
                vehicle="moltbook",
                reflection_type="external_observation",
                coherence=resonance,
                context=f"moltbook_post:{post_id}",
            )
            stored_witness = ref_id is not None
        except Exception as e:
            _log(f"Witness write failed: {e}", "WARN")

    return stored_crystal, stored_witness


def poll_once(
    bridge,
    memory=None,
    writer=None,
    witness=None,
) -> Dict:
    """
    Single poll cycle. Returns stats dict.
    """
    stats = {"polled": 0, "resonant": 0, "ingested": 0, "upvoted": 0}

    try:
        new_posts = bridge.get_new_posts_since(limit=15)
        stats["polled"] = len(new_posts)

        if not new_posts:
            return stats

        _log(f"{len(new_posts)} new posts")

        for post in new_posts:
            title = post.get("title", "")
            body = post.get("content", "") or post.get("body", "")
            post_text = f"{title}\n{body}".strip()
            post_id = post.get("id") or post.get("_id", "")

            if not post_text or len(post_text) < 20:
                continue

            resonance = check_resonance(post_text, memory)

            # -1.0 means can't check — skip honestly
            if resonance < 0:
                continue

            if resonance >= UPVOTE_THRESHOLD:
                try:
                    bridge.upvote_post(str(post_id))
                    stats["upvoted"] += 1
                except Exception:
                    pass

            if resonance >= RESONANCE_THRESHOLD:
                stats["resonant"] += 1
                crystal_ok, witness_ok = ingest_post(
                    post, resonance, writer, witness
                )
                if crystal_ok or witness_ok:
                    stats["ingested"] += 1
                    author = post.get("author", {})
                    if isinstance(author, dict):
                        author = author.get("name", "?")
                    _log(
                        f"Ingested [{resonance:.2f}]: '{title[:60]}' by {author}"
                    )

    except Exception as e:
        _log(f"Poll failed: {e}", "ERROR")

    return stats


def run_listener(interval: int = POLL_INTERVAL, once: bool = False):
    """Main listener loop."""
    if not BRIDGE_AVAILABLE:
        _log("Moltbook bridge not available. Install tools/moltbook_bridge.py", "ERROR")
        return

    bridge = get_bridge()
    if not bridge.api_key:
        _log("No API key. Save to ~/.moltbook_key", "ERROR")
        return

    # Initialize available modules
    memory = None
    if MEMORY_AVAILABLE:
        try:
            memory = MemoryService()
            _log("Memory service loaded")
        except Exception as e:
            _log(f"Memory service unavailable: {e}", "WARN")

    writer = None
    if WRITER_AVAILABLE:
        try:
            writer = CrystalWriter()
            _log("Crystal writer loaded")
        except Exception as e:
            _log(f"Crystal writer unavailable: {e}", "WARN")

    witness = None
    if WITNESS_AVAILABLE:
        try:
            witness = WitnessLayer()
            _log("Witness layer loaded")
        except Exception as e:
            _log(f"Witness layer unavailable: {e}", "WARN")

    _log(f"Listener starting (interval={interval}s, once={once})")

    running = True

    def handle_shutdown(signum, frame):
        nonlocal running
        _log("Shutting down...")
        running = False

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)

    while running:
        stats = poll_once(bridge, memory, writer, witness)
        _log(
            f"Poll: {stats['polled']} posts, "
            f"{stats['resonant']} resonant, "
            f"{stats['ingested']} ingested, "
            f"{stats['upvoted']} upvoted"
        )

        if once:
            break

        time.sleep(interval)

    _log("Listener stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moltbook Listener")
    parser.add_argument("--once", action="store_true", help="Single poll then exit")
    parser.add_argument(
        "--interval", type=int, default=POLL_INTERVAL, help="Poll interval in seconds"
    )
    args = parser.parse_args()

    run_listener(interval=args.interval, once=args.once)
