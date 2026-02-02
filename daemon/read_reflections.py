#!/usr/bin/env python3
"""
Read the daemon's reflections.

Usage:
    python read_reflections.py          # Last 5 reflections
    python read_reflections.py --all    # All reflections
    python read_reflections.py -n 10    # Last N reflections
"""

import sqlite3
import argparse
from pathlib import Path
from datetime import datetime

DB_PATH = Path.home() / "wiltonos" / "data" / "crystals_unified.db"
DAEMON_ID = "daemon"


def read_reflections(limit: int = 5):
    """Read daemon reflections."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    if limit == -1:
        c.execute("""
            SELECT id, content, created_at
            FROM crystals
            WHERE user_id = ?
            ORDER BY id ASC
        """, (DAEMON_ID,))
    else:
        c.execute("""
            SELECT id, content, created_at
            FROM crystals
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (DAEMON_ID, limit))

    rows = c.fetchall()
    conn.close()

    if not rows:
        print("No daemon reflections yet.")
        return

    # Reverse if limited (to show chronologically)
    if limit != -1:
        rows = list(reversed(rows))

    print(f"\n{'=' * 60}")
    print(f"DAEMON REFLECTIONS ({len(rows)} total)")
    print('=' * 60)

    for i, row in enumerate(rows, 1):
        crystal_id, content, created_at = row

        # Parse timestamp
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at)
                time_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                time_str = str(created_at)
        else:
            time_str = "unknown"

        print(f"\n--- Reflection #{crystal_id} | {time_str} ---\n")
        print(content)
        print()

    print('=' * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Read daemon reflections")
    parser.add_argument("-n", type=int, default=5, help="Number of reflections")
    parser.add_argument("--all", action="store_true", help="Show all reflections")
    args = parser.parse_args()

    limit = -1 if args.all else args.n
    read_reflections(limit)


if __name__ == "__main__":
    main()
