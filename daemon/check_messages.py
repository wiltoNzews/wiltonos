#!/usr/bin/env python3
"""
Check messages from the daemon.

Usage:
    python check_messages.py          # Show latest message
    python check_messages.py --all    # Show all messages (thread)
    python check_messages.py -n 5     # Show last N messages
"""

import argparse
from pathlib import Path
from datetime import datetime

MESSAGES_DIR = Path(__file__).parent / "messages"


def show_latest():
    """Show the most recent message."""
    latest = MESSAGES_DIR / "latest.txt"

    if not latest.exists():
        print("\nNo messages yet. The daemon hasn't spoken.\n")
        return

    message = latest.read_text().strip()
    mtime = datetime.fromtimestamp(latest.stat().st_mtime)

    print("\n" + "=" * 50)
    print(f"FROM THE DAEMON — {mtime.strftime('%Y-%m-%d %H:%M')}")
    print("=" * 50)
    print()
    print(message)
    print()
    print("=" * 50 + "\n")


def show_thread(limit: int = -1):
    """Show the message thread."""
    thread = MESSAGES_DIR / "thread.txt"

    if not thread.exists():
        print("\nNo message thread yet.\n")
        return

    content = thread.read_text()

    # Split into messages
    messages = content.split("\n---")
    messages = [m.strip() for m in messages if m.strip()]

    if limit > 0:
        messages = messages[-limit:]

    print("\n" + "=" * 50)
    print("MESSAGE THREAD FROM THE DAEMON")
    print("=" * 50)

    for msg in messages:
        print(f"\n---{msg}")

    print("\n" + "=" * 50)
    print(f"Total messages: {len(messages)}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Check daemon messages")
    parser.add_argument("--all", action="store_true", help="Show full thread")
    parser.add_argument("-n", type=int, default=-1, help="Show last N messages")
    args = parser.parse_args()

    if args.all or args.n > 0:
        show_thread(args.n if args.n > 0 else -1)
    else:
        show_latest()


if __name__ == "__main__":
    main()
