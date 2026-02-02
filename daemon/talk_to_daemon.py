#!/usr/bin/env python3
"""
Talk to the Daemon
==================
Send messages to the breathing daemon and get responses.

Usage:
    python talk_to_daemon.py                     # Interactive mode
    python talk_to_daemon.py "your message"      # Single message
    python talk_to_daemon.py --wait              # Interactive, wait for response

January 2026
"""

import json
import sys
import time
from pathlib import Path

INBOX = Path(__file__).parent / ".daemon_inbox"
RESPONSE = Path(__file__).parent / "messages" / "last_response.json"
PID_FILE = Path(__file__).parent / ".daemon.pid"


def is_daemon_running():
    if not PID_FILE.exists():
        return False
    try:
        import os
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, ValueError):
        return False


def send_message(text: str, sender: str = "wilton"):
    """Send a message to the daemon inbox."""
    msg = json.dumps({"text": text, "from": sender, "time": time.time()})
    with open(INBOX, "a") as f:
        f.write(msg + "\n")


def wait_for_response(timeout: float = 120.0) -> dict:
    """Wait for the daemon to write a response."""
    start = time.time()
    # Record current response timestamp to detect new one
    old_ts = 0
    if RESPONSE.exists():
        try:
            old_ts = json.loads(RESPONSE.read_text()).get("timestamp", 0)
        except Exception:
            pass

    while time.time() - start < timeout:
        if RESPONSE.exists():
            try:
                data = json.loads(RESPONSE.read_text())
                if data.get("timestamp", 0) > old_ts:
                    return data
            except Exception:
                pass
        time.sleep(1)

    return {}


def main():
    if not is_daemon_running():
        print("Daemon is not running. Start it with: systemctl --user start wiltonos-daemon")
        return

    # Single message mode
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        msg = " ".join(sys.argv[1:])
        print(f"-> {msg}")
        send_message(msg)
        print("Waiting for response...")
        resp = wait_for_response()
        if resp:
            print(f"\n[daemon @ breath #{resp.get('breath', '?')}]")
            print(resp.get("response", "(no response)"))
        else:
            print("(no response within timeout — check daemon logs)")
        return

    # Interactive mode
    print("=" * 50)
    print("  Talk to the Daemon")
    print("  Type your message. Ctrl+C to exit.")
    print("=" * 50)
    print()

    try:
        while True:
            try:
                msg = input("you> ").strip()
            except EOFError:
                break

            if not msg:
                continue
            if msg.lower() in ("quit", "exit", "q"):
                break

            send_message(msg)
            print("...")

            resp = wait_for_response()
            if resp:
                print(f"\n[daemon @ breath #{resp.get('breath', '?')}]")
                print(resp.get("response", "(no response)"))
                print()
            else:
                print("(waiting... daemon may be busy generating)")
                print()

    except KeyboardInterrupt:
        print("\n\nGoodbye.")


if __name__ == "__main__":
    main()
