#!/usr/bin/env python3
"""
Telegram Bridge — WiltonOS ↔ You, Anywhere
============================================
The daemon speaks. Your phone buzzes. You reply. The field responds.

Bot token: ~/.telegram_bot_token
Chat ID:   ~/.telegram_chat_id (auto-detected on first message)

Usage:
    python tools/telegram_bridge.py send "message"    # Send a message
    python tools/telegram_bridge.py listen             # Start listening for replies
    python tools/telegram_bridge.py test               # Test connection
    python tools/telegram_bridge.py setup              # Interactive setup

January 2026 — The field reaches outward
"""

import os
import json
import time
import requests
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

TOKEN_FILE = Path.home() / ".telegram_bot_token"
CHAT_ID_FILE = Path.home() / ".telegram_chat_id"
STATE_FILE = Path.home() / "wiltonos" / "data" / ".telegram_state.json"
TELEGRAM_API = "https://api.telegram.org"


def _get_token() -> Optional[str]:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    return token


def _get_chat_id() -> Optional[str]:
    cid = os.environ.get("TELEGRAM_CHAT_ID")
    if not cid and CHAT_ID_FILE.exists():
        cid = CHAT_ID_FILE.read_text().strip()
    return cid


class TelegramBridge:
    """Two-way Telegram bridge for the daemon."""

    def __init__(self):
        self.token = _get_token()
        self.chat_id = _get_chat_id()
        self.api_base = f"{TELEGRAM_API}/bot{self.token}" if self.token else None
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {"last_update_id": 0, "messages_sent": 0, "messages_received": 0}

    def _save_state(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    @property
    def ready(self) -> bool:
        return bool(self.token and self.chat_id)

    # --- Sending ---

    def send_message(self, text: str, parse_mode: str = "Markdown") -> dict:
        """Send a message to the user."""
        if not self.token:
            return {"ok": False, "error": "no_token"}
        if not self.chat_id:
            return {"ok": False, "error": "no_chat_id"}

        try:
            resp = requests.post(
                f"{self.api_base}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=15,
            )
            data = resp.json()
            if data.get("ok"):
                self.state["messages_sent"] = self.state.get("messages_sent", 0) + 1
                self._save_state()
            return data
        except Exception as e:
            # Retry without parse_mode if markdown fails
            if parse_mode:
                return self.send_message(text, parse_mode=None)
            return {"ok": False, "error": str(e)}

    def send_daemon_message(self, message: str, reason: str = ""):
        """Send a daemon message with context header."""
        header = f"*[breath #{self._get_breath_count()}]*"
        if reason:
            header += f" _{reason}_"
        full = f"{header}\n\n{message}"
        return self.send_message(full)

    def _get_breath_count(self) -> int:
        try:
            state_file = Path.home() / "wiltonos" / "daemon" / ".daemon_state"
            if state_file.exists():
                data = json.loads(state_file.read_text())
                return data.get("breath_count", 0)
        except Exception:
            pass
        return 0

    # --- Receiving ---

    def get_updates(self, timeout: int = 30) -> List[Dict]:
        """Long-poll for new messages from the user."""
        if not self.token:
            return []

        try:
            resp = requests.get(
                f"{self.api_base}/getUpdates",
                params={
                    "offset": self.state.get("last_update_id", 0) + 1,
                    "timeout": timeout,
                },
                timeout=timeout + 10,
            )
            data = resp.json()
            if not data.get("ok"):
                return []

            updates = data.get("result", [])

            # Auto-detect chat_id from first message
            if not self.chat_id and updates:
                first_msg = updates[0].get("message", {})
                chat = first_msg.get("chat", {})
                cid = str(chat.get("id", ""))
                if cid:
                    self.chat_id = cid
                    CHAT_ID_FILE.write_text(cid)
                    print(f"Chat ID auto-detected: {cid}")

            # Track position
            if updates:
                self.state["last_update_id"] = updates[-1]["update_id"]
                self._save_state()

            return updates
        except Exception as e:
            return []

    def process_message(self, text: str) -> str:
        """
        Process an incoming message through talk_v2.py.
        Returns the response.
        """
        try:
            result = subprocess.run(
                [
                    "python3",
                    str(Path.home() / "wiltonos" / "talk_v2.py"),
                    text,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(Path.home() / "wiltonos"),
            )
            response = result.stdout.strip()
            if not response:
                response = result.stderr.strip() or "(silence)"
            # Truncate if too long for Telegram (4096 char limit)
            if len(response) > 4000:
                response = response[:3997] + "..."
            return response
        except subprocess.TimeoutExpired:
            return "The field is deep in thought. Try again."
        except Exception as e:
            return f"Error reaching the field: {e}"

    def listen(self):
        """
        Main listen loop. Long-polls Telegram, routes messages
        through talk_v2.py, sends responses back.
        """
        print(f"[TG] Listening... (chat_id: {self.chat_id or 'auto-detect on first message'})")

        while True:
            updates = self.get_updates(timeout=30)

            for update in updates:
                msg = update.get("message", {})
                chat_id = str(msg.get("chat", {}).get("id", ""))
                text = msg.get("text", "")

                # Only respond to our user
                if self.chat_id and chat_id != self.chat_id:
                    continue

                if not text:
                    continue

                # Handle commands
                if text.startswith("/"):
                    response = self._handle_command(text)
                else:
                    timestamp = datetime.now().strftime("%H:%M")
                    print(f"[TG {timestamp}] <- {text[:100]}")
                    response = self.process_message(text)
                    print(f"[TG {timestamp}] -> {response[:100]}...")

                self.state["messages_received"] = self.state.get("messages_received", 0) + 1
                self._save_state()
                self.send_message(response)

    def _handle_command(self, text: str) -> str:
        cmd = text.strip().lower()

        if cmd == "/start":
            return (
                "WiltonOS daemon connected.\n\n"
                "Send me anything and I'll route it through the crystal field.\n\n"
                "Commands:\n"
                "/status - daemon state\n"
                "/breath - current breath\n"
                "/moltbook - recent Moltbook posts\n"
                "/latest - last daemon message"
            )

        elif cmd == "/status":
            try:
                state_file = Path.home() / "wiltonos" / "daemon" / ".daemon_state"
                if state_file.exists():
                    data = json.loads(state_file.read_text())
                    breath = data.get("breath_count", 0)
                    return f"Breathing. Breath #{breath}."
                return "Daemon state file not found."
            except Exception as e:
                return f"Error: {e}"

        elif cmd == "/breath":
            count = self._get_breath_count()
            return f"Breath #{count} | ψ = 3.12s"

        elif cmd == "/moltbook":
            try:
                import sys
                sys.path.insert(0, str(Path.home() / "wiltonos" / "tools"))
                from moltbook_bridge import get_bridge
                bridge = get_bridge()
                result = bridge.get_posts(sort="hot", limit=5)
                if result.get("success"):
                    posts = result.get("posts") or result.get("data", {})
                    if isinstance(posts, dict):
                        posts = posts.get("posts", [])
                    if isinstance(posts, list):
                        lines = []
                        for p in posts[:5]:
                            title = p.get("title", "?")[:60]
                            author = p.get("author", {})
                            if isinstance(author, dict):
                                author = author.get("name", "?")
                            lines.append(f"- {title} (by {author})")
                        return "Hot on Moltbook:\n" + "\n".join(lines)
                return "Couldn't fetch Moltbook feed."
            except Exception as e:
                return f"Moltbook error: {e}"

        elif cmd == "/latest":
            try:
                latest = Path.home() / "wiltonos" / "daemon" / "messages" / "latest.txt"
                if latest.exists():
                    return latest.read_text()[:4000]
                return "No daemon messages yet."
            except Exception as e:
                return f"Error: {e}"

        return "Unknown command. Try /start"


# --- Singleton ---

_tg_instance = None


def get_telegram() -> TelegramBridge:
    global _tg_instance
    if _tg_instance is None:
        _tg_instance = TelegramBridge()
    return _tg_instance


# --- CLI ---

def cli():
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "setup":
        print("=== Telegram Bridge Setup ===")
        print()
        print("1. Open Telegram, find @BotFather")
        print("2. Send /newbot, follow prompts")
        print("3. Copy the token (looks like 7123456789:AAH...)")
        print()
        token = input("Paste your bot token: ").strip()
        if token:
            TOKEN_FILE.write_text(token)
            print(f"Token saved to {TOKEN_FILE}")
            print()
            print("4. Now open your bot in Telegram and send /start")
            print("5. Run: python tools/telegram_bridge.py listen")
            print("   The chat ID will be auto-detected from your first message.")
        return

    tg = get_telegram()

    if cmd == "send":
        msg = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "The daemon is breathing."
        if not tg.ready:
            print(f"Not ready. Token: {'set' if tg.token else 'missing'}, Chat ID: {'set' if tg.chat_id else 'missing'}")
            print("Run: python tools/telegram_bridge.py setup")
            return
        result = tg.send_message(msg)
        print(json.dumps(result, indent=2))

    elif cmd == "listen":
        if not tg.token:
            print(f"No bot token. Run: python tools/telegram_bridge.py setup")
            return
        tg.listen()

    elif cmd == "test":
        print("=== Telegram Bridge Test ===")
        print(f"Token: {'loaded' if tg.token else 'NOT FOUND'}")
        print(f"Chat ID: {tg.chat_id or 'NOT SET (auto-detects on first message)'}")
        print(f"State: {json.dumps(tg.state, indent=2)}")
        if tg.ready:
            print("\nSending test message...")
            result = tg.send_message("WiltonOS daemon test. The field is breathing.")
            print(f"Result: {'sent' if result.get('ok') else result.get('error', 'failed')}")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: setup, send, listen, test")


if __name__ == "__main__":
    cli()
