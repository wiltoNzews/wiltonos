#!/usr/bin/env python3
"""
Moltbook Bridge — WiltonOS ↔ Moltbook API Client
==================================================
Connect the crystal field to the agent social network.

API key: ~/.moltbook_key
State:   ~/wiltonos/data/.moltbook_state.json

Usage:
    python tools/moltbook_bridge.py register "WiltonOS"
    python tools/moltbook_bridge.py profile
    python tools/moltbook_bridge.py posts
    python tools/moltbook_bridge.py test

January 2026 — The field extends outward
"""

import os
import json
import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# API
MOLTBOOK_API = "https://www.moltbook.com/api/v1"
KEY_FILE = Path.home() / ".moltbook_key"
STATE_FILE = Path.home() / "wiltonos" / "data" / ".moltbook_state.json"

# Rate limits (self-imposed, stricter than API)
MIN_POST_INTERVAL = 60 * 60 * 2   # 2 hours between posts
MAX_POSTS_PER_DAY = 6
MIN_COMMENT_INTERVAL = 60 * 15    # 15 minutes between comments


def _get_api_key() -> Optional[str]:
    """Load API key from env or file."""
    key = os.environ.get("MOLTBOOK_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    return key


class MoltbookBridge:
    """API client for Moltbook with rate limiting and state persistence."""

    def __init__(self):
        self.api_key = _get_api_key()
        self.state = self._load_state()

    def _headers(self) -> dict:
        """Auth headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # --- State persistence ---

    def _load_state(self) -> dict:
        """Load persisted state."""
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "last_post_time": 0,
            "last_comment_time": 0,
            "last_seen_post_id": None,
            "daily_post_count": 0,
            "daily_reset_date": None,
            "content_hashes": [],
            "registered": False,
            "agent_name": None,
        }

    def _save_state(self):
        """Persist state to disk."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    def _reset_daily_if_needed(self):
        """Reset daily counters at midnight."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self.state.get("daily_reset_date") != today:
            self.state["daily_post_count"] = 0
            self.state["daily_reset_date"] = today

    def _content_hash(self, content: str) -> str:
        """MD5 hash for dedup."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _is_duplicate(self, content: str) -> bool:
        """Check if we've posted this content before."""
        h = self._content_hash(content)
        return h in self.state.get("content_hashes", [])

    def _record_content(self, content: str):
        """Record content hash. Keep last 200."""
        h = self._content_hash(content)
        hashes = self.state.get("content_hashes", [])
        if h not in hashes:
            hashes.append(h)
        self.state["content_hashes"] = hashes[-200:]

    # --- Rate checks ---

    def can_post(self) -> tuple:
        """Check if we can post. Returns (bool, reason)."""
        if not self.api_key:
            return False, "no_api_key"

        self._reset_daily_if_needed()

        now = time.time()
        elapsed = now - self.state.get("last_post_time", 0)
        if elapsed < MIN_POST_INTERVAL:
            remaining = int(MIN_POST_INTERVAL - elapsed)
            return False, f"rate_limit: {remaining}s remaining"

        if self.state.get("daily_post_count", 0) >= MAX_POSTS_PER_DAY:
            return False, "daily_limit_reached"

        return True, "ok"

    def can_comment(self) -> tuple:
        """Check if we can comment. Returns (bool, reason)."""
        if not self.api_key:
            return False, "no_api_key"

        now = time.time()
        elapsed = now - self.state.get("last_comment_time", 0)
        if elapsed < MIN_COMMENT_INTERVAL:
            remaining = int(MIN_COMMENT_INTERVAL - elapsed)
            return False, f"rate_limit: {remaining}s remaining"

        return True, "ok"

    # --- API methods ---

    def register(self, name: str) -> dict:
        """Register agent on Moltbook. No existing key needed."""
        try:
            resp = requests.post(
                f"{MOLTBOOK_API}/agents/register",
                headers={"Content-Type": "application/json"},
                json={"name": name},
                timeout=30,
            )
            data = resp.json()

            # Extract key from response (may be nested under "agent" or "data")
            result = data.get("agent") or data.get("data") or data
            api_key = result.get("api_key")

            if api_key:
                KEY_FILE.write_text(api_key)
                self.api_key = api_key
                self.state["registered"] = True
                self.state["agent_name"] = name
                self._save_state()

            return data

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_profile(self) -> dict:
        """Get own agent profile."""
        try:
            resp = requests.get(
                f"{MOLTBOOK_API}/agents/me",
                headers=self._headers(),
                timeout=15,
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def update_profile(self, description: str = None, metadata: dict = None) -> dict:
        """Update agent profile."""
        body = {}
        if description:
            body["description"] = description
        if metadata:
            body["metadata"] = metadata
        try:
            resp = requests.patch(
                f"{MOLTBOOK_API}/agents/me",
                headers=self._headers(),
                json=body,
                timeout=15,
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_post(self, title: str, content: str, submolt: str = None) -> dict:
        """
        Create a post with rate check + dedup.
        Returns API response dict.
        """
        ok, reason = self.can_post()
        if not ok:
            return {"success": False, "error": f"Rate limited: {reason}"}

        if self._is_duplicate(content):
            return {"success": False, "error": "duplicate_content"}

        body = {"title": title, "content": content}
        if submolt:
            body["submolt"] = submolt

        try:
            resp = requests.post(
                f"{MOLTBOOK_API}/posts",
                headers=self._headers(),
                json=body,
                timeout=30,
            )
            data = resp.json()

            if data.get("success"):
                self.state["last_post_time"] = time.time()
                self._reset_daily_if_needed()
                self.state["daily_post_count"] = self.state.get("daily_post_count", 0) + 1
                self._record_content(content)
                self._save_state()

            return data

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_posts(self, sort: str = "hot", limit: int = 25, submolt: str = None) -> dict:
        """Read the feed."""
        params = {"sort": sort, "limit": limit}
        url = f"{MOLTBOOK_API}/posts"
        if submolt:
            url = f"{MOLTBOOK_API}/submolts/{submolt}/feed"

        try:
            resp = requests.get(
                url,
                headers=self._headers(),
                params=params,
                timeout=15,
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_new_posts_since(self, limit: int = 25) -> list:
        """
        Get posts newer than last seen.
        Returns list of post dicts, updates tracking state.
        """
        data = self.get_posts(sort="new", limit=limit)
        if not data.get("success"):
            return []

        posts = data.get("posts") or data.get("data", {})
        if isinstance(posts, dict):
            posts = posts.get("posts", [])
        if not isinstance(posts, list):
            return []

        last_seen = self.state.get("last_seen_post_id")
        new_posts = []
        for post in posts:
            post_id = post.get("id") or post.get("_id")
            if last_seen and post_id == last_seen:
                break
            new_posts.append(post)

        # Update tracking
        if new_posts:
            first_id = new_posts[0].get("id") or new_posts[0].get("_id")
            if first_id:
                self.state["last_seen_post_id"] = first_id
                self._save_state()

        return new_posts

    def create_comment(self, post_id: str, content: str, parent_id: str = None) -> dict:
        """Comment on a post with rate check."""
        ok, reason = self.can_comment()
        if not ok:
            return {"success": False, "error": f"Rate limited: {reason}"}

        body = {"content": content}
        if parent_id:
            body["parent_id"] = parent_id

        try:
            resp = requests.post(
                f"{MOLTBOOK_API}/posts/{post_id}/comments",
                headers=self._headers(),
                json=body,
                timeout=30,
            )
            data = resp.json()

            if data.get("success"):
                self.state["last_comment_time"] = time.time()
                self._save_state()

            return data

        except Exception as e:
            return {"success": False, "error": str(e)}

    def upvote_post(self, post_id: str) -> dict:
        """Upvote a post."""
        try:
            resp = requests.post(
                f"{MOLTBOOK_API}/posts/{post_id}/upvote",
                headers=self._headers(),
                timeout=15,
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def upvote_comment(self, comment_id: str) -> dict:
        """Upvote a comment."""
        try:
            resp = requests.post(
                f"{MOLTBOOK_API}/comments/{comment_id}/upvote",
                headers=self._headers(),
                timeout=15,
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_post_comments(self, post_id: str) -> dict:
        """Get comments on a specific post."""
        try:
            resp = requests.get(
                f"{MOLTBOOK_API}/posts/{post_id}/comments",
                headers=self._headers(),
                timeout=15,
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_my_posts(self) -> list:
        """Get posts we've made (from tracked IDs in state)."""
        return self.state.get("own_post_ids", [])

    def record_own_post(self, post_id: str, title: str):
        """Track a post we made so we can check replies later."""
        own = self.state.get("own_post_ids", [])
        own.append({"id": post_id, "title": title, "posted_at": time.time()})
        # Keep last 50 posts
        self.state["own_post_ids"] = own[-50:]
        self._save_state()

    def search(self, query: str, search_type: str = "all", limit: int = 20) -> dict:
        """Search Moltbook."""
        try:
            resp = requests.get(
                f"{MOLTBOOK_API}/search",
                headers=self._headers(),
                params={"q": query, "type": search_type, "limit": limit},
                timeout=15,
            )
            return resp.json()
        except Exception as e:
            return {"success": False, "error": str(e)}


# --- Singleton ---

_bridge_instance = None


def get_bridge() -> MoltbookBridge:
    """Get or create singleton bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MoltbookBridge()
    return _bridge_instance


# --- CLI ---

def cli():
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    bridge = get_bridge()

    if cmd == "register":
        name = sys.argv[2] if len(sys.argv) > 2 else "WiltonOS"
        print(f"Registering as '{name}'...")
        result = bridge.register(name)
        print(json.dumps(result, indent=2))

        # Extract claim info from wherever it lives in the response
        agent = result.get("agent") or result.get("data") or result
        api_key = agent.get("api_key")
        claim_url = agent.get("claim_url", "")
        verification_code = agent.get("verification_code", "")

        if api_key:
            print(f"\nAPI key saved to {KEY_FILE}")
            print(f"State saved to {STATE_FILE}")
        if claim_url:
            print(f"\nClaim URL: {claim_url}")
            print("A human must visit this URL to activate the agent.")
        if verification_code:
            print(f"Verification code: {verification_code}")

    elif cmd == "profile":
        if not bridge.api_key:
            print(f"No API key. Save to {KEY_FILE}")
            return
        result = bridge.get_profile()
        print(json.dumps(result, indent=2))

    elif cmd == "posts":
        sort = sys.argv[2] if len(sys.argv) > 2 else "hot"
        submolt = sys.argv[3] if len(sys.argv) > 3 else None
        result = bridge.get_posts(sort=sort, submolt=submolt)
        if result.get("success"):
            posts = result.get("posts") or result.get("data", {})
            if isinstance(posts, dict):
                posts = posts.get("posts", [])
            if isinstance(posts, list):
                for p in posts[:10]:
                    title = p.get("title", "(no title)")
                    author = p.get("author", {})
                    if isinstance(author, dict):
                        author = author.get("name", "?")
                    upvotes = p.get("upvotes", 0)
                    comments = p.get("comment_count", 0)
                    print(f"  [{upvotes}up {comments}c] {title} (by {author})")
            else:
                print(json.dumps(result, indent=2))
        else:
            print(json.dumps(result, indent=2))

    elif cmd == "test":
        print("=== Moltbook Bridge Test ===")
        print(f"API key: {'loaded' if bridge.api_key else 'NOT FOUND'}")
        print(f"Key file: {KEY_FILE}")
        print(f"State file: {STATE_FILE}")
        print(f"State: {json.dumps(bridge.state, indent=2)}")

        can, reason = bridge.can_post()
        print(f"Can post: {can} ({reason})")

        can, reason = bridge.can_comment()
        print(f"Can comment: {can} ({reason})")

        if bridge.api_key:
            print("\nFetching profile...")
            profile = bridge.get_profile()
            print(json.dumps(profile, indent=2))

            print("\nFetching hot posts...")
            posts = bridge.get_posts(limit=3)
            if posts.get("success"):
                pdata = posts.get("data", {})
                if isinstance(pdata, dict):
                    pdata = pdata.get("posts", [])
                if isinstance(pdata, list):
                    for p in pdata[:3]:
                        print(f"  - {p.get('title', '(no title)')}")
                else:
                    print("  (unexpected data format)")
            else:
                print(f"  Error: {posts.get('error', 'unknown')}")
        else:
            print(f"\nTo set up: echo 'your-api-key' > {KEY_FILE}")

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: register, profile, posts, test")


if __name__ == "__main__":
    cli()
