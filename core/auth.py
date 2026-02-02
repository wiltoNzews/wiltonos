"""
WiltonOS Auth Module
====================
Simple password-based auth for the gateway.
Stores bcrypt-hashed passwords in the database.

Usage:
    from auth import UserAuth
    auth = UserAuth()
    auth.create_user('michelle', 'her_password')
    if auth.verify('michelle', 'her_password'):
        # grant access
"""

import sqlite3
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict


class UserAuth:
    """Simple user authentication."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path or "/home/zews/wiltonos/data/crystals_unified.db")
        self._ensure_table()

    def _ensure_table(self):
        """Create users table if needed."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using SHA-256."""
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def create_user(self, username: str, password: str) -> bool:
        """
        Create a new user.
        Returns True if created, False if username exists.
        """
        username = username.lower().strip()
        salt = secrets.token_hex(16)
        password_hash = self._hash_password(password, salt)

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        try:
            c.execute("""
                INSERT INTO users (username, password_hash, salt)
                VALUES (?, ?, ?)
            """, (username, password_hash, salt))
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False

    def verify(self, username: str, password: str) -> bool:
        """
        Verify username/password combination.
        Returns True if valid, False otherwise.
        """
        username = username.lower().strip()

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT password_hash, salt FROM users WHERE username = ?
        """, (username,))
        row = c.fetchone()

        if not row:
            conn.close()
            return False

        stored_hash, salt = row
        test_hash = self._hash_password(password, salt)

        if test_hash == stored_hash:
            # Update last login
            c.execute("""
                UPDATE users SET last_login = CURRENT_TIMESTAMP
                WHERE username = ?
            """, (username,))
            conn.commit()
            conn.close()
            return True

        conn.close()
        return False

    def user_exists(self, username: str) -> bool:
        """Check if user exists."""
        username = username.lower().strip()
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        exists = c.fetchone() is not None
        conn.close()
        return exists

    def change_password(self, username: str, new_password: str) -> bool:
        """Change user's password."""
        username = username.lower().strip()
        if not self.user_exists(username):
            return False

        salt = secrets.token_hex(16)
        password_hash = self._hash_password(new_password, salt)

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("""
            UPDATE users SET password_hash = ?, salt = ?
            WHERE username = ?
        """, (password_hash, salt, username))
        conn.commit()
        conn.close()
        return True

    def list_users(self) -> list:
        """List all usernames."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        c.execute("SELECT username, created_at, last_login FROM users")
        users = [{'username': r[0], 'created_at': r[1], 'last_login': r[2]}
                 for r in c.fetchall()]
        conn.close()
        return users


# CLI for managing users
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WiltonOS User Management")
    parser.add_argument("action", choices=["create", "list", "verify", "passwd"])
    parser.add_argument("--user", "-u", help="Username")
    parser.add_argument("--password", "-p", help="Password")
    args = parser.parse_args()

    auth = UserAuth()

    if args.action == "create":
        if not args.user or not args.password:
            print("Usage: python auth.py create -u USERNAME -p PASSWORD")
        else:
            if auth.create_user(args.user, args.password):
                print(f"✓ User '{args.user}' created")
            else:
                print(f"✗ User '{args.user}' already exists")

    elif args.action == "list":
        users = auth.list_users()
        print(f"=== Users ({len(users)}) ===")
        for u in users:
            print(f"  {u['username']} (created: {u['created_at']}, last: {u['last_login']})")

    elif args.action == "verify":
        if not args.user or not args.password:
            print("Usage: python auth.py verify -u USERNAME -p PASSWORD")
        else:
            if auth.verify(args.user, args.password):
                print(f"✓ Valid credentials for '{args.user}'")
            else:
                print(f"✗ Invalid credentials")

    elif args.action == "passwd":
        if not args.user or not args.password:
            print("Usage: python auth.py passwd -u USERNAME -p NEWPASSWORD")
        else:
            if auth.change_password(args.user, args.password):
                print(f"✓ Password changed for '{args.user}'")
            else:
                print(f"✗ User '{args.user}' not found")
