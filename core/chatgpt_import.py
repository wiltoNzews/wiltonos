#!/usr/bin/env python3
"""
ChatGPT Import - Consolidated Living Memory Importer
=====================================================
Extracts conversations from ChatGPT exports into crystals.

This isn't just data import. It's memory convergence.
The conversations map the journey. Both sides matter.

Key features:
- Incremental import (only new conversations)
- Both sides preserved (user + assistant = full dialogue)
- Conversation threading (conv_id for "show me that conversation")
- 3-axis scoring (temporal/ontological/depth)
- Semantic embedding for search

Usage:
    python chatgpt_import.py path/to/conversations.json
    python chatgpt_import.py --since 2025-12-20
    python chatgpt_import.py --dry-run

January 2026 - Memory converging
"""

import sys
import json
import sqlite3
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Generator
from dataclasses import dataclass
import logging

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from memory_service import MemoryService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message from a conversation."""
    content: str
    author: str  # 'user', 'assistant', 'system', 'tool'
    timestamp: float
    conv_id: str
    conv_title: str
    node_id: str
    turn_index: int  # Position in conversation

    def content_hash(self) -> str:
        """Generate hash for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()

    def to_crystal_content(self) -> str:
        """Format for crystal storage - preserving the voice."""
        role_marker = {
            'user': 'U',
            'assistant': 'A',
            'system': 'S',
            'tool': 'T'
        }.get(self.author, '?')
        return f"[{role_marker}] {self.content}"


@dataclass
class Conversation:
    """A full conversation with all messages."""
    id: str
    title: str
    create_time: float
    messages: List[Message]

    def timestamp_dt(self) -> datetime:
        """Convert timestamp to datetime."""
        if self.create_time:
            return datetime.fromtimestamp(self.create_time, tz=timezone.utc)
        return datetime.now(timezone.utc)


class ChatGPTImporter:
    """
    Import ChatGPT conversations as crystals.

    Preserves the dialogue structure while fitting into the crystal model.
    """

    def __init__(self, db_path: Path = None, user_id: str = "wilton"):
        self.db_path = db_path or Path.home() / "wiltonos" / "data" / "crystals_unified.db"
        self.user_id = user_id
        self.memory = MemoryService()
        self.stats = {
            'conversations_processed': 0,
            'messages_extracted': 0,
            'crystals_created': 0,
            'duplicates_skipped': 0,
            'errors': 0
        }

    def parse_conversation(self, conv: Dict) -> Optional[Conversation]:
        """Parse a single conversation from ChatGPT export format."""
        try:
            conv_id = conv.get('conversation_id', conv.get('id', ''))
            title = conv.get('title', 'Untitled')
            create_time = conv.get('create_time', 0)
            mapping = conv.get('mapping', {})

            if not mapping:
                return None

            # Build message chain - need to follow parent pointers
            # to get proper ordering
            messages = []
            node_order = self._order_nodes(mapping)

            for idx, node_id in enumerate(node_order):
                node = mapping.get(node_id)
                if not node:
                    continue

                msg_data = node.get('message')
                if not msg_data:
                    continue

                author_data = msg_data.get('author')
                if not author_data:
                    continue

                author = author_data.get('role', 'unknown')
                content_data = msg_data.get('content')
                if not content_data:
                    continue

                content = self._extract_content(content_data)

                if not content or len(content.strip()) < 10:
                    continue

                msg_time = msg_data.get('create_time', create_time) or create_time

                messages.append(Message(
                    content=content,
                    author=author,
                    timestamp=msg_time or 0,
                    conv_id=conv_id,
                    conv_title=title,
                    node_id=node_id,
                    turn_index=idx
                ))

            if not messages:
                return None

            return Conversation(
                id=conv_id,
                title=title,
                create_time=create_time or 0,
                messages=messages
            )

        except Exception as e:
            logger.error(f"Error parsing conversation: {e}")
            self.stats['errors'] += 1
            return None

    def _order_nodes(self, mapping: Dict) -> List[str]:
        """Order nodes by following the tree structure (iterative to avoid recursion limits)."""
        # Find root node (no parent)
        root_id = None
        children = {}  # parent_id -> [child_ids]

        for node_id, node in mapping.items():
            if not node:  # Skip None nodes
                continue
            parent = node.get('parent')
            if parent is None:
                root_id = node_id
            else:
                if parent not in children:
                    children[parent] = []
                children[parent].append(node_id)

        if not root_id:
            # Fallback: sort by create_time if no tree structure
            nodes_with_time = []
            for node_id, node in mapping.items():
                if not node:
                    continue
                msg = node.get('message') or {}
                create_time = msg.get('create_time', 0) or 0
                nodes_with_time.append((create_time, node_id))
            nodes_with_time.sort()
            return [n[1] for n in nodes_with_time]

        # Iterative DFS (stack-based to avoid recursion limit)
        order = []
        stack = [root_id]

        while stack:
            node_id = stack.pop()
            order.append(node_id)
            # Add children in reverse order so first child is processed first
            for child in reversed(children.get(node_id, [])):
                stack.append(child)

        return order

    def _extract_content(self, content: Dict) -> str:
        """Extract text content from message content structure."""
        if isinstance(content, str):
            return content

        parts = content.get('parts', [])
        text_parts = []

        for part in parts:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                if 'text' in part:
                    text_parts.append(part['text'])
                elif 'content' in part:
                    text_parts.append(part['content'])

        return '\n'.join(text_parts).strip()

    def stream_conversations(
        self,
        json_path: Path,
        since: datetime = None
    ) -> Generator[Conversation, None, None]:
        """Stream conversations from JSON file (memory efficient)."""
        logger.info(f"Opening {json_path.name} ({json_path.stat().st_size / 1024 / 1024:.1f} MB)")

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error("Expected list of conversations")
            return

        logger.info(f"Found {len(data)} conversations in export")

        for conv_data in data:
            conv = self.parse_conversation(conv_data)
            if not conv:
                continue

            # Filter by date if specified
            if since:
                conv_dt = conv.timestamp_dt()
                if conv_dt < since:
                    continue

            self.stats['conversations_processed'] += 1
            yield conv

    def check_duplicate(self, content_hash: str) -> bool:
        """Check if a crystal with this content hash already exists."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        # Check by content_hash column (already exists in schema)
        c.execute("""
            SELECT COUNT(*) FROM crystals
            WHERE content_hash = ?
            AND user_id = ?
        """, (content_hash, self.user_id))

        exists = c.fetchone()[0] > 0
        conn.close()
        return exists

    def store_message(
        self,
        msg: Message,
        dry_run: bool = False,
        embed: bool = True
    ) -> Optional[int]:
        """Store a message as a crystal."""
        content = msg.to_crystal_content()
        content_hash = msg.content_hash()

        # Check for duplicates
        if self.check_duplicate(content_hash):
            self.stats['duplicates_skipped'] += 1
            return None

        if dry_run:
            logger.info(f"  [DRY RUN] Would store: [{msg.author}] {content[:60]}...")
            self.stats['crystals_created'] += 1
            return -1

        # Determine mode based on author
        mode = 'wiltonos' if msg.author == 'user' else 'neutral'

        # Store crystal using actual schema
        try:
            embedding = self.memory.get_embedding(content) if embed else None

            conn = sqlite3.connect(str(self.db_path))
            c = conn.cursor()

            # Get timestamp
            created_at = datetime.fromtimestamp(msg.timestamp) if msg.timestamp else datetime.now()

            # Use actual schema columns
            c.execute("""
                INSERT INTO crystals (
                    user_id, content, content_hash, source, source_file,
                    mode, emotion, author, original_timestamp,
                    created_at, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id,
                content,
                content_hash,
                'chatgpt_export',
                f"{msg.conv_id}:{msg.turn_index}",  # conv_id:turn as source_file
                mode,
                'clarity',  # Default, scorer will update
                msg.author,
                msg.timestamp,
                created_at.isoformat(),
                embedding
            ))

            crystal_id = c.lastrowid
            conn.commit()
            conn.close()

            self.stats['crystals_created'] += 1
            return crystal_id

        except Exception as e:
            logger.error(f"Error storing crystal: {e}")
            self.stats['errors'] += 1
            return None

    def store_conversation_pair(
        self,
        conv: Conversation,
        dry_run: bool = False,
        embed: bool = True
    ) -> List[int]:
        """
        Store conversation as paired exchanges.

        Each user message + following assistant response = one crystal.
        This preserves the dialogue flow while reducing fragmentation.
        """
        crystal_ids = []
        messages = conv.messages

        i = 0
        while i < len(messages):
            msg = messages[i]

            # If user message, try to pair with following assistant response
            if msg.author == 'user' and i + 1 < len(messages):
                next_msg = messages[i + 1]
                if next_msg.author == 'assistant':
                    # Create paired content
                    paired_content = f"[U] {msg.content}\n\n[A] {next_msg.content}"
                    content_hash = hashlib.md5(paired_content.encode()).hexdigest()

                    if not self.check_duplicate(content_hash):
                        if dry_run:
                            logger.info(f"  [DRY RUN] Paired: {msg.content[:40]}... -> {next_msg.content[:40]}...")
                            self.stats['crystals_created'] += 1
                        else:
                            crystal_id = self._store_paired_crystal(
                                msg, next_msg, paired_content, content_hash, embed
                            )
                            if crystal_id:
                                crystal_ids.append(crystal_id)
                    else:
                        self.stats['duplicates_skipped'] += 1

                    self.stats['messages_extracted'] += 2
                    i += 2
                    continue

            # Store standalone message
            crystal_id = self.store_message(msg, dry_run, embed)
            if crystal_id:
                crystal_ids.append(crystal_id)
            self.stats['messages_extracted'] += 1
            i += 1

        return crystal_ids

    def _store_paired_crystal(
        self,
        user_msg: Message,
        asst_msg: Message,
        content: str,
        content_hash: str,
        embed: bool
    ) -> Optional[int]:
        """Store a user+assistant pair as one crystal."""
        try:
            embedding = self.memory.get_embedding(content) if embed else None
            created_at = datetime.fromtimestamp(user_msg.timestamp) if user_msg.timestamp else datetime.now()

            conn = sqlite3.connect(str(self.db_path))
            c = conn.cursor()

            # Use actual schema columns
            # source_file = conv_id:user_turn-asst_turn (for threading)
            c.execute("""
                INSERT INTO crystals (
                    user_id, content, content_hash, source, source_file,
                    mode, emotion, author, original_timestamp,
                    created_at, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.user_id,
                content,
                content_hash,
                'chatgpt_export',
                f"{user_msg.conv_id}:{user_msg.turn_index}-{asst_msg.turn_index}",
                'wiltonos',  # Dialogue is personal
                'clarity',
                'paired',  # Special author type for paired messages
                user_msg.timestamp,
                created_at.isoformat(),
                embedding
            ))

            crystal_id = c.lastrowid
            conn.commit()
            conn.close()

            self.stats['crystals_created'] += 1
            return crystal_id

        except Exception as e:
            logger.error(f"Error storing paired crystal: {e}")
            self.stats['errors'] += 1
            return None

    def import_file(
        self,
        json_path: Path,
        since: datetime = None,
        dry_run: bool = False,
        embed: bool = True,
        pair_messages: bool = True
    ) -> Dict:
        """
        Import a ChatGPT export file.

        Args:
            json_path: Path to conversations.json
            since: Only import conversations after this date
            dry_run: Don't actually store, just count
            embed: Generate embeddings (slower but enables search)
            pair_messages: Combine user+assistant as paired crystals
        """
        logger.info(f"Starting import from {json_path.name}")
        if since:
            logger.info(f"Filtering for conversations after {since.isoformat()}")
        if dry_run:
            logger.info("DRY RUN - no changes will be made")

        for conv in self.stream_conversations(json_path, since):
            logger.info(f"Processing: {conv.title[:50]}... ({len(conv.messages)} messages)")

            if pair_messages:
                self.store_conversation_pair(conv, dry_run, embed)
            else:
                for msg in conv.messages:
                    self.store_message(msg, dry_run, embed)
                    self.stats['messages_extracted'] += 1

        return self.stats

    def get_last_import_date(self) -> Optional[datetime]:
        """Get the date of the most recent ChatGPT import."""
        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()

        c.execute("""
            SELECT MAX(created_at) FROM crystals
            WHERE metadata LIKE '%chatgpt_export%'
            AND user_id = ?
        """, (self.user_id,))

        result = c.fetchone()[0]
        conn.close()

        if result:
            try:
                return datetime.fromisoformat(result.replace('Z', '+00:00'))
            except:
                return None
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Import ChatGPT conversations as crystals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Import new conversations
    python chatgpt_import.py ~/rag-local/ChatGPT_Exports/conversations8.json

    # Only conversations after a date
    python chatgpt_import.py conversations.json --since 2025-12-20

    # Dry run to see what would be imported
    python chatgpt_import.py conversations.json --dry-run

    # Fast import without embeddings
    python chatgpt_import.py conversations.json --no-embed
        """
    )

    parser.add_argument("json_path", type=Path, help="Path to conversations.json")
    parser.add_argument("--since", type=str, help="Only import after this date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true", help="Don't store, just count")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding generation")
    parser.add_argument("--no-pair", action="store_true", help="Store messages individually, not paired")
    parser.add_argument("--user", type=str, default="wilton", help="User ID")
    parser.add_argument("--auto", action="store_true", help="Auto-detect since date from last import")

    args = parser.parse_args()

    if not args.json_path.exists():
        print(f"File not found: {args.json_path}")
        sys.exit(1)

    importer = ChatGPTImporter(user_id=args.user)

    # Determine since date
    since = None
    if args.auto:
        since = importer.get_last_import_date()
        if since:
            logger.info(f"Auto-detected last import: {since.isoformat()}")
    elif args.since:
        since = datetime.fromisoformat(args.since)
        # Make timezone-aware
        since = since.replace(tzinfo=timezone.utc)

    # Run import
    stats = importer.import_file(
        json_path=args.json_path,
        since=since,
        dry_run=args.dry_run,
        embed=not args.no_embed,
        pair_messages=not args.no_pair
    )

    # Print summary
    print("\n" + "=" * 60)
    print("IMPORT COMPLETE")
    print("=" * 60)
    print(f"Conversations processed: {stats['conversations_processed']}")
    print(f"Messages extracted:      {stats['messages_extracted']}")
    print(f"Crystals created:        {stats['crystals_created']}")
    print(f"Duplicates skipped:      {stats['duplicates_skipped']}")
    print(f"Errors:                  {stats['errors']}")
    print("=" * 60)

    if args.dry_run:
        print("\nThis was a DRY RUN. No changes were made.")
        print("Remove --dry-run to actually import.")


if __name__ == "__main__":
    main()
