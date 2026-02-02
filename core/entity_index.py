#!/usr/bin/env python3
"""
Entity Index
============
Tracks who/what the system knows about per user.
Three-pass extraction (no LLM): known entities, relationship patterns, proper nouns.

Tables: entity_mentions, entity_mention_log

Usage:
    from entity_index import EntityIndex
    ei = EntityIndex(db_path)
    entities = ei.extract_entities("Juliana called me", "wilton")

Bootstrap existing crystals:
    python core/entity_index.py --bootstrap
"""

import re
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Set

_DEFAULT_DB = str(Path.home() / "wiltonos" / "data" / "crystals_unified.db")

# Relationship patterns: (regex, entity_type)
# Group 1 captures the entity name or relationship label
RELATIONSHIP_PATTERNS = [
    # "my wife/husband/friend/etc" — captures the relationship word
    (re.compile(
        r"\bmy (wife|husband|girlfriend|boyfriend|partner|ex|sister|brother|"
        r"mom|dad|mother|father|friend|boss|coworker|therapist|son|daughter|"
        r"uncle|aunt|cousin|grandma|grandmother|grandpa|grandfather|"
        r"fiancée?|roommate|neighbor)\b", re.IGNORECASE
    ), "person"),
    # "a friend named/called X" — captures the name
    (re.compile(
        r"\b(?:a|my) (?:friend|colleague|coworker) (?:named |called )?([A-Z][a-záàâãéèêíïóôõúç]+)",
    ), "person"),
    # "moved to / living in / went to PLACE"
    (re.compile(
        r"\b(?:moved to|living in|went to|came from|flew to|visiting|back in) "
        r"([A-Z][a-záàâãéèêíïóôõúç]+(?:\s[A-Z][a-záàâãéèêíïóôõúç]+)?)",
    ), "place"),
    # "working on / building / started PROJECT"
    (re.compile(
        r"\b(?:working on|building|started|launched|shipped|released) "
        r"([A-Z][a-záàâãéèêíïóôõúç]+(?:\s[A-Z][a-záàâãéèêíïóôõúç]+)?)",
    ), "project"),
]

# Words to exclude from proper noun detection
_STOP_WORDS = {
    "i", "the", "a", "an", "this", "that", "my", "your", "his", "her",
    "we", "they", "it", "is", "was", "are", "were", "be", "been",
    "but", "and", "or", "not", "so", "if", "when", "then", "just",
    "now", "here", "there", "what", "how", "why", "who", "where",
    "yes", "no", "ok", "okay", "yeah", "sure", "well", "like",
    "really", "actually", "maybe", "probably", "definitely",
    "today", "yesterday", "tomorrow", "monday", "tuesday", "wednesday",
    "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "about", "after", "before", "between", "during", "from",
    "into", "over", "through", "under", "with", "without",
    "also", "very", "much", "more", "most", "some", "any",
    "each", "every", "all", "both", "few", "many",
    "could", "would", "should", "might", "must", "will", "shall",
    "had", "has", "have", "did", "does", "do", "can",
    "let", "got", "get", "been", "being", "having",
    "said", "told", "asked", "went", "came", "made", "took",
    "still", "already", "even", "though", "because", "since",
    "something", "nothing", "everything", "anything",
    "someone", "nobody", "everybody", "anyone",
    "however", "therefore", "instead", "otherwise",
    # Common sentence starters that aren't entities
    "anyway", "anyway", "honestly", "basically", "literally",
    # WiltonOS terms that aren't entities
    "wiltonos", "psi", "zeta", "omega", "crystal", "glyph",
}

# Sentence boundary pattern
_SENTENCE_START = re.compile(r'(?:^|[.!?]\s+)([A-Z])')


class EntityIndex:
    """Per-user entity tracking with three-pass extraction."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or _DEFAULT_DB
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                display_name TEXT,
                crystal_id INTEGER,
                mention_context TEXT,
                sentiment TEXT,
                first_seen TEXT,
                last_seen TEXT,
                mention_count INTEGER DEFAULT 1,
                is_active INTEGER DEFAULT 1,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_mentions_user_name
            ON entity_mentions(user_id, entity_name)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_mentions_user_seen
            ON entity_mentions(user_id, last_seen)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_mentions_user_type
            ON entity_mentions(user_id, entity_type)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS entity_mention_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                crystal_id INTEGER,
                mention_context TEXT,
                sentiment TEXT,
                wound_active TEXT,
                emotion_active TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_entity_log_user_name_ts
            ON entity_mention_log(user_id, entity_name, timestamp)
        """)
        conn.commit()
        conn.close()

    def _get_known_entities(self, user_id: str) -> Dict[str, Dict]:
        """Load all known entities for a user, keyed by normalized name."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM entity_mentions WHERE user_id = ? AND is_active = 1",
            (user_id,),
        ).fetchall()
        conn.close()
        result = {}
        for r in rows:
            result[r['entity_name']] = dict(r)
        return result

    def extract_entities(self, text: str, user_id: str) -> List[Dict]:
        """
        Three-pass entity extraction (no LLM).

        Pass 1: Match against known entities for this user.
        Pass 2: Relationship pattern matching ("my wife", "a friend named X").
        Pass 3: Proper noun heuristic (capitalized words not at sentence start).

        Returns list of dicts: [{name, type, confidence, context}]
        """
        if not text or not text.strip():
            return []

        found = {}  # name_lower -> {name, type, confidence, context}
        text_lower = text.lower()

        # Pass 1: Known entity matching
        known = self._get_known_entities(user_id)
        for name_lower, info in known.items():
            if name_lower in text_lower:
                # Verify it's a word boundary match
                pattern = re.compile(r'\b' + re.escape(name_lower) + r'\b', re.IGNORECASE)
                match = pattern.search(text)
                if match:
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end].strip()
                    found[name_lower] = {
                        'name': info.get('display_name') or name_lower,
                        'type': info['entity_type'],
                        'confidence': 0.95,
                        'context': context,
                    }

        # Pass 2: Relationship pattern matching
        for pattern, entity_type in RELATIONSHIP_PATTERNS:
            for match in pattern.finditer(text):
                captured = match.group(1).strip()
                name_lower_cap = captured.lower()

                # Skip if already found
                if name_lower_cap in found:
                    continue

                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end].strip()

                # Relationship words (wife, dad, etc.) become relationship type, not name
                if name_lower_cap in {
                    'wife', 'husband', 'girlfriend', 'boyfriend', 'partner',
                    'ex', 'sister', 'brother', 'mom', 'dad', 'mother', 'father',
                    'friend', 'boss', 'coworker', 'therapist', 'son', 'daughter',
                    'uncle', 'aunt', 'cousin', 'grandma', 'grandmother',
                    'grandpa', 'grandfather', 'fiancée', 'fiancee',
                    'roommate', 'neighbor', 'colleague',
                }:
                    # Store as relationship reference, not entity name
                    # We'll catch the actual name in pass 3 if present
                    continue

                found[name_lower_cap] = {
                    'name': captured,
                    'type': entity_type,
                    'confidence': 0.8,
                    'context': context,
                }

        # Pass 3: Proper noun heuristic
        # Find capitalized words that aren't at sentence start
        sentence_starts = set()
        for m in _SENTENCE_START.finditer(text):
            sentence_starts.add(m.start(1))

        # Also mark position 0 as sentence start
        if text and text[0].isupper():
            sentence_starts.add(0)

        word_pattern = re.compile(r'\b([A-Z][a-záàâãéèêíïóôõúç]{2,})\b')
        for match in word_pattern.finditer(text):
            word = match.group(1)
            word_lower = word.lower()

            # Skip if at sentence start
            if match.start() in sentence_starts:
                continue

            # Skip stop words and already found
            if word_lower in _STOP_WORDS or word_lower in found:
                continue

            # Skip very short words
            if len(word) < 3:
                continue

            start = max(0, match.start() - 100)
            end = min(len(text), match.end() + 100)
            context = text[start:end].strip()

            found[word_lower] = {
                'name': word,
                'type': 'person',  # Default assumption for proper nouns
                'confidence': 0.5,
                'context': context,
            }

        return list(found.values())

    def record_mention(
        self,
        user_id: str,
        entity_name: str,
        entity_type: str,
        crystal_id: int = None,
        context: str = None,
        sentiment: str = None,
        wound_active: str = None,
        emotion_active: str = None,
        display_name: str = None,
        metadata: dict = None,
    ):
        """Upsert entity_mentions and insert entity_mention_log row."""
        now = datetime.now().isoformat()
        name_lower = entity_name.lower().strip()
        display = display_name or entity_name

        conn = sqlite3.connect(self.db_path)
        try:
            # Check existing
            row = conn.execute(
                "SELECT id, mention_count FROM entity_mentions WHERE user_id = ? AND entity_name = ?",
                (user_id, name_lower),
            ).fetchone()

            if row:
                # Update existing
                updates = {
                    'last_seen': now,
                    'updated_at': now,
                    'mention_count': row[1] + 1,
                }
                if crystal_id is not None:
                    updates['crystal_id'] = crystal_id
                if context is not None:
                    updates['mention_context'] = context[:200]
                if sentiment is not None:
                    updates['sentiment'] = sentiment
                if metadata is not None:
                    # Merge with existing metadata
                    existing_meta = {}
                    try:
                        existing_row = conn.execute(
                            "SELECT metadata FROM entity_mentions WHERE id = ?",
                            (row[0],),
                        ).fetchone()
                        if existing_row and existing_row[0]:
                            existing_meta = json.loads(existing_row[0])
                    except (json.JSONDecodeError, TypeError):
                        pass
                    existing_meta.update(metadata)
                    updates['metadata'] = json.dumps(existing_meta)

                set_clause = ", ".join(f"{k} = ?" for k in updates)
                values = list(updates.values()) + [row[0]]
                conn.execute(
                    f"UPDATE entity_mentions SET {set_clause} WHERE id = ?",
                    values,
                )
            else:
                # Insert new
                meta_json = json.dumps(metadata) if metadata else None
                conn.execute(
                    """INSERT INTO entity_mentions
                       (user_id, entity_name, entity_type, display_name, crystal_id,
                        mention_context, sentiment, first_seen, last_seen, mention_count,
                        is_active, metadata, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?, ?)""",
                    (user_id, name_lower, entity_type, display, crystal_id,
                     context[:200] if context else None, sentiment,
                     now, now, meta_json, now, now),
                )

            # Always log the mention
            conn.execute(
                """INSERT INTO entity_mention_log
                   (user_id, entity_name, crystal_id, mention_context,
                    sentiment, wound_active, emotion_active, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, name_lower, crystal_id,
                 context[:200] if context else None,
                 sentiment, wound_active, emotion_active, now),
            )

            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_stale_entities(self, user_id: str, days: int = 14, limit: int = 10) -> List[Dict]:
        """
        Entities not mentioned in N days, sorted by mention_count (most important first).
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT entity_name, entity_type, display_name, last_seen,
                      mention_count, sentiment, metadata
               FROM entity_mentions
               WHERE user_id = ? AND is_active = 1 AND last_seen < ?
               ORDER BY mention_count DESC
               LIMIT ?""",
            (user_id, cutoff, limit),
        ).fetchall()
        conn.close()

        result = []
        now = datetime.now()
        for r in rows:
            last = datetime.fromisoformat(r['last_seen']) if r['last_seen'] else now
            days_stale = (now - last).total_seconds() / 86400
            result.append({
                'name': r['display_name'] or r['entity_name'],
                'entity_name': r['entity_name'],
                'type': r['entity_type'],
                'last_seen': r['last_seen'],
                'days_stale': round(days_stale, 1),
                'mention_count': r['mention_count'],
                'sentiment': r['sentiment'],
                'metadata': json.loads(r['metadata']) if r['metadata'] else None,
            })
        return result

    def get_entity_timeline(self, user_id: str, entity_name: str, limit: int = 20) -> List[Dict]:
        """Get entity_mention_log rows for a specific entity."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM entity_mention_log
               WHERE user_id = ? AND entity_name = ?
               ORDER BY timestamp DESC
               LIMIT ?""",
            (user_id, entity_name.lower().strip(), limit),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def bootstrap_from_profile(self, user_id: str, profile_text: str):
        """
        Parse WILTON_PROFILE-style text for known entities.
        Looks for ### NAME and key relationship markers.
        """
        # Find all ### headers and their positions
        header_pattern = re.compile(r'###\s+([A-Z][A-Z]+(?:\s+\([^)]+\))?)')
        matches = list(header_pattern.finditer(profile_text))

        for i, match in enumerate(matches):
            raw = match.group(1).strip()
            # Strip parenthetical
            name = re.sub(r'\s*\([^)]*\)', '', raw).strip()
            display_name = name.title()
            name_lower = name.lower()

            # Limit block to text between this header and the next header (or ## header)
            start = match.end()
            if i + 1 < len(matches):
                end = matches[i + 1].start()
            else:
                # Look for next ## header or end of text
                next_section = re.search(r'\n##\s', profile_text[start:])
                end = start + next_section.start() if next_section else min(len(profile_text), start + 500)
            block = profile_text[start:end]

            metadata = {}
            if 'RUPTURE' in raw or 'BETRAYAL' in block.upper():
                metadata['rupture_vector'] = True

            # Relationship detection: only check the first ~150 chars (the description line)
            first_lines = block[:150].lower()
            if 'girlfriend' in first_lines or 'namorada' in first_lines:
                metadata['relationship'] = 'girlfriend'
            elif 'boyfriend' in first_lines:
                metadata['relationship'] = 'boyfriend'
            elif 'wife' in first_lines or 'husband' in first_lines:
                metadata['relationship'] = 'spouse'
            elif re.search(r'\bex\b', first_lines) or 'ex-partner' in first_lines:
                metadata['relationship'] = 'ex-partner'
            elif re.search(r'\bbest friend\b', first_lines) or re.search(r'\bbrother\b', first_lines):
                metadata['relationship'] = 'brother-friend'
            elif re.search(r'\bfriend\b', first_lines):
                metadata['relationship'] = 'friend'
            elif re.search(r'\b(?:mom|mother)\b', first_lines):
                metadata['relationship'] = 'mother'
            elif re.search(r'\b(?:dad|father)\b', first_lines):
                metadata['relationship'] = 'father'

            # Determine sentiment from block (scoped to this section only)
            sentiment = 'neutral'
            if 'BETRAYAL' in block.upper() or 'wound' in block.lower():
                sentiment = 'negative'
            elif 'love' in block.lower() or re.search(r'\btrust\w*\b', block.lower()):
                sentiment = 'positive'
            elif 'tension' in block.lower() or 'paradox' in block.lower():
                sentiment = 'complex'

            self.record_mention(
                user_id=user_id,
                entity_name=name_lower,
                entity_type='person',
                display_name=display_name,
                sentiment=sentiment,
                metadata=metadata if metadata else None,
            )

    def bootstrap_from_crystals(self, user_id: str, batch_size: int = 500):
        """
        One-time batch scan of existing crystals for entity extraction.

        Uses only pre-seeded known entities (from profile bootstrap) to avoid
        snowball noise. Run --profile first to seed entities, then --bootstrap.
        """
        # Snapshot known entities at start — only these will be tracked
        known = self._get_known_entities(user_id)
        if not known:
            print("No known entities seeded. Run --profile first to seed from user profile.")
            return

        print(f"Bootstrapping from {len(known)} seeded entities: {', '.join(known.keys())}")

        # Build regex patterns for each known entity
        entity_patterns = {}
        for name_lower, info in known.items():
            try:
                pattern = re.compile(r'\b' + re.escape(name_lower) + r'\b', re.IGNORECASE)
                entity_patterns[name_lower] = (pattern, info)
            except re.error:
                continue

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row

        # Scan both crystal tables (crystals = imported, auto_insights = generated)
        crystal_tables = []
        for table, content_col in [('crystals', 'content'), ('auto_insights', 'content')]:
            try:
                cnt = conn.execute(f"SELECT COUNT(*) FROM [{table}]").fetchone()[0]
                if cnt > 0:
                    crystal_tables.append((table, content_col, cnt))
            except Exception:
                pass

        if not crystal_tables:
            print("No crystal tables found.")
            conn.close()
            return

        total_all = sum(cnt for _, _, cnt in crystal_tables)
        print(f"Scanning {total_all} crystals across {len(crystal_tables)} tables...")

        extracted_count = 0
        for table, content_col, table_total in crystal_tables:
            print(f"  Table: {table} ({table_total} rows)")
            offset = 0
            while offset < table_total:
                rows = conn.execute(
                    f"""SELECT id, {content_col}
                       FROM [{table}]
                       ORDER BY id
                       LIMIT ? OFFSET ?""",
                    (batch_size, offset),
                ).fetchall()

                if not rows:
                    break

                for row in rows:
                    content = row[content_col] or ''
                    if len(content) < 10:
                        continue

                    for name_lower, (pattern, info) in entity_patterns.items():
                        match = pattern.search(content)
                        if match:
                            start = max(0, match.start() - 100)
                            end = min(len(content), match.end() + 100)
                            context = content[start:end].strip()

                            self.record_mention(
                                user_id=user_id,
                                entity_name=name_lower,
                                entity_type=info['entity_type'],
                                crystal_id=row['id'],
                                context=context,
                            )
                            extracted_count += 1

                offset += batch_size
                processed = min(offset, table_total)
                print(f"    {processed}/{table_total} ({extracted_count} mentions total)")

        conn.close()
        print(f"Bootstrap complete: {extracted_count} entity mentions across {len(known)} entities.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entity Index tools")
    parser.add_argument("--bootstrap", action="store_true", help="Bootstrap from existing crystals")
    parser.add_argument("--user", default="wilton", help="User ID")
    parser.add_argument("--profile", action="store_true", help="Bootstrap from user profile")
    parser.add_argument("--stale", action="store_true", help="Show stale entities")
    parser.add_argument("--days", type=int, default=14, help="Days for staleness check")
    args = parser.parse_args()

    ei = EntityIndex()

    if args.profile:
        from identity import get_profile
        profile = get_profile(args.user)
        ei.bootstrap_from_profile(args.user, profile)
        print("Profile bootstrap complete.")
    elif args.bootstrap:
        ei.bootstrap_from_crystals(args.user)
    elif args.stale:
        stale = ei.get_stale_entities(args.user, days=args.days)
        if stale:
            for e in stale:
                print(f"  {e['name']} ({e['type']}) — {e['days_stale']:.0f} days stale, {e['mention_count']} mentions")
        else:
            print("  No stale entities found.")
    else:
        parser.print_help()
