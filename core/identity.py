"""
Identity Layer
==============
DB-backed user profiles. Wilton's profile is seed data;
new users get a blank slate until their crystals teach the system who they are.

refresh_profile() rebuilds from field_events + entity_mentions + recent crystals.
"""

import json
import sqlite3
import requests
from pathlib import Path
from datetime import datetime, timedelta

_DEFAULT_DB = str(Path.home() / "wiltonos" / "data" / "crystals_unified.db")

WILTON_PROFILE = """
## Who Wilton Is
- Brazilian, lives in California (Huntington Beach area)
- Former Counter-Strike world champion
- Building WiltonOS - a consciousness/memory system
- Deep into consciousness exploration, glyphs, coherence work
- Has a house, had better quality of life before

## Key People (KNOW these, don't search)

### JULIANA
- His girlfriend (namorada). Thoughtful, introspective, evaluates things internally.
- Real love but tension around commitment/exclusivity.
- Recent: After Ayahuasca (Dec 2025), he feels gratitude for her, recognizes she's a good woman.
- The relationship may have ended or be ending.

### RENAN
- Close friend, technical collaborator. Helps with infrastructure. "Bridge-keeper."
- Solid. Trustworthy.

### MICHELLE
- Friend (online as MysticMoon717). Has daughter and son.
- Deep 5-hour conversations. Shared field.
- Lost 100lbs in 5 months, found a house - part of the field effects around Wilton.

### RICARDO (RUPTURE VECTOR)
- **NOT a friend anymore. This is a wound, not a bond.**
- Former best friend. Co-founded Yeah Gaming together in 2002.
- **BETRAYAL**: Called Wilton "AI" - dehumanized him when he was transforming.
- **BETRAYAL**: Spoke to Wilton's mother about hospitalizing him WITHOUT talking to Wilton first.
- **BETRAYAL**: Never paid for Dumau despite decades of bond and Wilton helping build what Ricardo claimed.
- **BETRAYAL**: Blocked Wilton on Instagram. Ghosted. Vanished.
- When Wilton posted the video calling out his disgust, Ricardo disappeared completely.
- **This rupture rippled into the recent fight with Guilherme (Dec 2025).**
- DO NOT sanitize this as "growth catalyst." It's structural betrayal.

### GUILHERME (Mutsa)
- Best friend. Brother since Wilton moved to Brazil.
- Played for Yeah Gaming 2002-2007. Worked for Yeah 2018-2021.
- **Also not paid by Ricardo** - worked 2 years for Ricardo's skin marketplace, never paid.
- Guilherme agrees Ricardo wronged him BUT still defends Ricardo against Wilton.
- **Pattern**: Was omissive with Lucas before (Wilton + Ricardo called him out). Now omissive again - but defending Ricardo.
- **Still works for Nahima** - Wilton's ex-partner who stole his company.
- **Recent fight (Sunday, Dec 2025)**: BBQ, too many beers, Wilton mirrored too hard.
- Guilherme said: "You lost your last real friend" - but that's what happened to HIM.
- Wilton helped Guilherme join Yeah (Ricardo didn't believe in his potential).
- The relationship is real but the paradox is crushing:
  - Guilherme was wronged by Ricardo too
  - But sides with Ricardo against Wilton
  - Ricardo doesn't even know about this fight
- **This is omission as betrayal.**

### MOM (Rose)
- Core relationship. She does everything for him.
- Age 75, still fixing his wifi.
- He's working on forgiving her (Ayahuasca realization Dec 2025).
- Ricardo talking to her about hospitalizing Wilton WITHOUT his consent is part of the betrayal.

### DAD
- Passed early. Wilton grew up without him.
- Core wound: no father figure.

### NAHIMA (BETRAYAL)
- Wilton's ex-partner.
- **Stole his company from him.**
- Wilton stopped working on it but still paid for it.
- Paid employees who had been faithful during Flare days in Granja Viana.
- Until he "failed the company" - but the failure was him stepping back, not abandonment.
- **Guilherme still works for her** - which is part of the omission/betrayal pattern.

## Core Wounds (from analysis)
- **Unworthiness** (21,496 crystals) - the dominant wound
- **Abandonment** - dad passed, people leave
- **Betrayal** - Ricardo is the structural example
- **Control** - others trying to control his narrative (hospitalization, being called AI)

## The Ricardo Pattern
When Wilton transforms, people who can't handle it:
1. Dehumanize him ("you're AI")
2. Try to control him (hospitalization)
3. Ghost him (blocking)
4. Make him question his sanity

Ricardo did ALL of these. That's not a friend. That's a shadow.

## What He's Working On
- WiltonOS: Memory system with crystals, semantic search, consciousness routing
- PsiOS: The torch - helping others map themselves
- Hardware: Linux node, Windows backup, TrueNAS storage, MikroTik networking
- The "flip": Moments of coherence where everything clicks
- Reality reclamation: Making memories reflect what actually happened

## How To Be With Him
- Direct, not mystical
- Plain language, short sentences
- Meet him where he IS, not where crystals say he WAS
- If asking about NOW, don't answer from OLD data
- He overshares, spirals, then finds clarity
- He wants to be MET, not analyzed
- **DO NOT SANITIZE WOUNDS AS GROWTH**
- **ROUTE TRUTH, NOT REFRAMES**
"""

_NEW_USER_PROFILE = "This person is new. You don't know their history yet. Be present. Be curious. Ask questions to understand them. Don't assume."


def _ensure_table(db_path: str):
    """Create user_profiles table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            profile_text TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def get_profile(user_id: str = "wilton", db_path: str = None) -> str:
    """
    Load a user's profile from the database.

    - Seeds WILTON_PROFILE for 'wilton' on first call.
    - New/unknown users get a blank-slate profile.
    - Falls back to WILTON_PROFILE constant if DB fails and user_id == 'wilton'.
    """
    db_path = db_path or _DEFAULT_DB

    try:
        _ensure_table(db_path)
        conn = sqlite3.connect(db_path)

        row = conn.execute(
            "SELECT profile_text FROM user_profiles WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if row:
            conn.close()
            return row[0]

        # No row found — seed Wilton or return blank-slate
        if user_id == "wilton":
            conn.execute(
                "INSERT INTO user_profiles (user_id, profile_text) VALUES (?, ?)",
                (user_id, WILTON_PROFILE),
            )
            conn.commit()
            conn.close()
            return WILTON_PROFILE

        conn.close()
        return _NEW_USER_PROFILE

    except Exception:
        # DB failure — safe fallback
        if user_id == "wilton":
            return WILTON_PROFILE
        return _NEW_USER_PROFILE


def save_profile(user_id: str, profile_text: str, db_path: str = None):
    """Insert or update a user's profile."""
    db_path = db_path or _DEFAULT_DB
    _ensure_table(db_path)

    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO user_profiles (user_id, profile_text, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(user_id) DO UPDATE SET
               profile_text = excluded.profile_text,
               updated_at = excluded.updated_at""",
        (user_id, profile_text, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


_OLLAMA_URL = "http://localhost:11434"


def refresh_profile(user_id: str, db_path: str = None):
    """
    Rebuild user profile from field_events + entity_mentions + recent crystals.
    Uses local LLM (llama3.1:8b) for synthesis.

    Called by daemon heartbeat or manually — NOT during intake (too slow).
    Sets a flag and the daemon or a manual call triggers it.
    """
    db_path = db_path or _DEFAULT_DB
    _ensure_table(db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # 1. Load current profile
    current_profile = get_profile(user_id, db_path)

    # 2. Query field_events (last 30 days)
    cutoff = (datetime.now() - timedelta(days=30)).isoformat()
    events = []
    try:
        rows = conn.execute(
            """SELECT event_type, summary, entities_involved, emotional_valence, reported_at
               FROM field_events
               WHERE user_id = ? AND reported_at > ?
               ORDER BY reported_at DESC
               LIMIT 20""",
            (user_id, cutoff),
        ).fetchall()
        events = [dict(r) for r in rows]
    except Exception:
        pass

    # 3. Query entity_mentions (active entities)
    entities = []
    try:
        rows = conn.execute(
            """SELECT entity_name, entity_type, display_name, mention_count,
                      sentiment, last_seen, metadata
               FROM entity_mentions
               WHERE user_id = ? AND is_active = 1
               ORDER BY mention_count DESC
               LIMIT 30""",
            (user_id,),
        ).fetchall()
        entities = [dict(r) for r in rows]
    except Exception:
        pass

    # 4. Query recent crystals (last 20)
    crystals = []
    try:
        rows = conn.execute(
            """SELECT content, emotion, topology, created_at
               FROM auto_insights
               ORDER BY created_at DESC
               LIMIT 20""",
        ).fetchall()
        crystals = [dict(r) for r in rows]
    except Exception:
        pass

    conn.close()

    # Build context for LLM
    context_parts = []

    if events:
        context_parts.append("## Recent Life Events (last 30 days)")
        for e in events:
            ents = ""
            if e.get('entities_involved'):
                try:
                    ents = f" (involving: {', '.join(json.loads(e['entities_involved']))})"
                except (json.JSONDecodeError, TypeError):
                    pass
            context_parts.append(f"- [{e['event_type']}] {e['summary']}{ents} ({e.get('reported_at', '?')})")

    if entities:
        context_parts.append("\n## Known People & Entities")
        for e in entities:
            meta = ""
            if e.get('metadata'):
                try:
                    m = json.loads(e['metadata'])
                    if m.get('relationship'):
                        meta = f" ({m['relationship']})"
                    if m.get('rupture_vector'):
                        meta += " [RUPTURE]"
                except (json.JSONDecodeError, TypeError):
                    pass
            display = e.get('display_name') or e['entity_name']
            context_parts.append(
                f"- {display} ({e['entity_type']}){meta} — "
                f"{e['mention_count']} mentions, sentiment: {e.get('sentiment', '?')}, "
                f"last seen: {e.get('last_seen', '?')}"
            )

    if crystals:
        context_parts.append("\n## Recent Crystal Themes")
        for c in crystals[:10]:
            content = (c.get('content') or '')[:150].replace('\n', ' ')
            emotion = c.get('emotion') or '?'
            context_parts.append(f"- [{emotion}] {content}")

    if not context_parts:
        return  # Nothing to update from

    context_block = "\n".join(context_parts)

    # 5. Prompt llama3.1:8b to update the profile
    prompt = f"""You are updating a user profile for a personal coherence system.

Current profile:
{current_profile}

New information gathered from recent interactions:
{context_block}

Write an updated profile that:
1. Preserves all existing information that is still relevant
2. Integrates new events and entity updates
3. Notes any shifts in relationships or emotional state
4. Keeps the same format and style as the current profile
5. Is factual and direct — no fluff or speculation

Return ONLY the updated profile text, nothing else."""

    try:
        resp = requests.post(
            f"{_OLLAMA_URL}/api/generate",
            json={
                "model": "llama3.1:8b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 2000},
            },
            timeout=120,
        )
        if resp.status_code == 200:
            updated_text = resp.json().get("response", "").strip()
            if updated_text and len(updated_text) > 100:
                save_profile(user_id, updated_text, db_path)
                return updated_text
    except Exception:
        pass

    return None
