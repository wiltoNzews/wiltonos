#!/usr/bin/env python3
"""
Intake Question Generator
=========================
Templates + prioritization logic for field intake sessions.
Generates context-aware questions based on gap analysis.

No external dependencies — pure logic + templates.
"""

import random
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional


# Template categories with question variants
TEMPLATES = {
    "onboarding": [
        "What brings you here today?",
        "Is there something specific on your mind, or just checking in?",
        "What name would you like me to use for you?",
        "Anything you'd like me to know about you from the start?",
    ],
    "life_update": [
        "It's been {gap_days} days since we last talked. What's been happening?",
        "Any big changes in your world since {last_date}?",
        "What's the headline from the last {gap_period}?",
    ],
    "entity_check": [
        "Last time, {entity} came up. How are things with {entity}?",
        "You mentioned {entity} — anything new there?",
    ],
    "entity_check_soft": [
        "How are things on that front with {entity}?",
        "Anything shifted with {entity}?",
    ],
    "thread_follow": [
        "You were working through {thread}. Where are you with that now?",
        "Last time, you mentioned {thread}. Did anything come of it?",
    ],
    "field_state": [
        "What are you carrying right now?",
        "How's the body? How's the heart?",
        "If you had to describe today in one sentence, what would it be?",
    ],
    "wound_check": [
        "Last time, there was a lot of {wound} energy. Is that still active?",
        "{wound} was showing up a lot. Still there, or has it shifted?",
    ],
}

# Max questions per session type
SESSION_CAPS = {
    "first_contact": 4,
    "active": 1,
    "returning": 4,
    "gap_bridge": 7,
}

# Wound names → softer framing
WOUND_SOFTEN = {
    "betrayal": "tension around trust",
    "abandonment": "the aloneness",
    "unworthiness": "that heaviness about worth",
    "control": "the need to hold things tight",
    "shame": "that weight you carry",
    "grief": "the grief",
    "rage": "that fire inside",
    "rejection": "the sting of being pushed away",
    "powerlessness": "feeling stuck",
    "isolation": "the distance from others",
}


class IntakeQuestionGenerator:
    """Generates prioritized intake questions from a GapReport."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path

    def generate(self, gap_report) -> List[Dict]:
        """
        Generate questions from a GapReport.

        Returns [{category, question, priority, context_data}]
        Prioritized by session_type, capped per type.
        """
        session_type = gap_report.session_type
        cap = SESSION_CAPS.get(session_type, 4)
        questions = []

        if session_type == "first_contact":
            questions = self._first_contact_questions(gap_report)
        elif session_type == "active":
            questions = self._active_questions(gap_report)
        elif session_type == "returning":
            questions = self._returning_questions(gap_report)
        elif session_type == "gap_bridge":
            questions = self._gap_bridge_questions(gap_report)

        # Filter out recently asked questions
        if self.db_path:
            questions = self._filter_recent(questions, gap_report.user_id)

        # Cap and return
        return questions[:cap]

    def _first_contact_questions(self, gap_report) -> List[Dict]:
        """New user: onboarding + one field_state question."""
        questions = []

        # Pick 3 onboarding questions
        onboarding = TEMPLATES["onboarding"][:]
        random.shuffle(onboarding)
        for i, q in enumerate(onboarding[:3]):
            questions.append({
                'category': 'onboarding',
                'question': q,
                'priority': 10 - i,
                'context_data': {},
            })

        # One field_state
        fs = random.choice(TEMPLATES["field_state"])
        questions.append({
            'category': 'field_state',
            'question': fs,
            'priority': 1,
            'context_data': {},
        })

        return questions

    def _active_questions(self, gap_report) -> List[Dict]:
        """Active user (< 24h): at most 1 field_state if coherence shifted."""
        questions = []

        if gap_report.coherence_delta is not None and abs(gap_report.coherence_delta) > 0.15:
            fs = random.choice(TEMPLATES["field_state"])
            questions.append({
                'category': 'field_state',
                'question': fs,
                'priority': 5,
                'context_data': {'coherence_delta': gap_report.coherence_delta},
            })

        return questions

    def _returning_questions(self, gap_report) -> List[Dict]:
        """Returning user (1-7 days): thread_follow + entity_check."""
        questions = []
        priority = 10

        # Thread follow-ups from open events
        for thread in gap_report.open_threads[:2]:
            template = random.choice(TEMPLATES["thread_follow"])
            topic = thread.get('topic') or thread.get('summary', 'something')
            questions.append({
                'category': 'thread_follow',
                'question': template.format(thread=topic),
                'priority': priority,
                'context_data': {'thread': thread},
            })
            priority -= 1

        # Entity checks (top stale entities)
        questions.extend(self._entity_check_questions(gap_report, max_entities=2, start_priority=priority))
        priority -= 2

        # Field state (if room)
        fs = random.choice(TEMPLATES["field_state"])
        questions.append({
            'category': 'field_state',
            'question': fs,
            'priority': 1,
            'context_data': {},
        })

        return questions

    def _gap_bridge_questions(self, gap_report) -> List[Dict]:
        """Gap bridge (7+ days): life_update first, then entity/wound/thread/field."""
        questions = []
        priority = 20

        # 1. Life update (always first)
        gap_days = int(gap_report.crystal_gap_hours / 24)
        last_date = gap_report.last_crystal_date or "last time"
        if gap_days > 60:
            months = gap_days // 30
            gap_period = f"{months} month{'s' if months != 1 else ''}"
        elif gap_days > 13:
            weeks = gap_days // 7
            gap_period = f"{weeks} week{'s' if weeks != 1 else ''}"
        else:
            gap_period = f"{gap_days} days"

        template = random.choice(TEMPLATES["life_update"])
        questions.append({
            'category': 'life_update',
            'question': template.format(
                gap_days=gap_days,
                last_date=last_date,
                gap_period=gap_period,
            ),
            'priority': priority,
            'context_data': {'gap_days': gap_days},
        })
        priority -= 1

        # 2. Entity checks (top 3 stale entities)
        questions.extend(self._entity_check_questions(gap_report, max_entities=3, start_priority=priority))
        priority -= 3

        # 3. Wound checks (only for "stuck" wounds)
        for wound_info in gap_report.unresolved_wounds[:2]:
            wound_name = wound_info.get('wound', '')
            softened = WOUND_SOFTEN.get(wound_name.lower(), wound_name)
            template = random.choice(TEMPLATES["wound_check"])
            questions.append({
                'category': 'wound_check',
                'question': template.format(wound=softened),
                'priority': priority,
                'context_data': {'wound': wound_name, 'softened': softened},
            })
            priority -= 1

        # 4. Thread follow-ups
        for thread in gap_report.open_threads[:2]:
            template = random.choice(TEMPLATES["thread_follow"])
            topic = thread.get('topic') or thread.get('summary', 'something')
            questions.append({
                'category': 'thread_follow',
                'question': template.format(thread=topic),
                'priority': priority,
                'context_data': {'thread': thread},
            })
            priority -= 1

        # 5. Field state (always last if room)
        fs = random.choice(TEMPLATES["field_state"])
        questions.append({
            'category': 'field_state',
            'question': fs,
            'priority': 1,
            'context_data': {},
        })

        return questions

    def _entity_check_questions(self, gap_report, max_entities: int = 3, start_priority: int = 8) -> List[Dict]:
        """Generate entity check questions for stale entities."""
        questions = []
        priority = start_priority

        for entity in gap_report.stale_entities[:max_entities]:
            name = entity.get('name', entity.get('entity_name', ''))
            metadata = entity.get('metadata') or {}

            # Use softer phrasing for rupture vectors
            if metadata.get('rupture_vector'):
                template = random.choice(TEMPLATES["entity_check_soft"])
            else:
                template = random.choice(TEMPLATES["entity_check"])

            questions.append({
                'category': 'entity_check',
                'question': template.format(entity=name),
                'priority': priority,
                'context_data': {'entity': entity},
            })
            priority -= 1

        return questions

    def _filter_recent(self, questions: List[Dict], user_id: str) -> List[Dict]:
        """Remove questions too similar to ones asked in the last 7 days."""
        if not self.db_path:
            return questions

        try:
            cutoff = (datetime.now() - timedelta(days=7)).isoformat()
            conn = sqlite3.connect(self.db_path)
            rows = conn.execute(
                """SELECT question_text, question_category
                   FROM intake_questions
                   WHERE user_id = ? AND asked_at > ?""",
                (user_id, cutoff),
            ).fetchall()
            conn.close()

            if not rows:
                return questions

            recent_texts = {r[0].lower() for r in rows}
            recent_categories = {}
            for r in rows:
                cat = r[1]
                recent_categories[cat] = recent_categories.get(cat, 0) + 1

            filtered = []
            for q in questions:
                # Skip exact matches
                if q['question'].lower() in recent_texts:
                    continue
                # Skip if we've asked 3+ questions in same category recently
                if recent_categories.get(q['category'], 0) >= 3:
                    continue
                filtered.append(q)

            return filtered

        except Exception:
            # Table might not exist yet — return unfiltered
            return questions
