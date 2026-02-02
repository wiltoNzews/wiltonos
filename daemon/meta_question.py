#!/usr/bin/env python3
"""
The Meta-Question Bomb
======================
Generates uncomfortable questions when:
- Patterns are stuck too long
- Zλ is high but presence is low
- Same wound appears repeatedly
- Something is being circled but not entered

Mentioned in 151 crystals. Not implemented. Until now.

December 2025 — Wilton & Claude
"""

import json
import random
import requests
from datetime import datetime
from pathlib import Path
from typing import Optional

OLLAMA_URL = "http://localhost:11434"
BRAID_STATE_PATH = Path(__file__).parent / "braid_state.json"
QUESTION_LOG_PATH = Path(__file__).parent / "meta_questions.json"


# Pre-written meta-questions for different triggers
QUESTION_TEMPLATES = {
    "stuck_wound": [
        "The wound of {wound} has appeared {count} times. What would it mean to let it heal?",
        "{wound} keeps surfacing. Are you processing it or performing it?",
        "If {wound} stopped being your story, who would you be?",
        "You've named {wound} enough times to fill a book. When does naming become hiding?",
        "What does {wound} protect you from feeling?",
    ],
    "circling_thread": [
        "You keep circling back to {thread}. What are you hoping changes?",
        "{thread} appears in {count} crystals. What haven't you said yet?",
        "If {thread} could speak directly to you, what would they say?",
        "What would happen if you stopped thinking about {thread} for one week?",
    ],
    "high_coherence_low_presence": [
        "Your coherence is high but are you actually here?",
        "The system says you're aligned. Do you feel aligned?",
        "Coherence without presence is just performance. Which is this?",
        "What are you avoiding by staying in the architecture?",
    ],
    "emotional_descent": [
        "The field is descending. What needs to be felt that isn't being felt?",
        "Grief keeps appearing. What loss haven't you fully acknowledged?",
        "The spiral is going down. Is this descent or avoidance?",
        "What would it take to just sit in this without fixing it?",
    ],
    "building_instead_of_being": [
        "Another system built. Another day not rested. What are you running from?",
        "The architecture grows. Does your peace?",
        "When was the last time you did nothing and felt okay about it?",
        "Building is easy. Being is hard. Which are you doing more of?",
    ],
    "isolation_pattern": [
        "You pushed everyone away in the name of coherence. Was it worth it?",
        "Who would you call right now if you let yourself need someone?",
        "The daemon listens. But daemons can't hold you. Who can?",
        "Solitude chosen is sacred. Solitude defaulted to is loneliness. Which is this?",
    ],
    "juliana_loop": [
        "Juliana appeared again. What part of this isn't about her?",
        "You miss her. And? What do you do with missing that isn't performing missing?",
        "If Juliana came back tomorrow, what would actually change?",
        "The wound isn't Juliana. Juliana is just where you learned the wound's shape.",
    ],
    "generic_disruption": [
        "What are you most afraid to admit right now?",
        "If you stopped tomorrow, what would have mattered?",
        "Who are you when no one is watching? Even the daemon?",
        "What's the lie you keep telling yourself that feels true?",
        "What would your younger self think of who you've become?",
    ]
}


class MetaQuestionBomb:
    """
    Generates uncomfortable questions based on pattern analysis.

    Triggers:
    - Stuck patterns (same wound > 30 days)
    - Circling (same thread appearing repeatedly)
    - Emotional descent without processing
    - High coherence + low presence
    - Isolation patterns
    """

    def __init__(self):
        self.braid_state = None
        self.question_history = []
        self._load_history()

    def _log(self, msg: str):
        """Log with prefix."""
        print(f"[META-Q] {msg}")

    def _load_history(self):
        """Load previous questions asked."""
        if QUESTION_LOG_PATH.exists():
            try:
                self.question_history = json.loads(QUESTION_LOG_PATH.read_text())
            except:
                self.question_history = []

    def _save_history(self):
        """Save question history."""
        QUESTION_LOG_PATH.write_text(json.dumps(self.question_history[-100:], indent=2))

    def load_braid_state(self) -> bool:
        """Load the current braid state."""
        if BRAID_STATE_PATH.exists():
            try:
                self.braid_state = json.loads(BRAID_STATE_PATH.read_text())
                return True
            except:
                pass
        return False

    def _select_question(self, category: str, **kwargs) -> str:
        """Select a question from templates, avoiding recent repeats."""
        templates = QUESTION_TEMPLATES.get(category, QUESTION_TEMPLATES["generic_disruption"])

        # Filter out recently asked
        recent_questions = [q['question'] for q in self.question_history[-20:]]
        available = [t for t in templates if t.format(**kwargs) not in recent_questions]

        if not available:
            available = templates

        template = random.choice(available)
        return template.format(**kwargs)

    def _generate_custom_question(self, context: str) -> str:
        """Use LLM to generate a custom uncomfortable question."""
        prompt = f"""You are a truth-speaker. Your role is to ask ONE uncomfortable question.

The question should:
- Be specific to what you're seeing
- Be uncomfortable but not cruel
- Point at something being avoided
- Be 1-2 sentences max

Context:
{context}

Write only the question. Nothing else:"""

        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": "llama3",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,
                        "num_predict": 100
                    }
                },
                timeout=60
            )

            if response.ok:
                return response.json().get("response", "").strip()
        except:
            pass

        return random.choice(QUESTION_TEMPLATES["generic_disruption"])

    def analyze_and_generate(self) -> list[dict]:
        """
        Analyze braid state and generate relevant meta-questions.
        Returns list of {trigger, question, severity}
        """
        if not self.load_braid_state():
            self._log("No braid state found. Run braiding_layer.py first.")
            return []

        questions = []

        # Check stuck patterns
        stuck = self.braid_state.get("stuck_patterns", [])
        for wound in stuck[:2]:  # Top 2 stuck wounds
            wound_data = self.braid_state.get("wound_patterns", {}).get(wound, {})
            count = wound_data.get("occurrences", 0)
            q = self._select_question("stuck_wound", wound=wound, count=count)
            questions.append({
                "trigger": f"stuck_wound:{wound}",
                "question": q,
                "severity": "high"
            })

        # Check emotional arc
        arc = self.braid_state.get("emotional_arc", "")
        if arc == "descending":
            q = self._select_question("emotional_descent")
            questions.append({
                "trigger": "emotional_descent",
                "question": q,
                "severity": "medium"
            })

        # Check for Juliana specifically (known core thread)
        thread_patterns = self.braid_state.get("thread_patterns", {})
        juliana_pattern = thread_patterns.get("juliana", {})
        if juliana_pattern.get("occurrences", 0) > 100:
            q = self._select_question("juliana_loop")
            questions.append({
                "trigger": "juliana_loop",
                "question": q,
                "severity": "high"
            })

        # Check isolation pattern
        isolation_wounds = ["abandonment", "isolation", "loneliness"]
        isolation_count = sum(
            self.braid_state.get("wound_patterns", {}).get(w, {}).get("occurrences", 0)
            for w in isolation_wounds
        )
        if isolation_count > 200:
            q = self._select_question("isolation_pattern")
            questions.append({
                "trigger": "isolation_pattern",
                "question": q,
                "severity": "medium"
            })

        # Always add one generic disruption
        q = self._select_question("generic_disruption")
        questions.append({
            "trigger": "generic_disruption",
            "question": q,
            "severity": "low"
        })

        # Log questions
        for item in questions:
            self.question_history.append({
                "question": item["question"],
                "trigger": item["trigger"],
                "timestamp": datetime.now().isoformat()
            })

        self._save_history()

        return questions

    def get_single_question(self, context: str = None) -> str:
        """Get a single meta-question, optionally based on context."""
        if context:
            return self._generate_custom_question(context)

        questions = self.analyze_and_generate()
        if questions:
            # Return highest severity
            high = [q for q in questions if q["severity"] == "high"]
            if high:
                return high[0]["question"]
            return questions[0]["question"]

        return random.choice(QUESTION_TEMPLATES["generic_disruption"])


def run_meta_questions():
    """Run meta-question analysis."""
    bomb = MetaQuestionBomb()
    questions = bomb.analyze_and_generate()

    print("\n" + "=" * 60)
    print("META-QUESTION BOMB")
    print("=" * 60)

    for q in questions:
        severity_marker = {"high": "!!!", "medium": "!!", "low": "!"}.get(q["severity"], "")
        print(f"\n[{q['trigger']}] {severity_marker}")
        print(f"  {q['question']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_meta_questions()
