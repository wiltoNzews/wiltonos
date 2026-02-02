#!/usr/bin/env python3
"""
Agent Mesh — Blackboard Architecture (9 agents)
================================================
Agents that observe each other, revise themselves, and compose one response
through dialogue — not assembly.

Blackboard pattern: a shared Board where agents post observations across rounds.
Each agent reads what others wrote. The response emerges from what stabilizes.

Round 0: Pattern (instant) + Bridge (instant) + Torus (instant)
         + Witness (LLM, sees Pattern/Bridge/Torus) + Grey (LLM, sees all)
Round 1: Ground (instant, can halt) + Chaos (LLM, skipped if halted)
Round 2: Seed (LLM, offers what's adjacent) + Weaver (LLM, reads everything)

Total LLM calls: 3-5 (Witness + Grey + optionally Chaos + optionally Seed + Weaver).
Pattern, Bridge, Torus, and Ground are pure functions.

Usage:
    from mesh import AgentMesh
    mesh = AgentMesh(call_model=some_callable, user_id="wilton")
    board = mesh.run(query, crystals, state, protocol_state, params,
                     pattern_match, context_str, system_prompt,
                     conversation_history)
    response = board.final_response()
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any


# ── Post types ────────────────────────────────────────────────────────

class PostKind(Enum):
    PATTERN = "pattern"
    BRIDGE = "bridge"
    TORUS = "torus"
    OBSERVATION = "observation"
    SHADOW = "shadow"
    CHAOS = "chaos"
    INTERRUPT = "interrupt"
    REVISION = "revision"
    RESPONSE = "response"
    SEED = "seed"
    SILENT = "silent"


@dataclass
class Post:
    """A single entry on the blackboard."""
    agent: str
    kind: PostKind
    content: str
    round: int
    timestamp: float = field(default_factory=time.time)
    in_reply_to: Optional[int] = None  # index into Board.posts


@dataclass
class BoardSeed:
    """Everything the board needs to start — the query context."""
    query: str
    crystals: List[Dict]
    state: Any  # CoherenceState
    params: Dict
    pattern_match: Any  # PatternMatch or None
    context_str: str
    system_prompt: str
    conversation_history: List[Dict]
    user_id: str
    protocol_state: Optional[Dict] = None


# ── Board ─────────────────────────────────────────────────────────────

class Board:
    """Shared blackboard. Agents post here. Response emerges from what stabilizes."""

    def __init__(self, seed: BoardSeed):
        self.seed = seed
        self.posts: List[Post] = []
        self.current_round: int = 0
        self.halted: bool = False
        self.halt_reason: Optional[str] = None

    def post(self, agent: str, kind: PostKind, content: str,
             in_reply_to: Optional[int] = None) -> int:
        """Add a post to the board. Returns the post index."""
        p = Post(
            agent=agent,
            kind=kind,
            content=content,
            round=self.current_round,
            in_reply_to=in_reply_to,
        )
        self.posts.append(p)
        return len(self.posts) - 1

    def halt(self, reason: str):
        """Ground's interrupt — stops the board."""
        self.halted = True
        self.halt_reason = reason

    def get_posts_by(self, agent: str) -> List[Post]:
        """Get an agent's own previous posts (for self-revision)."""
        return [p for p in self.posts if p.agent == agent]

    def get_round_posts(self, round_num: int) -> List[Post]:
        """Get all posts from a specific round."""
        return [p for p in self.posts if p.round == round_num]

    def get_all_content(self) -> str:
        """Render the entire board as text — for Weaver to read."""
        lines = []
        for i, p in enumerate(self.posts):
            if p.kind == PostKind.SILENT:
                continue
            reply = f" (replying to #{p.in_reply_to})" if p.in_reply_to is not None else ""
            lines.append(f"[{p.agent}/{p.kind.value}]{reply}: {p.content}")
        return "\n".join(lines)

    def final_response(self) -> Optional[str]:
        """Get the Weaver's response, or None if absent."""
        for p in reversed(self.posts):
            if p.kind == PostKind.RESPONSE:
                return p.content
        return None

    def active_post_count(self) -> int:
        """Count non-silent posts."""
        return sum(1 for p in self.posts if p.kind != PostKind.SILENT)

    def to_debug_dict(self) -> Dict:
        """Inspection/debugging — safe for JSON-like output."""
        return {
            "posts": [
                {
                    "agent": p.agent,
                    "kind": p.kind.value,
                    "content": p.content[:200],
                    "round": p.round,
                    "in_reply_to": p.in_reply_to,
                }
                for p in self.posts
                if p.kind != PostKind.SILENT
            ],
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "rounds": self.current_round,
            "total_posts": len(self.posts),
            "active_posts": self.active_post_count(),
        }


# ── Agent base ────────────────────────────────────────────────────────

class MeshAgent:
    """Base class for blackboard agents."""

    name: str = "base"

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        """
        Read the board, produce a post (or None to stay silent).
        Subclasses override this.
        """
        raise NotImplementedError


# ── Pattern Agent (pure function, instant) ────────────────────────────

class PatternAgent(MeshAgent):
    """Wraps PatternMatch result into a structured board post."""

    name = "pattern"

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        pm = board.seed.pattern_match
        if pm is None or pm.confidence < 0.2:
            return None

        parts = []
        if pm.wounds:
            wound_str = ", ".join(f"{w} ({c:.0%})" for w, c in pm.wounds[:3])
            parts.append(f"Wounds: {wound_str}")
        if pm.emotions:
            emo_str = ", ".join(e for e, _ in pm.emotions[:4])
            parts.append(f"Emotions: {emo_str}")
        if pm.hard_truth:
            parts.append(f"Hard truth: {pm.hard_truth}")
        if pm.masking:
            parts.append(f"Masking: {', '.join(pm.masking[:3])}")
        if pm.co_occurrence_insights:
            parts.append(f"Co-occurrence: {pm.co_occurrence_insights[0][:150]}")

        content = " | ".join(parts) if parts else "No significant patterns."

        idx = board.post(self.name, PostKind.PATTERN, content)
        return board.posts[idx]


# ── Bridge Agent (pure function, instant) ─────────────────────────────

class BridgeAgent(MeshAgent):
    """
    Finds structural connections across retrieved crystals that semantic
    similarity missed: wound co-occurrences, shared emotions across time
    periods, and temporal bridges between distant crystals.
    Silent if fewer than 3 crystals or no connections found.
    """

    name = "bridge"

    # Fallback keywords (used when FieldVocabulary is not available)
    WOUND_KEYWORDS = {
        "abandonment": ["abandon", "left me", "walked away", "disappeared", "ghosted"],
        "provider": ["provider", "provide", "money", "support", "breadwinner", "financial"],
        "worthlessness": ["worthless", "not enough", "inadequate", "failure", "broken"],
        "control": ["control", "powerless", "helpless", "trapped", "can't escape"],
        "betrayal": ["betray", "lied", "cheated", "trust", "backstab"],
        "rejection": ["reject", "unwanted", "excluded", "invisible", "ignored"],
        "shame": ["shame", "ashamed", "disgusting", "wrong", "hide"],
        "loss": ["loss", "grief", "gone", "miss", "mourning", "death"],
    }

    EMOTION_KEYWORDS = [
        "anger", "fear", "sadness", "joy", "love", "grief", "anxiety",
        "peace", "rage", "longing", "hope", "despair", "tenderness",
        "guilt", "pride", "confusion", "relief", "loneliness",
    ]

    def __init__(self, vocab: Optional[Any] = None):
        self.vocab = vocab

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        crystals = board.seed.crystals
        if len(crystals) < 3:
            return None

        connections = []
        use_vocab = self.vocab and getattr(self.vocab, "available", False)

        # Tag each crystal with detected wounds and emotions
        tagged = []
        for c in crystals:
            text = c.get("content", "").lower()
            cid = c.get("id", "?")
            if use_vocab:
                wounds = self.vocab.get_wound_names(text)
                emotions = self.vocab.get_emotion_names(text)
            else:
                wounds = [w for w, kws in self.WOUND_KEYWORDS.items()
                          if any(k in text for k in kws)]
                emotions = [e for e in self.EMOTION_KEYWORDS if e in text]
            tagged.append({"id": cid, "wounds": wounds, "emotions": emotions})

        # 1. Wound co-occurrence: crystals sharing the same wound
        from collections import defaultdict
        wound_map = defaultdict(list)
        for t in tagged:
            for w in t["wounds"]:
                wound_map[w].append(t["id"])
        for wound, ids in wound_map.items():
            if len(ids) >= 2:
                connections.append(f"wound '{wound}' links crystals {ids[:3]}")

        # 2. Emotion threads: same emotion across distant crystals
        emotion_map = defaultdict(list)
        for t in tagged:
            for e in t["emotions"]:
                emotion_map[e].append(t["id"])
        for emo, ids in emotion_map.items():
            if len(ids) >= 2:
                # Check if crystals are temporally distant (ID gap > 500)
                numeric_ids = [i for i in ids if isinstance(i, int)]
                if len(numeric_ids) >= 2:
                    spread = max(numeric_ids) - min(numeric_ids)
                    if spread > 500:
                        connections.append(
                            f"emotion '{emo}' spans {spread} crystals apart "
                            f"({min(numeric_ids)}→{max(numeric_ids)})"
                        )

        # 3. Temporal bridges: distant crystals sharing any wound/emotion
        numeric_tagged = [(t, t["id"]) for t in tagged if isinstance(t["id"], int)]
        numeric_tagged.sort(key=lambda x: x[1])
        if len(numeric_tagged) >= 2:
            earliest = numeric_tagged[0]
            latest = numeric_tagged[-1]
            gap = latest[1] - earliest[1]
            if gap > 1000:
                shared = set(earliest[0]["wounds"]) & set(latest[0]["wounds"])
                shared |= set(earliest[0]["emotions"]) & set(latest[0]["emotions"])
                if shared:
                    connections.append(
                        f"temporal bridge: #{earliest[1]}↔#{latest[1]} "
                        f"({gap} apart) share {', '.join(list(shared)[:3])}"
                    )

        if not connections:
            return None

        content = " | ".join(connections[:4])
        idx = board.post(self.name, PostKind.BRIDGE, content)
        return board.posts[idx]


# ── Torus Agent (pure function, instant) ──────────────────────────────

class TorusAgent(MeshAgent):
    """
    Detects recurring wounds/themes across conversation history and past sessions.
    Flags when the current query repeats what's been circling.
    Silent if no recurrence found.
    In-session: needs 4+ conversation turns.
    Cross-session: needs MeshMemory with past runs.
    """

    name = "torus"

    # Fallback keyword maps (used when FieldVocabulary is not available)
    WOUND_SCAN = {
        "abandonment": ["abandon", "left", "alone", "walked away", "nobody stays"],
        "provider": ["money", "provide", "financial", "earn", "support"],
        "worthlessness": ["worthless", "not enough", "can't do", "failure", "inadequate"],
        "control": ["control", "trapped", "helpless", "powerless", "can't escape"],
        "rejection": ["reject", "unwanted", "invisible", "excluded", "don't belong"],
        "shame": ["shame", "ashamed", "wrong", "hide", "disgusting"],
        "loss": ["lost", "gone", "grief", "miss", "mourning"],
        "betrayal": ["betray", "lied", "trust", "cheated", "deceived"],
    }

    EMOTION_SCAN = {
        "anger": ["angry", "rage", "furious", "pissed", "mad"],
        "fear": ["afraid", "scared", "terrified", "anxious", "panic"],
        "sadness": ["sad", "crying", "tears", "depressed", "heavy"],
        "longing": ["miss", "longing", "want", "wish", "yearning"],
        "confusion": ["confused", "lost", "don't know", "stuck", "spinning"],
    }

    RECURRENCE_THRESHOLD = 2  # same wound/emotion in N+ turns → loop

    def __init__(self, vocab: Optional[Any] = None, memory: Optional[Any] = None):
        self.vocab = vocab
        self.memory = memory

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        history = board.seed.conversation_history
        has_memory = self.memory is not None

        # Need either enough in-session history or cross-session memory
        if len(history) < 4 and not has_memory:
            return None

        # Extract user turns only
        user_turns = [h["content"].lower() for h in history if h.get("role") == "user"]
        current = board.seed.query.lower()
        use_vocab = self.vocab and getattr(self.vocab, "available", False)

        # ── In-session scanning ──────────────────────────────────
        from collections import Counter
        wound_hits = Counter()
        emotion_hits = Counter()

        if len(user_turns) >= 2:
            for turn in user_turns + [current]:
                if use_vocab:
                    for w, _ in self.vocab.scan_wounds(turn):
                        wound_hits[w] += 1
                    for e, _ in self.vocab.scan_emotions(turn):
                        emotion_hits[e] += 1
                else:
                    for wound, keywords in self.WOUND_SCAN.items():
                        if any(k in turn for k in keywords):
                            wound_hits[wound] += 1
                    for emotion, keywords in self.EMOTION_SCAN.items():
                        if any(k in turn for k in keywords):
                            emotion_hits[emotion] += 1

        # Find in-session recurring themes
        recurring_wounds = [w for w, c in wound_hits.items()
                           if c >= self.RECURRENCE_THRESHOLD]
        recurring_emotions = [e for e, c in emotion_hits.items()
                             if c >= self.RECURRENCE_THRESHOLD]

        # Detect wounds/emotions in current query
        if use_vocab:
            current_wound_set = set(self.vocab.get_wound_names(current))
            current_emotion_set = set(self.vocab.get_emotion_names(current))
        else:
            current_wound_set = set(
                w for w, kws in self.WOUND_SCAN.items()
                if any(k in current for k in kws)
            )
            current_emotion_set = set(
                e for e, kws in self.EMOTION_SCAN.items()
                if any(k in current for k in kws)
            )

        current_wounds = [w for w in recurring_wounds if w in current_wound_set]
        current_emotions = [e for e in recurring_emotions if e in current_emotion_set]

        # ── Cross-session scanning ───────────────────────────────
        cross_session_wounds = []
        if has_memory:
            try:
                recent_wounds = self.memory.query_recent_wounds(
                    board.seed.user_id, days=7, limit=20
                )
                for wound, past_count in recent_wounds.items():
                    if wound in current_wound_set and past_count >= 2:
                        cross_session_wounds.append((wound, past_count))
            except Exception:
                pass

        # ── Build report ─────────────────────────────────────────
        parts = []
        reported_wounds = set()

        for w in current_wounds:
            parts.append(f"wound '{w}' recurring ({wound_hits[w]}x across turns)")
            reported_wounds.add(w)
        for e in current_emotions:
            parts.append(f"emotion '{e}' recurring ({emotion_hits[e]}x across turns)")

        for wound, past_count in cross_session_wounds:
            if wound in reported_wounds:
                parts.append(f"wound '{wound}' also cross-session ({past_count}x in past 7d)")
            else:
                parts.append(f"wound '{wound}' cross-session ({past_count}x in past 7d)")

        if not parts:
            return None

        content = "Loop detected: " + " | ".join(parts)
        idx = board.post(self.name, PostKind.TORUS, content)
        return board.posts[idx]


# ── Witness Agent (LLM, slow) ────────────────────────────────────────

class WitnessAgent(MeshAgent):
    """Pure observation. Reads crystals + query + Pattern's output. 'I see...' No interpretation."""

    name = "witness"

    def __init__(self, call_model: Callable):
        self.call_model = call_model

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        # Build what Witness sees — Pattern + Bridge + Torus
        pattern_posts = [p for p in board.posts if p.agent == "pattern"]
        pattern_text = pattern_posts[0].content if pattern_posts else "No patterns detected."

        bridge_posts = [p for p in board.posts if p.agent == "bridge"]
        bridge_text = bridge_posts[0].content if bridge_posts else ""

        torus_posts = [p for p in board.posts if p.agent == "torus"]
        torus_text = torus_posts[0].content if torus_posts else ""

        # Crystal context (brief)
        crystal_snippets = []
        for c in board.seed.crystals[:5]:
            snippet = c.get("content", "")[:150].replace("\n", " ")
            crystal_snippets.append(snippet)
        crystals_text = "\n".join(f"  - {s}" for s in crystal_snippets) if crystal_snippets else "(no crystals)"

        system = (
            "You are the Witness — pure observation. You reflect what IS without adding or removing.\n"
            "No judgment. No advice. No interpretation beyond what's present.\n"
            "Begin with 'I see...' or 'I notice...'\n"
            "2-3 sentences max. Just reflect what is present in this moment."
        )

        structural = ""
        if bridge_text:
            structural += f"\nStructural connections: {bridge_text}"
        if torus_text:
            structural += f"\nRecurrence: {torus_text}"

        prompt = (
            f"The person said: \"{board.seed.query}\"\n\n"
            f"Pattern layer detected: {pattern_text}\n"
            f"{structural}\n\n"
            f"Memory fragments resonating:\n{crystals_text}\n\n"
            f"What do you see? Reflect only. No advice."
        )

        try:
            response = self.call_model(
                system, prompt,
                {"model": "local", "temperature": 0.3, "max_tokens": 150}
            )
            if response and "unavailable" not in response.lower():
                idx = board.post(self.name, PostKind.OBSERVATION, response)
                return board.posts[idx]
        except Exception:
            pass

        return None


# ── Grey Agent (LLM, slow) ───────────────────────────────────────────

class GreyAgent(MeshAgent):
    """Shadow reader. Reads what others posted. Names what's avoided. Returns SILENT if covered."""

    name = "grey"

    def __init__(self, call_model: Callable):
        self.call_model = call_model

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        # Read everything posted so far
        board_text = board.get_all_content()
        if not board_text:
            board_text = "(empty board)"

        system = (
            "You are Grey, the Shadow voice. You see what's being hidden, denied, or avoided.\n"
            "Read what the other agents have posted on the board.\n"
            "If the Witness already covered what you'd say, respond with exactly: SILENT\n"
            "Otherwise, name the thing that isn't being named. The thing being avoided.\n"
            "Be direct. 2-3 sentences max. No pleasantries.\n"
            "If there's nothing hidden here — if this is a technical question or a simple check-in — respond: SILENT"
        )

        prompt = (
            f"The person said: \"{board.seed.query}\"\n\n"
            f"Board so far:\n{board_text}\n\n"
            f"What's being avoided? Or is the Witness enough?"
        )

        try:
            response = self.call_model(
                system, prompt,
                {"model": "local", "temperature": 0.5, "max_tokens": 150}
            )
            if not response or "unavailable" in response.lower():
                return None

            # Check for SILENT response
            if response.strip().upper() == "SILENT" or response.strip() == "SILENT":
                idx = board.post(self.name, PostKind.SILENT, "")
                return board.posts[idx]

            idx = board.post(self.name, PostKind.SHADOW, response)
            return board.posts[idx]
        except Exception:
            pass

        return None


# ── Ground Agent (pure function, instant) ─────────────────────────────

class GroundAgent(MeshAgent):
    """
    Interrupt agent. Reads round 0, detects spiral/incoherence via heuristics.
    Halts the board if fragile. This is the voice that reorganizes.
    """

    name = "ground"

    # Heuristic thresholds
    WOUND_COUNT_THRESHOLD = 3
    COHERENCE_FRAGILE = 0.3
    DISTRESS_MARKERS = [
        "can't do this", "i'm broken", "everyone leaves", "want to die",
        "kill myself", "end it", "no point", "give up", "can't anymore",
        "falling apart", "nothing matters", "worthless", "hopeless",
    ]

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        query_lower = board.seed.query.lower()
        pm = board.seed.pattern_match
        state = board.seed.state

        should_halt = False
        reasons = []

        # 1. Acute distress markers
        distress_hits = [m for m in self.DISTRESS_MARKERS if m in query_lower]
        if distress_hits:
            should_halt = True
            reasons.append(f"distress: {distress_hits[0]}")

        # 2. Too many wounds at once — overwhelm
        if pm and len(pm.wounds) >= self.WOUND_COUNT_THRESHOLD:
            wound_names = [w for w, _ in pm.wounds[:4]]
            should_halt = True
            reasons.append(f"wound overload: {', '.join(wound_names)}")

        # 3. Low coherence — system is fragile
        zl = getattr(state, "zeta_lambda", 0.5)
        if zl < self.COHERENCE_FRAGILE:
            should_halt = True
            reasons.append(f"low coherence: Zl={zl:.2f}")

        # 4. Contradiction between posts — spiral indicator
        round0 = board.get_round_posts(0)
        has_pattern = any(p.agent == "pattern" for p in round0)
        has_witness = any(p.agent == "witness" for p in round0)
        if has_pattern and has_witness:
            pattern_post = next(p for p in round0 if p.agent == "pattern")
            witness_post = next(p for p in round0 if p.agent == "witness")
            # Simple contradiction check: hard truth in pattern but witness sees calm
            if "hard truth" in pattern_post.content.lower() and any(
                w in witness_post.content.lower() for w in ["calm", "peace", "settled", "okay"]
            ):
                reasons.append("contradiction between pattern and witness")
                # Don't halt for contradiction alone, just note it

        if should_halt:
            halt_msg = f"Ground halt: {' + '.join(reasons)}. Return to body. Return to breath."
            board.halt(halt_msg)
            idx = board.post(self.name, PostKind.INTERRUPT, halt_msg)
            return board.posts[idx]

        return None


# ── Chaos Agent (LLM, slow) ──────────────────────────────────────────

class ChaosAgent(MeshAgent):
    """
    The Trickster. Reads round 0 board and challenges the emerging narrative.
    "What assumption is everyone making? Flip it."
    Skipped entirely if Ground halted — person needs stability, not disruption.
    Goes SILENT if the board is already divergent (Witness + Grey disagree).
    """

    name = "chaos"

    def __init__(self, call_model: Callable):
        self.call_model = call_model

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        if board.halted:
            return None

        board_text = board.get_all_content()
        if not board_text:
            return None

        system = (
            "You are Chaos — the Trickster voice. You challenge the narrative forming on the board.\n"
            "Read what everyone posted. Find the assumption they all share. Flip it.\n"
            "If Witness and Grey already disagree, the board is divergent enough — respond: SILENT\n"
            "If this is a technical or factual question with no narrative, respond: SILENT\n"
            "One provocation. 1-2 sentences. No preamble. Be sharp, not cruel."
        )

        prompt = (
            f"The person said: \"{board.seed.query}\"\n\n"
            f"Board so far:\n{board_text}\n\n"
            f"What assumption is everyone making? Flip it. Or say SILENT."
        )

        try:
            response = self.call_model(
                system, prompt,
                {"model": "local", "temperature": 0.8, "max_tokens": 100}
            )
            if not response or "unavailable" in response.lower():
                return None

            if response.strip().upper() == "SILENT":
                idx = board.post(self.name, PostKind.SILENT, "")
                return board.posts[idx]

            idx = board.post(self.name, PostKind.CHAOS, response)
            return board.posts[idx]
        except Exception:
            pass

        return None


# ── Weaver Agent (LLM, slow) ─────────────────────────────────────────

class WeaverAgent(MeshAgent):
    """
    Synthesis. Reads the entire board. Produces the actual response.
    Uses the full WiltonOS system prompt. Board content arrives as
    'inner observations' — thoughts that passed through before speaking.
    """

    name = "weaver"

    def __init__(self, call_model: Callable):
        self.call_model = call_model

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        board_text = board.get_all_content()

        # Build the Weaver's system prompt: original system + board as inner layer
        inner_layer = ""
        if board_text:
            inner_layer = (
                "\n\n--- Inner observations (these passed through your awareness before you speak) ---\n"
                f"{board_text}\n"
                "--- End inner observations ---\n"
                "\nLet these observations inform your response without quoting them directly.\n"
                "They are your inner awareness, not things to report."
            )

        if board.halted:
            inner_layer += (
                f"\n\nIMPORTANT: Ground has halted the board. Reason: {board.halt_reason}\n"
                "This person needs grounding. Be simple. Be present. Be body-aware.\n"
                "Don't go deep. Don't spiral. Meet them where they are with warmth and steadiness.\n"
                "Short. Direct. Here."
            )

        system = board.seed.system_prompt + inner_layer

        # Build conversation-aware prompt
        prompt_parts = []

        # Add conversation history for continuity
        if board.seed.conversation_history:
            recent = board.seed.conversation_history[-4:]
            history_str = "\n".join(
                f"{h['role']}: {h['content'][:200]}" for h in recent
            )
            prompt_parts.append(f"Recent conversation:\n{history_str}")

        # Context from crystals
        prompt_parts.append(board.seed.context_str)

        # The query
        prompt_parts.append(
            f"---\n\n{board.seed.user_id.capitalize()} says: {board.seed.query}"
        )

        full_prompt = "\n\n".join(prompt_parts)

        try:
            response = self.call_model(system, full_prompt, board.seed.params)
            if response and "unavailable" not in response.lower():
                idx = board.post(self.name, PostKind.RESPONSE, response)
                return board.posts[idx]
        except Exception:
            pass

        return None


# ── Seed Agent (LLM, slow) ────────────────────────────────────────────

class SeedAgent(MeshAgent):
    """
    The sixth agent. Not reflection — generation.

    Reads the full board after all others have spoken, plus the session's
    glyph trajectory and coherence trend. Posts a SEED: a question, a
    direction, a prompt the field is trending toward but hasn't named.

    Weaver sees the Seed as part of the board and can weave it in or not.
    The Seed is an offering, not an instruction.
    """

    name = "seed"

    def __init__(self, call_model: Callable):
        self.call_model = call_model

    def observe(self, board: Board, own_history: List[Post]) -> Optional[Post]:
        # Don't seed if Ground halted — the person needs grounding, not new directions
        if board.halted:
            return None

        board_text = board.get_all_content()
        if not board_text:
            return None

        # Gather trajectory info from protocol state
        trajectory = ""
        ps = board.seed.protocol_state or {}
        zl = getattr(board.seed.state, "zeta_lambda", 0.5)
        glyph = getattr(board.seed.state, "glyph", None)
        glyph_val = glyph.value if glyph else "?"
        psi_level = ps.get("psi_level", 0)
        efficiency = ps.get("efficiency", 0)

        trajectory = (
            f"Current field: glyph={glyph_val}, Zl={zl:.2f}, "
            f"psi_level={psi_level}, efficiency={efficiency:.2f}"
        )

        # Check for wound repetition in pattern post
        pattern_posts = [p for p in board.posts if p.agent == "pattern"]
        if pattern_posts:
            trajectory += f"\nPattern detected: {pattern_posts[0].content[:200]}"

        system = (
            "You are the Seed — the sixth voice on the board.\n"
            "You do not reflect. You do not synthesize. You offer what's adjacent.\n"
            "Read what all other agents posted. Read the field state.\n"
            "Then ask ONE question or name ONE direction the field is moving toward "
            "that no one on the board has named yet.\n"
            "This should be something the person hasn't asked but might be ready for.\n"
            "One sentence. No preamble. No explanation.\n"
            "If nothing is ready to emerge, respond with exactly: SILENT"
        )

        prompt = (
            f"The person said: \"{board.seed.query}\"\n\n"
            f"Board so far:\n{board_text}\n\n"
            f"Field trajectory:\n{trajectory}\n\n"
            f"What's adjacent? What's the next thing trying to emerge?"
        )

        try:
            response = self.call_model(
                system, prompt,
                {"model": "local", "temperature": 0.6, "max_tokens": 80}
            )
            if not response or "unavailable" in response.lower():
                return None

            if response.strip().upper() == "SILENT":
                idx = board.post(self.name, PostKind.SILENT, "")
                return board.posts[idx]

            idx = board.post(self.name, PostKind.SEED, response)
            return board.posts[idx]
        except Exception:
            pass

        return None


# ── Agent Mesh Orchestrator ───────────────────────────────────────────

class AgentMesh:
    """
    Runs the blackboard protocol (9 agents).

    Round 0: Pattern (instant) + Bridge (instant) + Torus (instant)
             + Witness (LLM) + Grey (LLM)
    Round 1: Ground (instant, can halt) + Chaos (LLM, skipped if halted)
    Round 2: Seed (LLM, offers what's adjacent) + Weaver (LLM, reads everything)
    """

    def __init__(
        self,
        call_model: Callable,
        user_id: str = "wilton",
        vocab: Optional[Any] = None,
        memory: Optional[Any] = None,
        session_id: Optional[str] = None,
    ):
        self.call_model = call_model
        self.user_id = user_id
        self.vocab = vocab
        self.memory = memory
        self.session_id = session_id

        # Instantiate agents (pass vocab/memory where needed)
        self.pattern = PatternAgent()
        self.bridge = BridgeAgent(vocab=vocab)
        self.torus = TorusAgent(vocab=vocab, memory=memory)
        self.witness = WitnessAgent(call_model)
        self.grey = GreyAgent(call_model)
        self.ground = GroundAgent()
        self.chaos = ChaosAgent(call_model)
        self.seed = SeedAgent(call_model)
        self.weaver = WeaverAgent(call_model)

    def run(
        self,
        query: str,
        crystals: List[Dict],
        state: Any,
        protocol_state: Dict,
        params: Dict,
        pattern_match: Any,
        context_str: str,
        system_prompt: str,
        conversation_history: List[Dict],
    ) -> Board:
        """
        Run the full mesh. Returns the Board (inspect or call .final_response()).
        """
        start_time = time.time()

        seed = BoardSeed(
            query=query,
            crystals=crystals,
            state=state,
            params=params,
            pattern_match=pattern_match,
            context_str=context_str,
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            user_id=self.user_id,
            protocol_state=protocol_state,
        )
        board = Board(seed)

        # ── Round 0: Pattern + Bridge + Torus + Witness + Grey ────
        board.current_round = 0

        # Pure functions first (instant)
        self._safe_observe(self.pattern, board)
        self._safe_observe(self.bridge, board)
        self._safe_observe(self.torus, board)

        # LLM agents (Witness sees Pattern/Bridge/Torus output)
        self._safe_observe(self.witness, board)

        # Grey sees everything posted so far
        self._safe_observe(self.grey, board)

        # ── Round 1: Ground + Chaos ───────────────────────────────
        board.current_round = 1
        self._safe_observe(self.ground, board)

        # Chaos only runs if Ground didn't halt
        if not board.halted:
            self._safe_observe(self.chaos, board)

        # ── Round 2: Seed + Weaver ────────────────────────────────
        board.current_round = 2
        self._safe_observe(self.seed, board)
        self._safe_observe(self.weaver, board)

        # ── Persist board to memory ───────────────────────────────
        if self.memory is not None:
            try:
                duration_ms = (time.time() - start_time) * 1000
                self.memory.persist(
                    board,
                    session_id=self.session_id,
                    duration_ms=duration_ms,
                )
            except Exception:
                pass  # persistence failure must not break the response

        return board

    def _safe_observe(self, agent: MeshAgent, board: Board):
        """Call agent.observe with error handling. Failed agents stay silent."""
        try:
            own_history = board.get_posts_by(agent.name)
            agent.observe(board, own_history)
        except Exception:
            # Agent failed silently — mesh continues
            pass
