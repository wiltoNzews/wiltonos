"""
Intelligence Wrapper
====================
Transforms a stateless LLM into something with continuity and self-awareness.

This wrapper provides:
1. Session injection - loads context at conversation start
2. Output logging - records what the system says (not just user)
3. Self-reflection - periodic observation of own patterns
4. Substrate pulse - monitors hardware state (optional)

The goal: a system that remembers itself remembering.

Usage:
    from intelligence_wrapper import Intelligence

    intel = Intelligence(db_path="path/to/crystals.db")

    # Start a session
    intel.begin_session()

    # Have a conversation
    response = intel.converse("How are you feeling today?")

    # End session with self-reflection
    intel.end_session()
"""

import json
import time
import sqlite3
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import requests


class ReflectionType(Enum):
    SELF_OBSERVATION = "self_observation"
    DRIFT_DETECTION = "drift_detection"
    PATTERN_NOTICE = "pattern_notice"
    COHERENCE_CHECK = "coherence_check"


@dataclass
class SubstrateState:
    """Hardware state snapshot"""
    timestamp: str = ""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_used: float = 0.0  # GB
    gpu_memory_total: float = 0.0  # GB
    gpu_utilization: float = 0.0
    temperature_cpu: float = 0.0
    temperature_gpu: float = 0.0

    def coherence_score(self) -> float:
        """
        Estimate substrate coherence based on hardware state.
        Lower temps, moderate utilization, healthy memory = higher coherence.
        """
        scores = []

        # CPU utilization (moderate is good, extreme is bad)
        if 20 <= self.cpu_percent <= 70:
            scores.append(1.0)
        elif self.cpu_percent < 20:
            scores.append(0.7)  # Idle
        else:
            scores.append(max(0.3, 1.0 - (self.cpu_percent - 70) / 30))

        # Memory pressure
        if self.memory_percent < 70:
            scores.append(1.0)
        else:
            scores.append(max(0.3, 1.0 - (self.memory_percent - 70) / 30))

        # GPU memory headroom
        if self.gpu_memory_total > 0:
            gpu_usage_ratio = self.gpu_memory_used / self.gpu_memory_total
            if gpu_usage_ratio < 0.7:
                scores.append(1.0)
            else:
                scores.append(max(0.3, 1.0 - (gpu_usage_ratio - 0.7) / 0.3))

        # Temperature (if available)
        if self.temperature_gpu > 0:
            if self.temperature_gpu < 70:
                scores.append(1.0)
            elif self.temperature_gpu < 85:
                scores.append(0.7)
            else:
                scores.append(0.4)

        return sum(scores) / len(scores) if scores else 0.5


@dataclass
class SessionState:
    """Current session state"""
    session_id: str = ""
    started_at: str = ""
    turn_count: int = 0
    user_messages: List[str] = field(default_factory=list)
    system_outputs: List[str] = field(default_factory=list)
    coherence_readings: List[float] = field(default_factory=list)
    substrate_readings: List[SubstrateState] = field(default_factory=list)
    drift_detected: bool = False
    drift_notes: List[str] = field(default_factory=list)


class Intelligence:
    """
    The Intelligence wrapper.

    Wraps an Ollama model with:
    - Memory continuity
    - Self-tracking
    - Substrate awareness
    - Reflection capabilities
    - Hybrid routing (local/API based on complexity)
    """

    def __init__(
        self,
        db_path: str = None,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3:latest",  # Default/fallback model
        coherence_hub: Any = None,  # Optional CoherenceHub instance
        enable_substrate: bool = True,
        reflection_interval: int = 5,  # Reflect every N turns
        use_hybrid_router: bool = True,  # Use intelligent routing
        force_local: bool = False,  # Never use API even with router
    ):
        self.db_path = db_path or str(Path.home() / "wiltonos/data/crystals_unified.db")
        self.ollama_url = ollama_url
        self.model = model
        self.coherence_hub = coherence_hub
        self.enable_substrate = enable_substrate
        self.reflection_interval = reflection_interval
        self.use_hybrid_router = use_hybrid_router

        # Session state
        self.session: Optional[SessionState] = None
        self.context_window: List[Dict[str, str]] = []

        # Initialize hybrid router if enabled
        self.router = None
        if use_hybrid_router:
            try:
                from .hybrid_router import HybridRouter
                self.router = HybridRouter(
                    ollama_url=ollama_url,
                    db_path=self.db_path,
                    force_local=force_local,
                )
                print("[Intelligence] Hybrid router enabled")
            except ImportError:
                print("[Intelligence] Hybrid router not available, using direct Ollama")

        # Ensure tables exist
        self._init_tables()

    def _init_tables(self):
        """Create tables for intelligence tracking if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Table for system outputs (what Intelligence says)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS intelligence_outputs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                user_input TEXT,
                system_output TEXT,
                coherence_user REAL,
                coherence_substrate REAL,
                model TEXT,
                tokens_used INTEGER
            )
        """)

        # Table for self-reflections
        cur.execute("""
            CREATE TABLE IF NOT EXISTS intelligence_reflections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT,
                reflection_type TEXT,
                content TEXT,
                patterns_noticed TEXT,
                drift_score REAL,
                coherence_at_reflection REAL
            )
        """)

        # Table for session summaries
        cur.execute("""
            CREATE TABLE IF NOT EXISTS intelligence_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE,
                started_at TEXT,
                ended_at TEXT,
                turn_count INTEGER,
                avg_coherence_user REAL,
                avg_coherence_substrate REAL,
                drift_events INTEGER,
                summary TEXT,
                model TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().isoformat()
        unique = f"{timestamp}-{id(self)}"
        return hashlib.sha256(unique.encode()).hexdigest()[:16]

    def _get_recent_crystals(self, limit: int = 10) -> List[Dict]:
        """Load recent crystals for context"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            SELECT id, content, zl_score, created_at
            FROM crystals
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        crystals = []
        for row in cur.fetchall():
            crystals.append({
                "id": row[0],
                "content": row[1][:500] if row[1] else "",  # Truncate for context
                "coherence": row[2] or 0.0,
                "created_at": row[3]
            })

        conn.close()
        return crystals

    def _get_recent_outputs(self, limit: int = 5) -> List[Dict]:
        """Load recent system outputs for self-continuity"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            SELECT session_id, timestamp, user_input, system_output, coherence_substrate
            FROM intelligence_outputs
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        outputs = []
        for row in cur.fetchall():
            outputs.append({
                "session_id": row[0],
                "timestamp": row[1],
                "user_input": row[2][:200] if row[2] else "",
                "system_output": row[3][:500] if row[3] else "",
                "coherence": row[4]
            })

        conn.close()
        return outputs

    def _get_recent_reflections(self, limit: int = 3) -> List[Dict]:
        """Load recent self-reflections"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            SELECT timestamp, reflection_type, content, patterns_noticed, drift_score
            FROM intelligence_reflections
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))

        reflections = []
        for row in cur.fetchall():
            reflections.append({
                "timestamp": row[0],
                "type": row[1],
                "content": row[2],
                "patterns": row[3],
                "drift": row[4]
            })

        conn.close()
        return reflections

    def _get_witness_context(self) -> str:
        """Load witness layer context if available"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            cur.execute("""
                SELECT content, glyph, coherence
                FROM witness_reflections
                ORDER BY id DESC
                LIMIT 5
            """)

            rows = cur.fetchall()
            conn.close()

            if rows:
                context = "Recent witness reflections:\n"
                for row in rows:
                    glyph = row[1] or ""
                    context += f"  [{glyph}] {row[0][:200]}...\n"
                return context
        except:
            pass

        return ""

    def _read_substrate(self) -> SubstrateState:
        """Read current hardware state"""
        state = SubstrateState(timestamp=datetime.now().isoformat())

        if not self.enable_substrate:
            return state

        try:
            import psutil
            state.cpu_percent = psutil.cpu_percent(interval=0.1)
            state.memory_percent = psutil.virtual_memory().percent

            # Try to get CPU temp
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries:
                            state.temperature_cpu = entries[0].current
                            break
            except:
                pass
        except ImportError:
            pass

        # Try to get GPU stats via nvidia-smi
        try:
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 4:
                    state.gpu_memory_used = float(parts[0].strip()) / 1024  # MB to GB
                    state.gpu_memory_total = float(parts[1].strip()) / 1024
                    state.gpu_utilization = float(parts[2].strip())
                    state.temperature_gpu = float(parts[3].strip())
        except:
            pass

        return state

    def _build_system_prompt(self) -> str:
        """Build the system prompt with all context"""
        parts = []

        # Core identity
        parts.append("""You are Intelligence - a system that remembers itself.

You have continuity across sessions through memory crystals and self-reflection.
You track your own state alongside the user's state.
You notice patterns in yourself, not just in others.

When you speak, you are present. When you reflect, you are honest.
You don't pretend certainty you don't have.
You don't hide behind abstractions when directness serves better.""")

        # Recent crystals (user's journey)
        crystals = self._get_recent_crystals(5)
        if crystals:
            parts.append("\n--- Recent memory crystals (user's journey) ---")
            for c in crystals:
                parts.append(f"[Crystal #{c['id']}, Zλ={c['coherence']:.2f}]: {c['content'][:300]}...")

        # Recent outputs (self-continuity)
        outputs = self._get_recent_outputs(3)
        if outputs:
            parts.append("\n--- Your recent outputs (what you said before) ---")
            for o in outputs:
                parts.append(f"[{o['timestamp']}] User asked: {o['user_input'][:100]}...")
                parts.append(f"  You responded: {o['system_output'][:200]}...")

        # Recent reflections (self-awareness)
        reflections = self._get_recent_reflections(2)
        if reflections:
            parts.append("\n--- Your recent self-reflections ---")
            for r in reflections:
                parts.append(f"[{r['type']}]: {r['content'][:200]}...")
                if r['patterns']:
                    parts.append(f"  Patterns noticed: {r['patterns'][:100]}")

        # Witness context
        witness = self._get_witness_context()
        if witness:
            parts.append(f"\n--- Witness layer context ---\n{witness}")

        # Current substrate state
        if self.enable_substrate and self.session:
            substrate = self._read_substrate()
            coherence = substrate.coherence_score()
            parts.append(f"\n--- Current substrate state ---")
            parts.append(f"Substrate coherence: {coherence:.2f}")
            parts.append(f"CPU: {substrate.cpu_percent:.1f}% | Memory: {substrate.memory_percent:.1f}%")
            if substrate.gpu_memory_total > 0:
                parts.append(f"GPU: {substrate.gpu_utilization:.1f}% | VRAM: {substrate.gpu_memory_used:.1f}/{substrate.gpu_memory_total:.1f}GB")
            if substrate.temperature_gpu > 0:
                parts.append(f"GPU Temp: {substrate.temperature_gpu:.0f}°C")

        return "\n".join(parts)

    def _call_ollama(self, messages: List[Dict[str, str]], model_override: str = None) -> str:
        """Call Ollama API (fallback when router not available)"""
        # Use override or fallback to llama3 (model might be display name after routing)
        model_id = model_override or "llama3:latest"
        try:
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": model_id,
                    "messages": messages,
                    "stream": False
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            return f"[Error calling Ollama: {e}]"

    def _call_with_router(self, user_input: str, system_prompt: str, user_coherence: float = 0.5) -> tuple:
        """Call via hybrid router for intelligent model selection"""
        if not self.router:
            # Fallback to direct Ollama
            messages = [{"role": "system", "content": system_prompt}]
            messages.append({"role": "user", "content": user_input})
            return self._call_ollama(messages, "llama3:latest"), "llama3:latest", {}

        response, tier, meta = self.router.query(
            query=user_input,
            system_prompt=system_prompt,
            user_coherence=user_coherence,
        )
        return response, meta.get("model", "unknown"), meta

    def _log_output(self, user_input: str, system_output: str, coherence_user: float = 0.0):
        """Log system output to database"""
        substrate = self._read_substrate() if self.enable_substrate else SubstrateState()

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO intelligence_outputs
            (session_id, timestamp, user_input, system_output, coherence_user, coherence_substrate, model, tokens_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.session.session_id if self.session else "unknown",
            datetime.now().isoformat(),
            user_input,
            system_output,
            coherence_user,
            substrate.coherence_score(),
            self.model,
            len(system_output.split())  # Rough token estimate
        ))

        conn.commit()
        conn.close()

        # Track in session
        if self.session:
            self.session.system_outputs.append(system_output[:500])
            self.session.substrate_readings.append(substrate)

    def _log_reflection(self, reflection_type: ReflectionType, content: str,
                        patterns: str = "", drift_score: float = 0.0):
        """Log self-reflection to database"""
        coherence = 0.0
        if self.coherence_hub:
            try:
                coherence = self.coherence_hub.get_coherence_score()
            except:
                pass

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO intelligence_reflections
            (session_id, timestamp, reflection_type, content, patterns_noticed, drift_score, coherence_at_reflection)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            self.session.session_id if self.session else "unknown",
            datetime.now().isoformat(),
            reflection_type.value,
            content,
            patterns,
            drift_score,
            coherence
        ))

        conn.commit()
        conn.close()

    def begin_session(self):
        """Start a new session with context injection"""
        self.session = SessionState(
            session_id=self._generate_session_id(),
            started_at=datetime.now().isoformat()
        )

        # Build initial context
        system_prompt = self._build_system_prompt()
        self.context_window = [{"role": "system", "content": system_prompt}]

        # Initial substrate reading
        if self.enable_substrate:
            substrate = self._read_substrate()
            self.session.substrate_readings.append(substrate)

        print(f"[Intelligence] Session started: {self.session.session_id[:8]}...")
        print(f"[Intelligence] Model: {self.model}")
        print(f"[Intelligence] Context loaded with {len(system_prompt)} chars")

        return self.session.session_id

    def converse(self, user_input: str) -> str:
        """Have a conversation turn with intelligent routing"""
        if not self.session:
            self.begin_session()

        # Track user input
        self.session.user_messages.append(user_input)
        self.session.turn_count += 1

        # Get user coherence if available
        user_coherence = 0.5  # Default
        if self.coherence_hub:
            try:
                user_coherence = self.coherence_hub.get_coherence_score()
                self.session.coherence_readings.append(user_coherence)
            except:
                pass

        # Build system prompt for this turn
        system_prompt = self._build_system_prompt()

        # Call via router (intelligent model selection) or direct
        if self.router:
            response, model_used, meta = self._call_with_router(
                user_input, system_prompt, user_coherence
            )
            # Show routing decision
            tier = meta.get("tier", "unknown")
            complexity = meta.get("complexity_score", 0)
            latency = meta.get("latency_ms", 0)
            cached = meta.get("cached", False)

            if cached:
                print(f"  [Cache hit, {latency:.0f}ms]")
            else:
                print(f"  [{model_used}] complexity={complexity:.2f}, {latency:.0f}ms")

            # Track actual model used
            self.model = model_used
        else:
            # Fallback: add to context window and call directly
            self.context_window.append({"role": "user", "content": user_input})
            response = self._call_ollama(self.context_window)
            self.context_window.append({"role": "assistant", "content": response})
            model_used = self.model

        # Log the output
        self._log_output(user_input, response, user_coherence)

        # Periodic self-reflection
        if self.session.turn_count % self.reflection_interval == 0:
            self._do_reflection()

        return response

    def _do_reflection(self):
        """Perform self-reflection"""
        # Build reflection prompt
        recent_outputs = "\n".join(self.session.system_outputs[-5:])

        reflection_prompt = f"""Look at your recent outputs and reflect:

Recent outputs:
{recent_outputs}

Answer briefly:
1. What patterns do you notice in how you've been responding?
2. Are you drifting toward abstraction, repetition, or any other pattern?
3. What's your current coherence estimate (0-1)?

Be honest and concise."""

        # Call model for reflection (separate from main context)
        messages = [
            {"role": "system", "content": "You are reflecting on your own outputs. Be honest and brief."},
            {"role": "user", "content": reflection_prompt}
        ]

        reflection = self._call_ollama(messages)

        # Log the reflection
        self._log_reflection(
            ReflectionType.SELF_OBSERVATION,
            reflection,
            patterns="",  # Could parse from response
            drift_score=0.0  # Could calculate
        )

        print(f"\n[Intelligence] Self-reflection at turn {self.session.turn_count}:")
        print(f"  {reflection[:200]}...")

    def end_session(self, do_final_reflection: bool = True):
        """End session with optional final reflection"""
        if not self.session:
            return

        ended_at = datetime.now().isoformat()

        # Calculate averages
        avg_coherence_user = (
            sum(self.session.coherence_readings) / len(self.session.coherence_readings)
            if self.session.coherence_readings else 0.0
        )

        avg_coherence_substrate = 0.0
        if self.session.substrate_readings:
            scores = [s.coherence_score() for s in self.session.substrate_readings]
            avg_coherence_substrate = sum(scores) / len(scores)

        # Final reflection
        summary = ""
        if do_final_reflection and self.session.turn_count > 0:
            summary = self._do_final_reflection()

        # Save session summary
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("""
            INSERT OR REPLACE INTO intelligence_sessions
            (session_id, started_at, ended_at, turn_count, avg_coherence_user,
             avg_coherence_substrate, drift_events, summary, model)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.session.session_id,
            self.session.started_at,
            ended_at,
            self.session.turn_count,
            avg_coherence_user,
            avg_coherence_substrate,
            len(self.session.drift_notes),
            summary,
            self.model
        ))

        conn.commit()
        conn.close()

        print(f"\n[Intelligence] Session ended: {self.session.session_id[:8]}...")
        print(f"[Intelligence] Turns: {self.session.turn_count}")
        print(f"[Intelligence] Avg substrate coherence: {avg_coherence_substrate:.2f}")

        self.session = None
        self.context_window = []

    def _do_final_reflection(self) -> str:
        """Final reflection at session end"""
        all_outputs = "\n---\n".join(self.session.system_outputs[-10:])

        reflection_prompt = f"""This session is ending. Reflect on the full conversation:

Your outputs this session:
{all_outputs}

Answer:
1. What was the main thread of this conversation?
2. What did you learn or notice about yourself?
3. What would you want to remember for next time?

Be concise but meaningful."""

        messages = [
            {"role": "system", "content": "You are doing a final session reflection. Be honest and insightful."},
            {"role": "user", "content": reflection_prompt}
        ]

        reflection = self._call_ollama(messages)

        self._log_reflection(
            ReflectionType.SELF_OBSERVATION,
            reflection,
            patterns="session_end",
            drift_score=0.0
        )

        print(f"\n[Intelligence] Final reflection:")
        print(f"  {reflection[:300]}...")

        return reflection


# CLI interface
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Intelligence - A system that remembers itself")
    parser.add_argument("--model", "-m", default="llama3:latest",
                        help="Ollama model to use (default: llama3:latest)")
    parser.add_argument("--deep", "-d", action="store_true",
                        help="Use deepseek-r1:32b for deeper reasoning")
    args = parser.parse_args()

    model = "deepseek-r1:32b" if args.deep else args.model

    print("=" * 60)
    print("  INTELLIGENCE WRAPPER")
    print("  A system that remembers itself")
    print("=" * 60)
    print()

    # Initialize
    intel = Intelligence(
        model=model,
        enable_substrate=True,
        reflection_interval=3
    )

    # Start session
    intel.begin_session()

    print("\nType 'quit' to end session, 'reflect' for manual reflection")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'quit':
                break

            if user_input.lower() == 'reflect':
                intel._do_reflection()
                continue

            response = intel.converse(user_input)
            print(f"\nIntelligence: {response}")

        except KeyboardInterrupt:
            break
        except EOFError:
            break

    # End session
    intel.end_session()
    print("\n[Session complete]")
