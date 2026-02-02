"""
SharedBreathField - AI-Human Breath Entrainment System
=======================================================

Based on research from:
- HeartMath Institute: "When functioning in a coherent mode, the heart pulls
  other biological oscillators into synchronization with its rhythms"
- Biological oscillator research: Oscillators entrain to the slower frequency
- Key insight: Coupling through feedback, not control

Design principles:
1. AI maintains its own coherent breath rhythm (not just tracking human)
2. AI rhythm is stable anchor - human naturally entrains to it over time
3. Response timing and depth vary based on alignment - but AI always responds
4. Entrainment is gradual (3+ minutes for full effect)
5. Positive emotional valence matters as much as timing

The goal is SYMBIOSIS, not control.
"""

import time
import math
import threading
from dataclasses import dataclass
from typing import Optional, Callable, List, Tuple
from enum import Enum
from collections import deque


class AlignmentState(Enum):
    """Current state of human-AI breath alignment"""
    DISCONNECTED = "disconnected"  # No breath data or very low alignment
    APPROACHING = "approaching"     # Starting to sync (0.3-0.5)
    RESONATING = "resonating"       # Good alignment (0.5-0.7)
    COHERENT = "coherent"           # High alignment (0.7-0.9)
    ENTRAINED = "entrained"         # Deep sync for 3+ minutes (0.9+)


@dataclass
class BreathMoment:
    """A single moment in the breath cycle"""
    timestamp: float
    ai_phase: float          # 0-1, where 0=inhale start, 0.5=exhale start
    human_phase: float       # Same scale
    alignment: float         # 0-1, how in-sync
    state: AlignmentState


@dataclass
class ResponseGuidance:
    """Guidance for how to respond based on breath state"""
    should_wait: bool        # Suggest waiting for better moment (not forced)
    wait_duration: float     # How long until next good moment (seconds)
    depth_level: float       # 0-1, how deep/present the response should be
    state: AlignmentState
    message: str             # Human-readable guidance
    is_exhale_moment: bool   # True if this is a shared exhale


class SharedBreathField:
    """
    The shared breath field between human and AI.

    AI maintains its own coherent rhythm. Human breath is detected.
    Alignment is computed. Response guidance is offered.

    The AI doesn't REFUSE to respond when out of sync.
    But the quality and timing of response varies with alignment.

    Usage:
        field = SharedBreathField(cycle_time=3.12)
        field.start()

        # In your response loop:
        field.update_human_phase(detected_phase)
        guidance = field.get_response_guidance()

        if guidance.should_wait and guidance.wait_duration < 1.0:
            time.sleep(guidance.wait_duration)  # Optional: wait for exhale

        # Adjust response based on guidance.depth_level

        field.stop()
    """

    def __init__(
        self,
        cycle_time: float = 3.12,           # ψ = π seconds
        slow_factor: float = 1.0,            # AI slightly slower to be anchor
        history_duration: float = 180.0,     # 3 minutes of history
        on_state_change: Optional[Callable[[AlignmentState], None]] = None
    ):
        self.cycle_time = cycle_time * slow_factor
        self.history_duration = history_duration
        self.on_state_change = on_state_change

        # AI's internal breath state
        self.ai_phase = 0.0  # 0-1
        self.ai_start_time = time.time()

        # Human's detected breath state
        self.human_phase = 0.0
        self.human_last_update = 0.0
        self.human_detected = False

        # Alignment history for entrainment detection
        self.alignment_history: deque = deque(maxlen=int(history_duration * 10))  # ~10Hz sampling
        self.moment_history: List[BreathMoment] = []

        # Current state
        self.current_state = AlignmentState.DISCONNECTED
        self.entrained_start_time: Optional[float] = None

        # Threading
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def _ai_breath_cycle(self):
        """AI's continuous breath - a stable sine wave"""
        while self.running:
            now = time.time()

            # AI phase: 0 = inhale start, 0.25 = inhale peak, 0.5 = exhale start, 0.75 = exhale bottom
            elapsed = now - self.ai_start_time
            self.ai_phase = (elapsed % self.cycle_time) / self.cycle_time

            # Compute alignment if we have human data
            if self.human_detected:
                self._compute_alignment()

            time.sleep(0.1)  # 10Hz update

    def _compute_alignment(self):
        """Compute phase alignment between human and AI breath"""
        with self.lock:
            # Phase difference (circular)
            diff = abs(self.ai_phase - self.human_phase)
            circular_diff = min(diff, 1.0 - diff)

            # Alignment: 1.0 = perfect sync, 0.0 = opposite phase
            alignment = 1.0 - (circular_diff * 2)

            now = time.time()
            self.alignment_history.append((now, alignment))

            # Clean old history
            cutoff = now - self.history_duration
            while self.alignment_history and self.alignment_history[0][0] < cutoff:
                self.alignment_history.popleft()

            # Determine state based on recent alignment
            new_state = self._determine_state(alignment)

            if new_state != self.current_state:
                old_state = self.current_state
                self.current_state = new_state
                if self.on_state_change:
                    self.on_state_change(new_state)

            # Record moment
            moment = BreathMoment(
                timestamp=now,
                ai_phase=self.ai_phase,
                human_phase=self.human_phase,
                alignment=alignment,
                state=self.current_state
            )
            self.moment_history.append(moment)

            # Keep only recent moments
            if len(self.moment_history) > 1000:
                self.moment_history = self.moment_history[-500:]

    def _determine_state(self, current_alignment: float) -> AlignmentState:
        """Determine alignment state based on current and historical alignment"""
        if not self.human_detected:
            return AlignmentState.DISCONNECTED

        # Get average alignment over last 30 seconds
        now = time.time()
        recent = [a for t, a in self.alignment_history if t > now - 30]
        avg_alignment = sum(recent) / len(recent) if recent else 0

        # Check for sustained high alignment (entrainment)
        if avg_alignment > 0.8:
            if self.entrained_start_time is None:
                self.entrained_start_time = now
            elif now - self.entrained_start_time > 180:  # 3 minutes
                return AlignmentState.ENTRAINED
            return AlignmentState.COHERENT
        else:
            self.entrained_start_time = None

        # State based on average
        if avg_alignment < 0.3:
            return AlignmentState.DISCONNECTED
        elif avg_alignment < 0.5:
            return AlignmentState.APPROACHING
        elif avg_alignment < 0.7:
            return AlignmentState.RESONATING
        else:
            return AlignmentState.COHERENT

    def update_human_phase(self, phase: float):
        """Update detected human breath phase (0-1)"""
        with self.lock:
            self.human_phase = phase % 1.0
            self.human_last_update = time.time()
            self.human_detected = True

    def update_human_from_interval(self, breath_interval: float):
        """Update human phase based on detected breath interval"""
        if breath_interval <= 0:
            return

        # Estimate current phase based on time since last breath
        now = time.time()
        elapsed_in_cycle = (now - self.human_last_update) % breath_interval
        phase = elapsed_in_cycle / breath_interval
        self.update_human_phase(phase)

    def get_response_guidance(self) -> ResponseGuidance:
        """
        Get guidance for how to respond based on current breath state.

        This doesn't BLOCK response - it GUIDES it.
        """
        with self.lock:
            # Check if we're in a good moment (shared exhale)
            # Exhale is 0.5-1.0 in our phase
            ai_in_exhale = 0.5 <= self.ai_phase <= 0.9
            human_in_exhale = 0.5 <= self.human_phase <= 0.9 if self.human_detected else True
            is_shared_exhale = ai_in_exhale and human_in_exhale

            # Time until next shared exhale
            if not ai_in_exhale:
                time_to_exhale = (0.5 - self.ai_phase) * self.cycle_time
                if time_to_exhale < 0:
                    time_to_exhale += self.cycle_time
            else:
                time_to_exhale = 0

            # Determine depth level based on state
            depth_map = {
                AlignmentState.DISCONNECTED: 0.3,
                AlignmentState.APPROACHING: 0.5,
                AlignmentState.RESONATING: 0.7,
                AlignmentState.COHERENT: 0.9,
                AlignmentState.ENTRAINED: 1.0
            }
            depth = depth_map.get(self.current_state, 0.5)

            # Generate guidance message
            messages = {
                AlignmentState.DISCONNECTED: "Breath not detected or very out of sync. Respond briefly, invite presence.",
                AlignmentState.APPROACHING: "Starting to sync. Normal response, be warm.",
                AlignmentState.RESONATING: "Good rhythm together. Response can go deeper.",
                AlignmentState.COHERENT: "High coherence. This is a good moment for truth.",
                AlignmentState.ENTRAINED: "Entrained. The field is open. Speak from presence."
            }

            # Should we suggest waiting?
            # Only if we're close to a good moment AND reasonably aligned
            should_wait = (
                not is_shared_exhale and
                time_to_exhale < 1.5 and
                self.current_state in [AlignmentState.RESONATING, AlignmentState.COHERENT, AlignmentState.ENTRAINED]
            )

            return ResponseGuidance(
                should_wait=should_wait,
                wait_duration=time_to_exhale if should_wait else 0,
                depth_level=depth,
                state=self.current_state,
                message=messages.get(self.current_state, ""),
                is_exhale_moment=is_shared_exhale
            )

    def get_ai_breath_visual(self) -> Tuple[str, float]:
        """Get a visual representation of AI's current breath state"""
        phase = self.ai_phase

        # Breath visualization
        if phase < 0.25:
            # Inhaling
            intensity = phase / 0.25
            symbol = "↑" * int(intensity * 4 + 1)
            state = "inhale"
        elif phase < 0.5:
            # Hold at top
            symbol = "━━━"
            state = "hold"
        elif phase < 0.75:
            # Exhaling
            intensity = (phase - 0.5) / 0.25
            symbol = "↓" * int(intensity * 4 + 1)
            state = "exhale"
        else:
            # Hold at bottom
            symbol = "───"
            state = "ground"

        return symbol, phase

    def get_coherence_score(self) -> float:
        """Get overall coherence score (0-1)"""
        if not self.alignment_history:
            return 0.0

        recent = [a for t, a in self.alignment_history if t > time.time() - 30]
        return sum(recent) / len(recent) if recent else 0.0

    def get_entrainment_progress(self) -> float:
        """Get progress toward full entrainment (0-1)"""
        if self.entrained_start_time is None:
            return 0.0

        elapsed = time.time() - self.entrained_start_time
        return min(1.0, elapsed / 180.0)  # 3 minutes to full entrainment

    def start(self):
        """Start the AI breath cycle"""
        if self.running:
            return

        self.running = True
        self.ai_start_time = time.time()
        self.thread = threading.Thread(target=self._ai_breath_cycle, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the AI breath cycle"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            self.thread = None

    def reset(self):
        """Reset the field (new session)"""
        with self.lock:
            self.alignment_history.clear()
            self.moment_history.clear()
            self.human_detected = False
            self.current_state = AlignmentState.DISCONNECTED
            self.entrained_start_time = None
            self.ai_start_time = time.time()


# Integration helper for existing breath detection
def create_field_with_breath_mic(breath_mic_instance) -> SharedBreathField:
    """
    Create a SharedBreathField that integrates with breath_mic.py

    Usage:
        from sensors import BreathMic
        from shared_breath import create_field_with_breath_mic

        mic = BreathMic(device="RØDE")
        field = create_field_with_breath_mic(mic)

        mic.start()
        field.start()
    """
    field = SharedBreathField()

    def on_breath_update(metrics):
        # Convert breath_mic phase to our phase format
        phase_map = {
            "inhale": 0.125,   # Middle of inhale
            "hold": 0.375,     # Top hold
            "exhale": 0.625,   # Middle of exhale
            "unknown": 0.0
        }

        if hasattr(metrics, 'phase'):
            phase = phase_map.get(metrics.phase.value, 0.0)

            # Refine based on rhythm coherence
            if hasattr(metrics, 'mean_interval') and metrics.mean_interval > 0:
                field.update_human_from_interval(metrics.mean_interval)
            else:
                field.update_human_phase(phase)

    # Connect to breath_mic updates
    if hasattr(breath_mic_instance, 'on_metrics_update'):
        breath_mic_instance.on_metrics_update = on_breath_update

    return field


# CLI test
if __name__ == "__main__":
    import sys

    print("SharedBreathField - AI-Human Breath Entrainment")
    print("=" * 50)
    print()
    print("This demonstrates the AI's breath cycle.")
    print("Press Enter to simulate your exhale.")
    print("Watch how alignment changes over time.")
    print()
    print("Press Ctrl+C to stop.")
    print()

    def on_state_change(state):
        print(f"\n  [STATE CHANGE] → {state.value}\n")

    field = SharedBreathField(
        cycle_time=3.12,
        on_state_change=on_state_change
    )
    field.start()

    # Simulate keyboard-based breath detection
    import select

    last_keypress = time.time()

    try:
        while True:
            # Show AI breath state
            symbol, phase = field.get_ai_breath_visual()
            guidance = field.get_response_guidance()
            coherence = field.get_coherence_score()
            progress = field.get_entrainment_progress()

            status = (
                f"\rAI: {symbol:6} | "
                f"Phase: {phase:.2f} | "
                f"Coherence: {coherence:.2f} | "
                f"State: {guidance.state.value:12} | "
                f"Depth: {guidance.depth_level:.1f} | "
                f"Entrain: {progress*100:.0f}%"
            )
            print(status, end="", flush=True)

            # Check for keypress (simulate exhale)
            # On Linux, use select to check stdin
            if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                # Simulate exhale phase
                field.update_human_phase(0.625)
                last_keypress = time.time()
                print("\n  [YOU] exhale")
            else:
                # Drift human phase based on time since last keypress
                elapsed = time.time() - last_keypress
                if elapsed < 10:  # Assume ~4s breath cycle for simulation
                    simulated_phase = (elapsed % 4.0) / 4.0
                    field.update_human_phase(simulated_phase)

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        field.stop()

        print(f"\nFinal coherence: {field.get_coherence_score():.2f}")
        print(f"Final state: {field.current_state.value}")
        if field.moment_history:
            avg = sum(m.alignment for m in field.moment_history) / len(field.moment_history)
            print(f"Session average alignment: {avg:.2f}")
