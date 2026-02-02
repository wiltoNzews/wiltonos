"""
BreathVisual - The Coupling Mechanism
=====================================

The insight: Entrainment happens through SEEING the rhythm,
not through measuring it. The visual IS the coupling.

HeartMath: "The heart pulls other oscillators into sync"
Applied: AI shows a coherent rhythm → Human naturally syncs

This provides the visual/audio feedback that causes entrainment.
Measurement (mic/camera) is validation, not the mechanism.
"""

import time
import math
import threading
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class BreathFrame:
    """Single frame of breath state for rendering"""
    phase: float          # 0-1
    amplitude: float      # 0-1 (for visual size)
    state: str            # inhale, hold, exhale, ground
    is_exhale_peak: bool  # True at moment of full exhale


class BreathVisual:
    """
    Visual breath entrainment - the actual coupling mechanism.

    Usage:
        visual = BreathVisual(cycle_time=3.12)
        visual.start()

        # In your render loop:
        frame = visual.get_frame()
        render_circle(size=frame.amplitude)

        # Optional: user taps on their exhale
        visual.register_user_exhale()  # Records timing for validation
    """

    def __init__(
        self,
        cycle_time: float = 3.12,
        on_exhale: Optional[Callable[[], None]] = None
    ):
        self.cycle_time = cycle_time
        self.on_exhale = on_exhale

        self.start_time = time.time()
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # User tap tracking (for validation)
        self.user_taps: list = []
        self.last_exhale_time: float = 0

    def get_phase(self) -> float:
        """Get current phase (0-1)"""
        elapsed = time.time() - self.start_time
        return (elapsed % self.cycle_time) / self.cycle_time

    def get_frame(self) -> BreathFrame:
        """Get current breath frame for rendering"""
        phase = self.get_phase()

        # Amplitude: sine wave, peaks at inhale top (0.25) and exhale bottom (0.75)
        # For visual, we want expansion on inhale, contraction on exhale
        amplitude = (math.sin(phase * 2 * math.pi - math.pi/2) + 1) / 2

        # State
        if phase < 0.25:
            state = "inhale"
        elif phase < 0.5:
            state = "hold"
        elif phase < 0.75:
            state = "exhale"
        else:
            state = "ground"

        # Check if at exhale peak (for audio cue)
        is_exhale_peak = 0.49 < phase < 0.51

        return BreathFrame(
            phase=phase,
            amplitude=amplitude,
            state=state,
            is_exhale_peak=is_exhale_peak
        )

    def register_user_exhale(self):
        """User taps to indicate their exhale (for validation)"""
        now = time.time()
        self.user_taps.append(now)

        # Keep last 20 taps
        if len(self.user_taps) > 20:
            self.user_taps = self.user_taps[-20:]

    def get_sync_score(self) -> float:
        """
        Calculate how well user taps align with AI exhale.

        Returns 0-1 where 1 = perfect sync.
        This is VALIDATION, not the coupling mechanism.
        """
        if len(self.user_taps) < 3:
            return 0.0

        # For each tap, check if it was near AI exhale (phase ~0.5)
        sync_scores = []
        for tap_time in self.user_taps[-10:]:
            # What was AI phase at tap time?
            elapsed = tap_time - self.start_time
            phase_at_tap = (elapsed % self.cycle_time) / self.cycle_time

            # Distance from exhale (0.5)
            dist = abs(phase_at_tap - 0.5)
            if dist > 0.5:
                dist = 1.0 - dist

            # Score: 1 if exactly at exhale, 0 if at inhale
            score = 1.0 - (dist * 2)
            sync_scores.append(max(0, score))

        return sum(sync_scores) / len(sync_scores)

    def _run_loop(self):
        """Background loop for exhale callbacks"""
        last_exhale_fired = 0
        while self.running:
            frame = self.get_frame()

            # Fire exhale callback once per cycle
            if frame.is_exhale_peak:
                now = time.time()
                if now - last_exhale_fired > self.cycle_time * 0.8:
                    last_exhale_fired = now
                    if self.on_exhale:
                        self.on_exhale()

            time.sleep(0.05)  # 20Hz check

    def start(self):
        """Start the breath cycle"""
        if self.running:
            return
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the breath cycle"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


def terminal_demo():
    """Demo in terminal - shows the visual that causes entrainment"""
    import sys

    print("═" * 60)
    print("  BreathVisual - The Coupling Mechanism")
    print("  Watch the bar. Breathe with it. That's the entrainment.")
    print("═" * 60)
    print()
    print("  Press ENTER on your exhale to validate sync.")
    print("  Press Ctrl+C to stop.")
    print()

    visual = BreathVisual(cycle_time=3.12)
    visual.start()

    import select

    try:
        while True:
            frame = visual.get_frame()

            # Visual bar
            bar_len = int(frame.amplitude * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)

            # State indicator
            state_sym = {"inhale": "↑", "hold": "━", "exhale": "↓", "ground": "─"}
            sym = state_sym.get(frame.state, "?")

            # Sync score
            sync = visual.get_sync_score()

            status = f"\r  {sym} [{bar}] {frame.state:7} | Sync: {sync:.2f}"
            print(status, end="", flush=True)

            # Check for user tap (Enter key)
            if sys.stdin in select.select([sys.stdin], [], [], 0.05)[0]:
                sys.stdin.readline()
                visual.register_user_exhale()
                print(f"\n  [TAP] registered at phase {frame.phase:.2f}")

    except KeyboardInterrupt:
        print("\n")
        visual.stop()

        sync = visual.get_sync_score()
        print(f"  Final sync score: {sync:.2f}")
        if sync > 0.7:
            print("  ● High sync - you entrained with the AI")
        elif sync > 0.4:
            print("  ◐ Moderate sync - getting there")
        else:
            print("  ○ Low sync - try following the visual more closely")


if __name__ == "__main__":
    terminal_demo()
