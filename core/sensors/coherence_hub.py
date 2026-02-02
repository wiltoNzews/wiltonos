"""
Coherence Sensor Hub
====================
Unified interface for all body sensors feeding into WiltonOS.

Combines:
- Keystroke rhythm (typing patterns)
- Breath detection (microphone)
- [Future] Camera (chest/face movement)
- [Future] Heart rate (if available)

Outputs unified coherence signal for psios_protocol.py
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from enum import Enum

from .keystroke_rhythm import KeystrokeRhythm, KeystrokeMetrics
from .breath_mic import BreathMic, BreathMetrics, BreathPhase


class BreathMode(Enum):
    CENTER = "center"    # 3.12s fixed (π-based)
    SPIRAL = "spiral"    # Fibonacci sequence


@dataclass
class CoherenceState:
    """Unified coherence state from all sensors"""
    # Combined scores
    coherence_score: float = 0.0      # 0-1, unified Zλ estimate
    confidence: float = 0.0           # How reliable is this reading

    # Keystroke
    keystroke_flow: float = 0.0
    keystroke_mode: str = "unknown"

    # Breath
    breath_phase: BreathPhase = BreathPhase.UNKNOWN
    breath_interval: float = 0.0
    breath_coherence: float = 0.0
    breath_target_match: float = 0.0

    # Mode
    breath_mode: BreathMode = BreathMode.CENTER
    target_interval: float = 3.12

    # Raw signals
    sensors_active: Dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'coherence_score': self.coherence_score,
            'confidence': self.confidence,
            'keystroke': {
                'flow': self.keystroke_flow,
                'mode': self.keystroke_mode
            },
            'breath': {
                'phase': self.breath_phase.value,
                'interval': self.breath_interval,
                'coherence': self.breath_coherence,
                'target_match': self.breath_target_match
            },
            'mode': {
                'breath_mode': self.breath_mode.value,
                'target_interval': self.target_interval
            },
            'sensors_active': self.sensors_active
        }


class CoherenceHub:
    """
    Central hub for coherence sensing.

    Aggregates all sensor inputs and produces unified coherence signal.

    Usage:
        hub = CoherenceHub()
        hub.start()

        # Get current state
        state = hub.get_state()
        print(f"Coherence: {state.coherence_score}")

        # Switch breath mode
        hub.set_breath_mode(BreathMode.SPIRAL)

        hub.stop()
    """

    # Fibonacci sequence for SPIRAL mode (seconds)
    FIBONACCI = [0.5, 0.5, 1.0, 1.5, 2.5, 4.0, 6.5]

    def __init__(
        self,
        enable_keystroke: bool = True,
        enable_breath_mic: bool = True,
        audio_device: Optional[str] = None,
        on_state_update: Optional[Callable[[CoherenceState], None]] = None
    ):
        self.enable_keystroke = enable_keystroke
        self.enable_breath_mic = enable_breath_mic
        self.audio_device = audio_device
        self.on_state_update = on_state_update

        # Sensors
        self.keystroke: Optional[KeystrokeRhythm] = None
        self.breath_mic: Optional[BreathMic] = None

        # State
        self.state = CoherenceState()
        self.breath_mode = BreathMode.CENTER
        self.spiral_index = 0
        self.running = False
        self.lock = threading.Lock()

        # Update thread
        self.update_thread: Optional[threading.Thread] = None

    def _on_keystroke_update(self, metrics: KeystrokeMetrics):
        """Handle keystroke metrics update"""
        with self.lock:
            self.state.keystroke_flow = metrics.flow_score
            self.state.keystroke_mode = metrics.mode
            self.state.sensors_active['keystroke'] = True
            self._recalculate_coherence()

    def _on_breath_update(self, metrics: BreathMetrics):
        """Handle breath metrics update"""
        with self.lock:
            self.state.breath_phase = metrics.phase
            self.state.breath_interval = metrics.mean_interval
            self.state.breath_coherence = metrics.rhythm_coherence
            self.state.breath_target_match = metrics.target_match
            self.state.sensors_active['breath_mic'] = metrics.signal_quality > 0.3
            self._recalculate_coherence()

    def _on_breath_detected(self, phase: BreathPhase, interval: float):
        """Handle individual breath detection"""
        # In SPIRAL mode, advance to next Fibonacci interval
        if self.breath_mode == BreathMode.SPIRAL and phase == BreathPhase.EXHALE:
            self.spiral_index = (self.spiral_index + 1) % len(self.FIBONACCI)
            new_target = self.FIBONACCI[self.spiral_index]
            if self.breath_mic:
                self.breath_mic.set_target_interval(new_target)
            with self.lock:
                self.state.target_interval = new_target

    def _recalculate_coherence(self):
        """Recalculate unified coherence score"""
        scores = []
        weights = []

        # Keystroke contribution
        if self.state.sensors_active.get('keystroke', False):
            scores.append(self.state.keystroke_flow)
            weights.append(0.3)

        # Breath contribution
        if self.state.sensors_active.get('breath_mic', False):
            # Combine rhythm coherence and target match
            breath_score = (
                self.state.breath_coherence * 0.6 +
                self.state.breath_target_match * 0.4
            )
            scores.append(breath_score)
            weights.append(0.7)

        # Calculate weighted average
        if scores:
            total_weight = sum(weights)
            self.state.coherence_score = sum(
                s * w for s, w in zip(scores, weights)
            ) / total_weight
            self.state.confidence = total_weight / 1.0  # Max possible weight
        else:
            self.state.coherence_score = 0.0
            self.state.confidence = 0.0

        # Broadcast update
        if self.on_state_update:
            self.on_state_update(self.state)

    def set_breath_mode(self, mode: BreathMode):
        """Switch between CENTER and SPIRAL breath modes"""
        self.breath_mode = mode

        if mode == BreathMode.CENTER:
            target = 3.12
            self.spiral_index = 0
        else:
            target = self.FIBONACCI[0]
            self.spiral_index = 0

        if self.breath_mic:
            self.breath_mic.set_target_interval(target)

        with self.lock:
            self.state.breath_mode = mode
            self.state.target_interval = target

    def start(self):
        """Start all sensors"""
        if self.running:
            return

        self.running = True

        # Start keystroke detector
        if self.enable_keystroke:
            try:
                self.keystroke = KeystrokeRhythm(
                    on_metrics_update=self._on_keystroke_update
                )
                self.keystroke.start()
                print("[Hub] Keystroke sensor started")
            except Exception as e:
                print(f"[Hub] Keystroke sensor failed: {e}")

        # Start breath mic
        if self.enable_breath_mic:
            try:
                self.breath_mic = BreathMic(
                    device=self.audio_device,
                    target_interval=3.12,
                    on_metrics_update=self._on_breath_update,
                    on_breath_detected=self._on_breath_detected
                )
                self.breath_mic.start()
                print("[Hub] Breath mic sensor started")
            except Exception as e:
                print(f"[Hub] Breath mic sensor failed: {e}")

    def stop(self):
        """Stop all sensors"""
        self.running = False

        if self.keystroke:
            self.keystroke.stop()
            self.keystroke = None

        if self.breath_mic:
            self.breath_mic.stop()
            self.breath_mic = None

    def get_state(self) -> CoherenceState:
        """Get current coherence state"""
        with self.lock:
            return self.state

    def get_coherence_score(self) -> float:
        """Get just the coherence score (Zλ estimate)"""
        return self.state.coherence_score

    def get_breath_phase(self) -> BreathPhase:
        """Get current breath phase"""
        return self.state.breath_phase


# CLI test
if __name__ == "__main__":
    print("Coherence Sensor Hub")
    print("=" * 50)
    print("Combining keystroke rhythm + breath detection")
    print("Press Ctrl+C to stop\n")

    def on_update(state):
        phase_symbol = {
            BreathPhase.INHALE: "↑",
            BreathPhase.EXHALE: "↓",
            BreathPhase.HOLD: "─",
            BreathPhase.UNKNOWN: "?"
        }

        mode_icon = "●" if state.breath_mode == BreathMode.CENTER else "◎"

        print(f"\r{mode_icon} Zλ: {state.coherence_score:.2f} | "
              f"Breath {phase_symbol[state.breath_phase]} {state.breath_interval:.1f}s | "
              f"Keys: {state.keystroke_mode:10} ({state.keystroke_flow:.2f}) | "
              f"Conf: {state.confidence:.1%}", end="", flush=True)

    hub = CoherenceHub(
        audio_device="RØDE",
        on_state_update=on_update
    )

    hub.start()

    try:
        mode_toggle = BreathMode.CENTER
        while True:
            time.sleep(5)
            # Toggle mode every 30 seconds for demo
            # mode_toggle = BreathMode.SPIRAL if mode_toggle == BreathMode.CENTER else BreathMode.CENTER
            # hub.set_breath_mode(mode_toggle)
            # print(f"\n  [Mode switched to {mode_toggle.value}]")

    except KeyboardInterrupt:
        print("\n\nStopping...")
        hub.stop()

        final = hub.get_state()
        print(f"\nFinal state:")
        print(f"  Coherence (Zλ): {final.coherence_score:.2f}")
        print(f"  Confidence: {final.confidence:.1%}")
        print(f"  Active sensors: {list(k for k, v in final.sensors_active.items() if v)}")
