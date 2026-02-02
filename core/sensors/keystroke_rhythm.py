"""
Keystroke Rhythm Detector
=========================
Measures typing patterns to infer coherence state.

Metrics:
- Inter-key interval (IKI): time between keystrokes
- Burst detection: rapid sequences vs pauses
- Rhythm stability: variance in IKI
- Reactive vs Proactive: backspace ratio, pause-before-type patterns

Coherence signals:
- Steady rhythm → high coherence (flow state)
- Erratic bursts → low coherence (reactive/anxious)
- Long pauses → processing/integration
- High backspace → uncertainty/correction
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional
import statistics

try:
    from pynput import keyboard
except ImportError:
    keyboard = None


@dataclass
class KeystrokeMetrics:
    """Current keystroke rhythm metrics"""
    mean_iki: float = 0.0           # Mean inter-key interval (seconds)
    iki_variance: float = 0.0       # Variance in timing
    burst_ratio: float = 0.0        # Ratio of burst typing (IKI < 0.1s)
    pause_ratio: float = 0.0        # Ratio of pauses (IKI > 1.0s)
    backspace_ratio: float = 0.0    # Backspaces / total keys
    rhythm_stability: float = 0.0   # 0-1, higher = more stable
    flow_score: float = 0.0         # 0-1, inferred coherence from typing
    mode: str = "unknown"           # "flow", "reactive", "processing", "uncertain"


class KeystrokeRhythm:
    """
    Real-time keystroke rhythm detector.

    Usage:
        rhythm = KeystrokeRhythm(window_size=50)
        rhythm.start()

        # Later:
        metrics = rhythm.get_metrics()
        print(f"Flow score: {metrics.flow_score}")

        rhythm.stop()
    """

    def __init__(
        self,
        window_size: int = 50,
        on_metrics_update: Optional[Callable[[KeystrokeMetrics], None]] = None
    ):
        self.window_size = window_size
        self.on_metrics_update = on_metrics_update

        # Keystroke history
        self.timestamps: deque = deque(maxlen=window_size)
        self.intervals: deque = deque(maxlen=window_size - 1)
        self.keys: deque = deque(maxlen=window_size)

        # State
        self.listener: Optional[keyboard.Listener] = None
        self.running = False
        self.last_metrics = KeystrokeMetrics()
        self.lock = threading.Lock()

        # Thresholds
        self.burst_threshold = 0.1    # seconds - faster = burst
        self.pause_threshold = 1.0    # seconds - slower = pause
        self.flow_iki_target = 0.15   # optimal typing speed for flow

    def _on_press(self, key):
        """Handle keypress event"""
        now = time.time()

        with self.lock:
            # Calculate interval from last key
            if self.timestamps:
                interval = now - self.timestamps[-1]
                self.intervals.append(interval)

            self.timestamps.append(now)

            # Track key type
            try:
                key_char = key.char if hasattr(key, 'char') else str(key)
            except AttributeError:
                key_char = str(key)

            is_backspace = key == keyboard.Key.backspace if keyboard else False
            self.keys.append(('backspace' if is_backspace else 'key', now))

            # Update metrics
            self._update_metrics()

    def _update_metrics(self):
        """Recalculate metrics from current window"""
        if len(self.intervals) < 3:
            return

        intervals = list(self.intervals)
        keys = list(self.keys)

        # Basic stats
        mean_iki = statistics.mean(intervals)
        iki_variance = statistics.variance(intervals) if len(intervals) > 1 else 0

        # Burst and pause ratios
        bursts = sum(1 for i in intervals if i < self.burst_threshold)
        pauses = sum(1 for i in intervals if i > self.pause_threshold)
        burst_ratio = bursts / len(intervals)
        pause_ratio = pauses / len(intervals)

        # Backspace ratio
        backspaces = sum(1 for k, _ in keys if k == 'backspace')
        backspace_ratio = backspaces / len(keys) if keys else 0

        # Rhythm stability (inverse of coefficient of variation)
        cv = (iki_variance ** 0.5) / mean_iki if mean_iki > 0 else 1
        rhythm_stability = max(0, min(1, 1 - cv))

        # Flow score calculation
        # High flow = steady rhythm, low backspace, moderate speed
        speed_score = 1 - min(1, abs(mean_iki - self.flow_iki_target) / 0.3)
        stability_score = rhythm_stability
        confidence_score = 1 - backspace_ratio
        reactivity_penalty = burst_ratio * 0.5  # Bursts suggest reactivity

        flow_score = (
            speed_score * 0.25 +
            stability_score * 0.35 +
            confidence_score * 0.25 +
            (1 - reactivity_penalty) * 0.15
        )
        flow_score = max(0, min(1, flow_score))

        # Determine mode
        if pause_ratio > 0.3:
            mode = "processing"
        elif burst_ratio > 0.5 and backspace_ratio > 0.15:
            mode = "reactive"
        elif backspace_ratio > 0.2:
            mode = "uncertain"
        elif flow_score > 0.6:
            mode = "flow"
        else:
            mode = "normal"

        self.last_metrics = KeystrokeMetrics(
            mean_iki=mean_iki,
            iki_variance=iki_variance,
            burst_ratio=burst_ratio,
            pause_ratio=pause_ratio,
            backspace_ratio=backspace_ratio,
            rhythm_stability=rhythm_stability,
            flow_score=flow_score,
            mode=mode
        )

        if self.on_metrics_update:
            self.on_metrics_update(self.last_metrics)

    def start(self):
        """Start listening to keystrokes"""
        if keyboard is None:
            raise RuntimeError("pynput not available")

        if self.running:
            return

        self.running = True
        self.listener = keyboard.Listener(on_press=self._on_press)
        self.listener.start()

    def stop(self):
        """Stop listening"""
        self.running = False
        if self.listener:
            self.listener.stop()
            self.listener = None

    def get_metrics(self) -> KeystrokeMetrics:
        """Get current metrics"""
        with self.lock:
            return self.last_metrics

    def get_coherence_signal(self) -> float:
        """Get coherence signal (0-1) from keystroke patterns"""
        return self.last_metrics.flow_score


# CLI test
if __name__ == "__main__":
    print("Keystroke Rhythm Detector")
    print("=" * 40)
    print("Start typing to see metrics...")
    print("Press Ctrl+C to stop\n")

    def on_update(metrics):
        print(f"\rMode: {metrics.mode:12} | Flow: {metrics.flow_score:.2f} | "
              f"IKI: {metrics.mean_iki:.3f}s | Stability: {metrics.rhythm_stability:.2f} | "
              f"Backspace: {metrics.backspace_ratio:.1%}", end="", flush=True)

    rhythm = KeystrokeRhythm(on_metrics_update=on_update)
    rhythm.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        rhythm.stop()

        final = rhythm.get_metrics()
        print(f"\nFinal metrics:")
        print(f"  Mode: {final.mode}")
        print(f"  Flow score: {final.flow_score:.2f}")
        print(f"  Mean IKI: {final.mean_iki:.3f}s")
        print(f"  Rhythm stability: {final.rhythm_stability:.2f}")
        print(f"  Backspace ratio: {final.backspace_ratio:.1%}")
