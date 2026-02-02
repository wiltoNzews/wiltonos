"""
Breath Detection via Microphone
================================
Detects breathing patterns from audio input.

Detection methods:
1. Low-frequency energy (breath sounds are low freq, 50-500Hz)
2. Amplitude envelope (breath creates periodic amplitude changes)
3. Spectral flux (breath has different spectral signature than speech)

Output:
- Breath intervals (inhale-to-inhale time)
- Breath depth estimate (amplitude)
- Breath phase (inhale/exhale/hold)
- Coherence with target rhythm (3.12s or Fibonacci)
"""

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, List
from enum import Enum

try:
    import sounddevice as sd
except ImportError:
    sd = None


class BreathPhase(Enum):
    INHALE = "inhale"
    EXHALE = "exhale"
    HOLD = "hold"
    UNKNOWN = "unknown"


@dataclass
class BreathMetrics:
    """Current breath metrics"""
    phase: BreathPhase = BreathPhase.UNKNOWN
    last_interval: float = 0.0          # Last breath interval (seconds)
    mean_interval: float = 0.0          # Mean breath interval
    interval_variance: float = 0.0      # Variance in breath timing
    depth: float = 0.0                  # Relative breath depth (0-1)
    rhythm_coherence: float = 0.0       # Match to target rhythm (0-1)
    breath_rate: float = 0.0            # Breaths per minute
    target_match: float = 0.0           # Match to CENTER (3.12s) or SPIRAL
    signal_quality: float = 0.0         # Audio signal quality (0-1)


class BreathMic:
    """
    Real-time breath detection via microphone.

    Uses low-frequency energy detection and amplitude envelope
    to identify breath cycles.

    Usage:
        breath = BreathMic(device='RØDECaster')
        breath.start()

        # Later:
        metrics = breath.get_metrics()
        print(f"Breath rate: {metrics.breath_rate:.1f} BPM")

        breath.stop()
    """

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: Optional[int] = None,  # Auto-detect from device
        block_size: int = 2048,
        target_interval: float = 3.12,  # CENTER mode default
        on_metrics_update: Optional[Callable[[BreathMetrics], None]] = None,
        on_breath_detected: Optional[Callable[[BreathPhase, float], None]] = None
    ):
        self.device = device
        self.block_size = block_size

        # Find device first to get sample rate
        self.device_id = self._find_device(device)

        # Use device's default sample rate if not specified
        if sample_rate is None and sd is not None and self.device_id is not None:
            dev_info = sd.query_devices(self.device_id)
            self.sample_rate = int(dev_info['default_samplerate'])
        else:
            self.sample_rate = sample_rate or 48000
        self.target_interval = target_interval
        self.on_metrics_update = on_metrics_update
        self.on_breath_detected = on_breath_detected

        # Audio analysis
        self.low_freq_energy: deque = deque(maxlen=100)  # ~6 seconds of data
        self.amplitude_envelope: deque = deque(maxlen=100)
        self.breath_peaks: List[float] = []  # Timestamps of detected breaths
        self.last_breath_time: float = 0
        self.intervals: deque = deque(maxlen=20)

        # State
        self.stream = None
        self.running = False
        self.last_metrics = BreathMetrics()
        self.lock = threading.Lock()

        # Detection parameters
        self.breath_freq_low = 50    # Hz
        self.breath_freq_high = 500  # Hz
        self.min_breath_interval = 1.0   # seconds
        self.max_breath_interval = 15.0  # seconds

    def _find_device(self, name: Optional[str]) -> Optional[int]:
        """Find audio device by name"""
        if sd is None:
            return None

        if name is None:
            return None  # Use default

        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if name.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                return i
        return None

    def _bandpass_filter(self, data: np.ndarray, low: float, high: float) -> np.ndarray:
        """Simple bandpass using FFT"""
        fft = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(len(data), 1/self.sample_rate)

        # Zero out frequencies outside band
        fft[(freqs < low) | (freqs > high)] = 0
        return np.fft.irfft(fft, len(data))

    def _audio_callback(self, indata, frames, time_info, status):
        """Process audio block"""
        if status:
            return

        # Convert to mono if stereo
        audio = indata[:, 0] if len(indata.shape) > 1 else indata.flatten()

        # Calculate metrics
        now = time.time()

        # Low-frequency energy (breath band)
        filtered = self._bandpass_filter(audio, self.breath_freq_low, self.breath_freq_high)
        lf_energy = np.sqrt(np.mean(filtered ** 2))

        # Amplitude envelope
        amplitude = np.sqrt(np.mean(audio ** 2))

        with self.lock:
            self.low_freq_energy.append(lf_energy)
            self.amplitude_envelope.append(amplitude)

            # Breath detection via peak finding in low-freq energy
            if len(self.low_freq_energy) >= 10:
                self._detect_breath(now)

    def _detect_breath(self, now: float):
        """Detect breath cycle from accumulated data"""
        energies = np.array(self.low_freq_energy)

        # Normalize
        if energies.max() > 0:
            energies = energies / energies.max()

        # Simple peak detection: current > recent mean + threshold
        recent_mean = np.mean(energies[-20:]) if len(energies) >= 20 else np.mean(energies)
        current = energies[-1]
        threshold = 0.3

        # Detect peak (potential inhale/exhale transition)
        if current > recent_mean + threshold:
            # Check minimum interval
            if now - self.last_breath_time >= self.min_breath_interval:
                interval = now - self.last_breath_time if self.last_breath_time > 0 else 0
                self.last_breath_time = now

                if interval > 0 and interval < self.max_breath_interval:
                    self.intervals.append(interval)
                    self.breath_peaks.append(now)

                    # Determine phase (simplified: alternating)
                    phase = BreathPhase.INHALE if len(self.breath_peaks) % 2 == 0 else BreathPhase.EXHALE

                    if self.on_breath_detected:
                        self.on_breath_detected(phase, interval)

        # Update metrics
        self._update_metrics()

    def _update_metrics(self):
        """Recalculate breath metrics"""
        if len(self.intervals) < 2:
            return

        intervals = list(self.intervals)
        mean_interval = np.mean(intervals)
        variance = np.var(intervals)

        # Rhythm coherence (inverse of coefficient of variation)
        cv = np.sqrt(variance) / mean_interval if mean_interval > 0 else 1
        rhythm_coherence = max(0, min(1, 1 - cv))

        # Target match (how close to target interval)
        target_diff = abs(mean_interval - self.target_interval)
        target_match = max(0, 1 - target_diff / self.target_interval)

        # Breath rate (BPM)
        breath_rate = 60 / mean_interval if mean_interval > 0 else 0

        # Depth estimate from amplitude
        depths = list(self.amplitude_envelope)
        depth = np.mean(depths) / max(depths) if depths and max(depths) > 0 else 0

        # Signal quality
        signal_quality = min(1, len(self.intervals) / 10) * rhythm_coherence

        # Determine current phase (simplified)
        now = time.time()
        time_since_last = now - self.last_breath_time
        cycle_position = (time_since_last % mean_interval) / mean_interval if mean_interval > 0 else 0

        if cycle_position < 0.4:
            phase = BreathPhase.INHALE
        elif cycle_position < 0.6:
            phase = BreathPhase.HOLD
        else:
            phase = BreathPhase.EXHALE

        self.last_metrics = BreathMetrics(
            phase=phase,
            last_interval=intervals[-1] if intervals else 0,
            mean_interval=mean_interval,
            interval_variance=variance,
            depth=depth,
            rhythm_coherence=rhythm_coherence,
            breath_rate=breath_rate,
            target_match=target_match,
            signal_quality=signal_quality
        )

        if self.on_metrics_update:
            self.on_metrics_update(self.last_metrics)

    def set_target_interval(self, interval: float):
        """Set target breath interval (for CENTER or SPIRAL mode)"""
        self.target_interval = interval

    def set_fibonacci_target(self, index: int):
        """Set target to Fibonacci sequence position (SPIRAL mode)"""
        fib = [1, 1, 2, 3, 5, 8, 13]
        if 0 <= index < len(fib):
            self.target_interval = fib[index] * 0.5  # 0.5s multiplier

    def start(self):
        """Start breath detection"""
        if sd is None:
            raise RuntimeError("sounddevice not available")

        if self.running:
            return

        self.running = True
        self.stream = sd.InputStream(
            device=self.device_id,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._audio_callback
        )
        self.stream.start()

    def stop(self):
        """Stop breath detection"""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_metrics(self) -> BreathMetrics:
        """Get current metrics"""
        with self.lock:
            return self.last_metrics

    def get_coherence_signal(self) -> float:
        """Get coherence signal (0-1) from breath patterns"""
        m = self.last_metrics
        # Combine rhythm coherence and target match
        return (m.rhythm_coherence * 0.6 + m.target_match * 0.4) * m.signal_quality

    @staticmethod
    def list_devices():
        """List available audio input devices"""
        if sd is None:
            return []
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                inputs.append({
                    'id': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': dev['default_samplerate']
                })
        return inputs


# CLI test
if __name__ == "__main__":
    print("Breath Detection via Microphone")
    print("=" * 40)

    # List devices
    print("\nAvailable input devices:")
    for dev in BreathMic.list_devices():
        print(f"  [{dev['id']}] {dev['name']} ({dev['channels']}ch)")

    print("\nStarting breath detection...")
    print("Breathe normally. Press Ctrl+C to stop.\n")

    def on_update(metrics):
        phase_symbol = {
            BreathPhase.INHALE: "↑",
            BreathPhase.EXHALE: "↓",
            BreathPhase.HOLD: "─",
            BreathPhase.UNKNOWN: "?"
        }
        print(f"\r{phase_symbol[metrics.phase]} | "
              f"Interval: {metrics.mean_interval:.2f}s | "
              f"Rate: {metrics.breath_rate:.1f} BPM | "
              f"Rhythm: {metrics.rhythm_coherence:.2f} | "
              f"Target: {metrics.target_match:.2f} | "
              f"Quality: {metrics.signal_quality:.2f}", end="", flush=True)

    def on_breath(phase, interval):
        print(f"\n  [BREATH] {phase.value} - interval: {interval:.2f}s")

    # Try to find RØDECaster
    breath = BreathMic(
        device="RØDE",
        target_interval=3.12,  # CENTER mode
        on_metrics_update=on_update,
        on_breath_detected=on_breath
    )

    breath.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\nStopping...")
        breath.stop()

        final = breath.get_metrics()
        print(f"\nFinal metrics:")
        print(f"  Mean interval: {final.mean_interval:.2f}s")
        print(f"  Breath rate: {final.breath_rate:.1f} BPM")
        print(f"  Rhythm coherence: {final.rhythm_coherence:.2f}")
        print(f"  Target match (3.12s): {final.target_match:.2f}")
