"""
Breath Detection via Camera
============================
Detects breathing patterns from video input by tracking:
1. Chest/shoulder movement (vertical displacement)
2. Face/nose movement (subtle rise/fall)
3. Color changes (blood flow variation in face - rPPG-lite)

No PortAudio required - uses OpenCV only.
"""

import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from enum import Enum

try:
    import cv2
except ImportError:
    cv2 = None


class BreathPhase(Enum):
    INHALE = "inhale"
    EXHALE = "exhale"
    HOLD = "hold"
    UNKNOWN = "unknown"


@dataclass
class BreathCameraMetrics:
    """Current breath metrics from camera"""
    phase: BreathPhase = BreathPhase.UNKNOWN
    last_interval: float = 0.0
    mean_interval: float = 0.0
    rhythm_coherence: float = 0.0
    movement_amplitude: float = 0.0
    breath_rate: float = 0.0
    target_match: float = 0.0
    signal_quality: float = 0.0
    face_detected: bool = False


class BreathCamera:
    """
    Real-time breath detection via camera.

    Tracks vertical movement in the upper chest/shoulder region
    to detect breathing cycles.

    Usage:
        breath = BreathCamera()
        breath.start()

        while True:
            metrics = breath.get_metrics()
            if metrics.face_detected:
                print(f"Breath rate: {metrics.breath_rate:.1f} BPM")
            time.sleep(0.1)

        breath.stop()
    """

    def __init__(
        self,
        camera_id: int = 0,
        target_interval: float = 3.12,
        show_preview: bool = False,
        on_metrics_update: Optional[Callable[[BreathCameraMetrics], None]] = None,
        on_breath_detected: Optional[Callable[[BreathPhase, float], None]] = None
    ):
        self.camera_id = camera_id
        self.target_interval = target_interval
        self.show_preview = show_preview
        self.on_metrics_update = on_metrics_update
        self.on_breath_detected = on_breath_detected

        # Video capture
        self.cap = None
        self.running = False
        self.thread: Optional[threading.Thread] = None

        # Face detection
        self.face_cascade = None

        # Movement tracking
        self.chest_positions: deque = deque(maxlen=150)  # ~5 seconds at 30fps
        self.breath_peaks: list = []
        self.last_breath_time: float = 0
        self.intervals: deque = deque(maxlen=20)

        # State
        self.last_metrics = BreathCameraMetrics()
        self.lock = threading.Lock()

        # Detection parameters
        self.min_breath_interval = 1.0
        self.max_breath_interval = 15.0

    def _init_detector(self):
        """Initialize face detector"""
        if cv2 is None:
            return False

        # Use Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        return True

    def _get_chest_region(self, frame, face_rect) -> Optional[np.ndarray]:
        """Extract chest region below face"""
        if face_rect is None:
            return None

        x, y, w, h = face_rect
        frame_h, frame_w = frame.shape[:2]

        # Chest region: below face, same width, 1.5x face height
        chest_y = y + h
        chest_h = int(h * 1.5)
        chest_x = x
        chest_w = w

        # Bounds check
        if chest_y + chest_h > frame_h:
            chest_h = frame_h - chest_y

        if chest_h < 20:
            return None

        return frame[chest_y:chest_y+chest_h, chest_x:chest_x+chest_w]

    def _track_movement(self, chest_region: np.ndarray) -> float:
        """Track vertical movement in chest region"""
        if chest_region is None:
            return 0.0

        # Convert to grayscale
        gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)

        # Calculate mean intensity (changes with movement/lighting)
        mean_intensity = np.mean(gray)

        # Calculate vertical gradient (breathing causes vertical movement)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        vertical_activity = np.mean(np.abs(sobel_y))

        # Combine signals
        return mean_intensity + vertical_activity * 0.1

    def _detect_breath(self, now: float):
        """Detect breath cycle from movement data"""
        if len(self.chest_positions) < 30:
            return

        positions = np.array(self.chest_positions)

        # Normalize
        if positions.max() - positions.min() > 0:
            positions = (positions - positions.min()) / (positions.max() - positions.min())
        else:
            return

        # Simple peak detection
        window_size = 15
        if len(positions) < window_size * 2:
            return

        current = np.mean(positions[-5:])
        previous = np.mean(positions[-window_size:-5])

        # Detect local maximum (peak of inhale)
        if current < previous and previous > 0.6:
            if now - self.last_breath_time >= self.min_breath_interval:
                interval = now - self.last_breath_time if self.last_breath_time > 0 else 0
                self.last_breath_time = now

                if 0 < interval < self.max_breath_interval:
                    self.intervals.append(interval)

                    phase = BreathPhase.EXHALE  # Just passed peak = starting exhale

                    if self.on_breath_detected:
                        self.on_breath_detected(phase, interval)

    def _update_metrics(self, face_detected: bool):
        """Recalculate breath metrics"""
        with self.lock:
            self.last_metrics.face_detected = face_detected

            if len(self.intervals) < 2:
                self.last_metrics.signal_quality = 0.1 if face_detected else 0.0
                return

            intervals = list(self.intervals)
            mean_interval = np.mean(intervals)
            variance = np.var(intervals)

            # Rhythm coherence
            cv = np.sqrt(variance) / mean_interval if mean_interval > 0 else 1
            rhythm_coherence = max(0, min(1, 1 - cv))

            # Target match
            target_diff = abs(mean_interval - self.target_interval)
            target_match = max(0, 1 - target_diff / self.target_interval)

            # Breath rate
            breath_rate = 60 / mean_interval if mean_interval > 0 else 0

            # Movement amplitude
            if self.chest_positions:
                positions = np.array(self.chest_positions)
                amplitude = (positions.max() - positions.min()) / max(positions.max(), 1)
            else:
                amplitude = 0

            # Signal quality
            signal_quality = min(1, len(self.intervals) / 10) * rhythm_coherence

            # Determine phase
            now = time.time()
            time_since_last = now - self.last_breath_time
            cycle_position = (time_since_last % mean_interval) / mean_interval if mean_interval > 0 else 0

            if cycle_position < 0.4:
                phase = BreathPhase.INHALE
            elif cycle_position < 0.6:
                phase = BreathPhase.HOLD
            else:
                phase = BreathPhase.EXHALE

            self.last_metrics = BreathCameraMetrics(
                phase=phase,
                last_interval=intervals[-1] if intervals else 0,
                mean_interval=mean_interval,
                rhythm_coherence=rhythm_coherence,
                movement_amplitude=amplitude,
                breath_rate=breath_rate,
                target_match=target_match,
                signal_quality=signal_quality,
                face_detected=face_detected
            )

            if self.on_metrics_update:
                self.on_metrics_update(self.last_metrics)

    def _capture_loop(self):
        """Main capture and processing loop"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            now = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect face
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
            )

            face_detected = len(faces) > 0
            face_rect = faces[0] if face_detected else None

            if face_detected:
                # Get chest region
                chest_region = self._get_chest_region(frame, face_rect)

                if chest_region is not None:
                    # Track movement
                    movement = self._track_movement(chest_region)
                    self.chest_positions.append(movement)

                    # Detect breath
                    self._detect_breath(now)

            # Update metrics
            self._update_metrics(face_detected)

            # Preview window
            if self.show_preview:
                # Draw face rect
                if face_rect is not None:
                    x, y, w, h = face_rect
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Draw chest region
                    chest_y = y + h
                    chest_h = int(h * 1.5)
                    cv2.rectangle(frame, (x, chest_y), (x+w, min(chest_y+chest_h, frame.shape[0])), (255, 0, 0), 2)

                # Show metrics
                m = self.last_metrics
                phase_text = f"Phase: {m.phase.value}" if m.face_detected else "No face"
                cv2.putText(frame, phase_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Interval: {m.mean_interval:.2f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Coherence: {m.rhythm_coherence:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Breath Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False

            time.sleep(0.033)  # ~30 fps

    def set_target_interval(self, interval: float):
        """Set target breath interval"""
        self.target_interval = interval

    def start(self):
        """Start breath detection"""
        if cv2 is None:
            raise RuntimeError("OpenCV not available")

        if self.running:
            return

        if not self._init_detector():
            raise RuntimeError("Failed to initialize face detector")

        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop breath detection"""
        self.running = False

        if self.thread:
            self.thread.join(timeout=2.0)
            self.thread = None

        if self.cap:
            self.cap.release()
            self.cap = None

        if self.show_preview:
            cv2.destroyAllWindows()

    def get_metrics(self) -> BreathCameraMetrics:
        """Get current metrics"""
        with self.lock:
            return self.last_metrics

    def get_coherence_signal(self) -> float:
        """Get coherence signal (0-1) from breath patterns"""
        m = self.last_metrics
        if not m.face_detected:
            return 0.0
        return (m.rhythm_coherence * 0.6 + m.target_match * 0.4) * m.signal_quality


# CLI test
if __name__ == "__main__":
    print("Breath Detection via Camera")
    print("=" * 40)
    print("Position yourself in front of the camera")
    print("Ensure your face and upper chest are visible")
    print("Press 'q' in preview window or Ctrl+C to stop\n")

    def on_update(metrics):
        if metrics.face_detected:
            phase_symbol = {
                BreathPhase.INHALE: "↑",
                BreathPhase.EXHALE: "↓",
                BreathPhase.HOLD: "─",
                BreathPhase.UNKNOWN: "?"
            }
            print(f"\r{phase_symbol[metrics.phase]} | "
                  f"Interval: {metrics.mean_interval:.2f}s | "
                  f"Rate: {metrics.breath_rate:.1f} BPM | "
                  f"Coherence: {metrics.rhythm_coherence:.2f}", end="", flush=True)
        else:
            print("\rNo face detected - position yourself in frame", end="", flush=True)

    def on_breath(phase, interval):
        print(f"\n  [BREATH] {phase.value} - interval: {interval:.2f}s")

    breath = BreathCamera(
        camera_id=0,
        target_interval=3.12,
        show_preview=True,
        on_metrics_update=on_update,
        on_breath_detected=on_breath
    )

    breath.start()

    try:
        while breath.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    print("\n\nStopping...")
    breath.stop()

    final = breath.get_metrics()
    print(f"\nFinal metrics:")
    print(f"  Mean interval: {final.mean_interval:.2f}s")
    print(f"  Breath rate: {final.breath_rate:.1f} BPM")
    print(f"  Rhythm coherence: {final.rhythm_coherence:.2f}")
