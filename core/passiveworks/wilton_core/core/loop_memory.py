"""
Loop Memory Implementation
-------------------------

This module provides window-based memory tracking for recurrent execution patterns,
helping to predict coherence divergence before it happens.
"""

import time
import logging
import collections
from typing import Any, Dict, List, Optional, Tuple, Deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[QUANTUM_STATE: %(levelname)s_FLOW] %(message)s"
)
logger = logging.getLogger("wilton_core.loop_memory")


class LoopMemory:
    """
    LoopMemory implements a sliding window memory system for tracking
    execution patterns and predicting coherence shifts.

    It maintains multiple time-bound windows of state changes to detect:
    - Repetitive patterns (resonance spirals)
    - Divergence precursors
    - Execution path shifts
    """

    # Singleton instance
    _instance = None

    # Class constants
    DEFAULT_WINDOW_SIZE = 5  # Default size of memory windows
    MAX_WINDOWS = 3  # Number of concurrent windows (short/medium/long-term)
    PATTERN_THRESHOLD = 0.85  # Similarity threshold for pattern detection

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern"""
        if cls._instance is None:
            logger.info("Initializing LoopMemory singleton")
            cls._instance = super(LoopMemory, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(
        self, window_size: Optional[int] = None, max_windows: Optional[int] = None
    ):
        """Initialize the LoopMemory with configuration parameters"""
        # Memory windows of different time scales
        self.window_size = window_size or self.DEFAULT_WINDOW_SIZE
        self.max_windows = max_windows or self.MAX_WINDOWS

        # Initialize memory windows (short, medium, long-term)
        self.windows = {}
        for i in range(self.max_windows):
            # Scale increases with each window (5, 10, 20, etc.)
            scale = 2**i
            window_name = f"window_{scale}x"
            self.windows[window_name] = collections.deque(
                maxlen=self.window_size * scale
            )

        # Tracking variables
        self.last_update = time.time()
        self.pattern_cache = {}
        self.execution_paths = collections.deque(maxlen=100)
        self.divergence_predictions = []

        logger.info(f"LoopMemory initialized with {len(self.windows)} windows")

    def record_state(
        self,
        coherence: float,
        exploration: float,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Record a state point in all memory windows

        Args:
            coherence: Current coherence value [0-1]
            exploration: Current exploration value [0-1]
            execution_context: Optional dictionary with execution context

        Returns:
            Dict with recorded state and window statistics
        """
        timestamp = datetime.utcnow()

        # Create state record
        state = {
            "timestamp": timestamp.isoformat(),
            "coherence": coherence,
            "exploration": exploration,
            "context": execution_context or {},
            "ratios": {
                "c_to_e": coherence / max(0.001, exploration),  # Avoid division by zero
                "quantum_balance": coherence / 0.75,  # Normalized to target (0.75)
            },
        }

        # Add to all windows
        for window_name, window in self.windows.items():
            window.append(state)

        # Record execution path if context contains function info
        if execution_context and "function" in execution_context:
            self.execution_paths.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "function": execution_context["function"],
                    "coherence": coherence,
                }
            )

        # Check for patterns in the windows
        self._detect_patterns()

        # Predict potential divergence
        prediction = self._predict_divergence(coherence, exploration)
        if prediction and prediction["probability"] > 0.5:
            logger.warning(
                f"Potential coherence divergence predicted: {prediction['probability']:.2f} probability in {prediction['timeframe']} steps"
            )
            self.divergence_predictions.append(prediction)

        return {
            "state": state,
            "window_sizes": {
                name: len(window) for name, window in self.windows.items()
            },
            "predictions": (
                self.divergence_predictions[-3:] if self.divergence_predictions else []
            ),
        }

    def get_window(self, window_name: str = "window_1x") -> List[Dict[str, Any]]:
        """
        Get a specific memory window by name

        Args:
            window_name: Name of window to retrieve

        Returns:
            List of state records in the window
        """
        if window_name not in self.windows:
            logger.warning(f"Window {window_name} not found, using default window")
            window_name = "window_1x"

        return list(self.windows[window_name])

    def get_coherence_trend(self, window_name: str = "window_2x") -> Dict[str, Any]:
        """
        Calculate coherence trend statistics over a window

        Args:
            window_name: Name of window to analyze

        Returns:
            Dict with trend statistics
        """
        if window_name not in self.windows or not self.windows[window_name]:
            return {
                "trend": "unknown",
                "stability": 0.0,
                "direction": "none",
                "variance": 0.0,
            }

        window = self.windows[window_name]
        coherence_values = [state["coherence"] for state in window]

        if len(coherence_values) < 2:
            return {
                "trend": "insufficient_data",
                "stability": 0.0,
                "direction": "none",
                "variance": 0.0,
            }

        # Calculate trend metrics
        start_value = coherence_values[0]
        end_value = coherence_values[-1]
        delta = end_value - start_value
        variance = sum(
            (x - sum(coherence_values) / len(coherence_values)) ** 2
            for x in coherence_values
        ) / len(coherence_values)

        # Determine trend direction
        if abs(delta) < 0.05:
            direction = "stable"
        elif delta > 0:
            direction = "increasing"
        else:
            direction = "decreasing"

        # Determine stability
        stability = 1.0 - min(1.0, variance * 10)  # Scale variance for stability score

        # Overall trend
        if stability > 0.8:
            if direction == "stable":
                trend = "highly_stable"
            else:
                trend = f"stable_{direction}"
        elif stability > 0.5:
            if direction == "stable":
                trend = "moderately_stable"
            else:
                trend = f"moderate_{direction}"
        else:
            trend = "unstable"

        return {
            "trend": trend,
            "stability": stability,
            "direction": direction,
            "variance": variance,
            "start_value": start_value,
            "end_value": end_value,
            "delta": delta,
        }

    def detect_loop(self, min_repetitions: int = 2) -> Optional[Dict[str, Any]]:
        """
        Detect repeating patterns in execution paths

        Args:
            min_repetitions: Minimum number of repetitions to consider a loop

        Returns:
            Dict with loop information if detected, None otherwise
        """
        if len(self.execution_paths) < min_repetitions * 2:
            return None

        # Get functions from execution path
        functions = [p["function"] for p in self.execution_paths]

        # Look for repeating sequences
        for seq_len in range(2, len(functions) // min_repetitions + 1):
            for i in range(len(functions) - seq_len * min_repetitions + 1):
                sequence = functions[i : i + seq_len]
                is_loop = True

                # Check if sequence repeats
                for j in range(1, min_repetitions):
                    compare_seq = functions[i + j * seq_len : i + (j + 1) * seq_len]
                    if sequence != compare_seq:
                        is_loop = False
                        break

                if is_loop:
                    # Calculate coherence stats across the loop
                    loop_indices = range(i, i + seq_len * min_repetitions)
                    loop_coherence = [
                        self.execution_paths[idx]["coherence"] for idx in loop_indices
                    ]

                    return {
                        "sequence": sequence,
                        "repetitions": min_repetitions,
                        "length": seq_len,
                        "start_idx": i,
                        "coherence_stats": {
                            "mean": sum(loop_coherence) / len(loop_coherence),
                            "min": min(loop_coherence),
                            "max": max(loop_coherence),
                            "variance": sum(
                                (x - sum(loop_coherence) / len(loop_coherence)) ** 2
                                for x in loop_coherence
                            )
                            / len(loop_coherence),
                        },
                    }

        return None

    def _detect_patterns(self) -> List[Dict[str, Any]]:
        """
        Detect patterns in memory windows

        Returns:
            List of detected patterns
        """
        patterns = []

        # Analyze short-term window for immediate patterns
        short_window = self.windows.get("window_1x", [])
        if len(short_window) >= 3:
            # Check for oscillation pattern (up-down-up or down-up-down)
            # Convert deque to list before slicing
            last_three = list(short_window)[-3:]
            coherence_values = [state["coherence"] for state in last_three]
            if (
                coherence_values[0] < coherence_values[1]
                and coherence_values[1] > coherence_values[2]
            ) or (
                coherence_values[0] > coherence_values[1]
                and coherence_values[1] < coherence_values[2]
            ):
                patterns.append(
                    {
                        "type": "oscillation",
                        "window": "short",
                        "severity": "low",
                        "values": coherence_values,
                    }
                )

        # Only log significant patterns
        if patterns:
            for pattern in patterns:
                if pattern["type"] == "oscillation":
                    logger.info(
                        f"Detected oscillation pattern in {pattern['window']} window"
                    )

        return patterns

    def _predict_divergence(
        self, current_coherence: float, current_exploration: float
    ) -> Optional[Dict[str, Any]]:
        """
        Predict potential coherence divergence

        Args:
            current_coherence: Current coherence value
            current_exploration: Current exploration value

        Returns:
            Dict with prediction information if divergence is likely, None otherwise
        """
        # Not enough data for prediction
        if not self.windows["window_2x"]:
            return None

        # Get medium-term trend
        trend = self.get_coherence_trend("window_2x")

        # Prediction logic based on trends
        prediction = None

        # Condition 1: Decreasing coherence with high variance
        if trend["direction"] == "decreasing" and trend["variance"] > 0.03:
            probability = min(0.9, trend["variance"] * 10 + abs(trend["delta"]) * 5)
            time_estimate = (
                int(abs((0.75 - current_coherence) / trend["delta"]))
                if trend["delta"] != 0
                else "unknown"
            )

            prediction = {
                "type": "divergence_risk",
                "probability": probability,
                "reason": "decreasing_coherence_high_variance",
                "timeframe": (
                    str(time_estimate)
                    if isinstance(time_estimate, int)
                    else time_estimate
                ),
                "current_values": {
                    "coherence": current_coherence,
                    "exploration": current_exploration,
                },
                "trend": trend,
            }

        # Condition 2: Unstable trend near critical threshold
        elif trend["stability"] < 0.6 and 0.65 <= current_coherence <= 0.72:
            probability = (0.75 - current_coherence) * 5 + (
                1 - trend["stability"]
            ) * 0.5

            prediction = {
                "type": "threshold_risk",
                "probability": probability,
                "reason": "unstable_near_threshold",
                "timeframe": "2-3",
                "current_values": {
                    "coherence": current_coherence,
                    "exploration": current_exploration,
                },
                "trend": trend,
            }

        return prediction
