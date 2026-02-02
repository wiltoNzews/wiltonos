"""
Meta Lens Implementation
----------------------

This module provides a high-level monitoring system that tracks coherence across the
entire application, detecting drift, entropy, and silence gaps to ensure the system
maintains quantum balance.
"""

import time
import re
import math
import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import deque, defaultdict
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[QUANTUM_STATE: %(levelname)s_FLOW] %(message)s"
)
logger = logging.getLogger("wilton_core.meta_lens")


class MetaLens:
    """
    MetaLens provides a higher-order view of system coherence and entropy,
    serving as a temporal mirror for the system and detecting patterns
    across multiple modules and time scales.

    It tracks drift, silence gaps, and provides recommendations for
    preventative actions before coherence issues become critical.
    """

    # Singleton instance
    _instance = None

    # Class constants
    MAX_LOG_HISTORY = 1000  # Maximum log entries to keep
    ANALYSIS_INTERVAL = 15  # Seconds between analyses
    ALERT_THRESHOLD = 0.65  # Coherence threshold for alerts
    TARGET_COHERENCE = 0.75  # Target coherence value (75%)
    SILENCE_THRESHOLD = 30  # Seconds of silence before alerting

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern"""
        if cls._instance is None:
            logger.info("Initializing MetaLens singleton")
            cls._instance = super(MetaLens, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self):
        """Initialize the MetaLens monitoring system"""
        # Setup monitoring
        self.log_buffer = deque(maxlen=self.MAX_LOG_HISTORY)
        self.coherence_readings = deque(maxlen=self.MAX_LOG_HISTORY)
        self.module_activity = defaultdict(
            lambda: {"last_seen": time.time(), "count": 0}
        )
        self.silence_monitors = defaultdict(
            lambda: {"last_seen": time.time(), "alerted": False}
        )

        # Analysis metrics
        self.last_analysis = time.time()
        self.system_health = 1.0
        self.coherence_trend = "stable"
        self.notifications = []
        self.recommendations = []

        # Pattern detection
        self.log_patterns = {}
        self.module_correlations = defaultdict(float)
        self.drift_indicators = defaultdict(float)

        # Start monitoring thread (in a real implementation)
        # self._start_monitoring()

        logger.info("MetaLens monitoring system initialized")

    def register_log(
        self, level: str, message: str, source: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Register a log message for analysis

        Args:
            level: Log level (INFO, WARNING, ERROR, etc.)
            message: Log message content
            source: Source module or component

        Returns:
            Dict with log analysis
        """
        timestamp = datetime.utcnow()

        # Parse log for coherence values
        coherence_data = self._extract_coherence_data(message)

        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "source": source or "unknown",
            "coherence_data": coherence_data,
        }

        # Add to buffer
        self.log_buffer.append(log_entry)

        # Update module activity
        self.module_activity[source or "unknown"]["last_seen"] = time.time()
        self.module_activity[source or "unknown"]["count"] += 1

        # If contains coherence data, add to readings
        if coherence_data:
            self.coherence_readings.append(
                {
                    "timestamp": timestamp,
                    "coherence": coherence_data.get("coherence"),
                    "exploration": coherence_data.get("exploration"),
                    "source": source or "unknown",
                }
            )

        # Run periodic analysis
        current_time = time.time()
        if current_time - self.last_analysis > self.ANALYSIS_INTERVAL:
            self._run_analysis()
            self.last_analysis = current_time

        return {
            "entry": log_entry,
            "health": self.system_health,
            "trend": self.coherence_trend,
            "notifications": self.notifications[-3:] if self.notifications else [],
        }

    def register_coherence_reading(
        self,
        coherence: float,
        exploration: float,
        source: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Register a direct coherence reading

        Args:
            coherence: Coherence value [0-1]
            exploration: Exploration value [0-1]
            source: Source of the reading
            context: Optional contextual information

        Returns:
            Dict with analysis information
        """
        timestamp = datetime.utcnow()

        # Create reading entry
        reading = {
            "timestamp": timestamp,
            "coherence": coherence,
            "exploration": exploration,
            "source": source,
            "context": context or {},
        }

        # Add to readings
        self.coherence_readings.append(reading)

        # Update module activity
        self.module_activity[source]["last_seen"] = time.time()
        self.module_activity[source]["count"] += 1

        # Check for alert conditions
        alerts = []

        # Alert on low coherence
        if coherence < self.ALERT_THRESHOLD:
            alert = {
                "type": "low_coherence",
                "severity": "warning" if coherence < 0.6 else "notice",
                "message": f"Low coherence detected in {source}: {coherence:.2f}",
                "source": source,
            }
            alerts.append(alert)
            self.notifications.append(alert)

        # Run periodic analysis
        current_time = time.time()
        if current_time - self.last_analysis > self.ANALYSIS_INTERVAL:
            self._run_analysis()
            self.last_analysis = current_time

        return {
            "reading": reading,
            "health": self.system_health,
            "trend": self.coherence_trend,
            "alerts": alerts,
            "notifications": self.notifications[-3:] if self.notifications else [],
        }

    def check_silence(self, module: str) -> Dict[str, Any]:
        """
        Check if a module has been silent for too long

        Args:
            module: Module name to check

        Returns:
            Dict with silence analysis
        """
        current_time = time.time()

        # Get module's last activity time
        last_seen = self.module_activity.get(module, {}).get("last_seen", current_time)
        silence_duration = current_time - last_seen

        # Update silence monitor
        self.silence_monitors[module]["last_seen"] = last_seen

        # Check if silence exceeds threshold
        is_silent = silence_duration > self.SILENCE_THRESHOLD

        # Generate alert if silent and not already alerted
        alert = None
        if is_silent and not self.silence_monitors[module]["alerted"]:
            alert = {
                "type": "module_silence",
                "severity": "warning",
                "message": f"Module {module} has been silent for {silence_duration:.1f} seconds",
                "source": module,
                "duration": silence_duration,
            }
            self.notifications.append(alert)
            self.silence_monitors[module]["alerted"] = True

        # Reset alert flag if no longer silent
        if not is_silent and self.silence_monitors[module]["alerted"]:
            self.silence_monitors[module]["alerted"] = False

        return {
            "module": module,
            "last_seen": last_seen,
            "silence_duration": silence_duration,
            "is_silent": is_silent,
            "alert": alert,
        }

    def get_coherence_metrics(
        self, window_seconds: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get coherence metrics over a time window

        Args:
            window_seconds: Optional time window in seconds

        Returns:
            Dict with coherence metrics
        """
        if not self.coherence_readings:
            return {
                "status": "no_data",
                "average_coherence": 0,
                "average_exploration": 0,
                "sample_count": 0,
            }

        # Filter by time window if specified
        if window_seconds:
            cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
            readings = [r for r in self.coherence_readings if r["timestamp"] >= cutoff]
        else:
            readings = list(self.coherence_readings)

        # Calculate metrics
        if not readings:
            return {
                "status": "no_data_in_window",
                "average_coherence": 0,
                "average_exploration": 0,
                "sample_count": 0,
            }

        coherence_values = [r["coherence"] for r in readings]
        exploration_values = [r["exploration"] for r in readings]

        avg_coherence = sum(coherence_values) / len(coherence_values)
        avg_exploration = sum(exploration_values) / len(exploration_values)

        # Calculate additional stats
        try:
            coherence_stddev = (
                statistics.stdev(coherence_values) if len(coherence_values) > 1 else 0
            )
            min_coherence = min(coherence_values)
            max_coherence = max(coherence_values)

            # Calculate trend
            if len(coherence_values) > 2:
                first_half = coherence_values[: len(coherence_values) // 2]
                second_half = coherence_values[len(coherence_values) // 2 :]

                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)

                if abs(second_avg - first_avg) < 0.03:
                    trend = "stable"
                elif second_avg > first_avg:
                    trend = "increasing"
                else:
                    trend = "decreasing"
            else:
                trend = "insufficient_data"
        except Exception as e:
            logger.error(f"Error calculating coherence stats: {e}")
            coherence_stddev = 0
            min_coherence = avg_coherence
            max_coherence = avg_coherence
            trend = "error"

        # Evaluate balance
        if 0.7 <= avg_coherence <= 0.8:
            balance = "optimal"
        elif 0.6 <= avg_coherence < 0.7:
            balance = "suboptimal"
        elif avg_coherence < 0.6:
            balance = "imbalanced"
        else:  # > 0.8
            balance = "overstabilized"

        return {
            "status": "success",
            "sample_count": len(readings),
            "time_range": {
                "start": readings[0]["timestamp"].isoformat() if readings else None,
                "end": readings[-1]["timestamp"].isoformat() if readings else None,
            },
            "coherence": {
                "average": avg_coherence,
                "stddev": coherence_stddev,
                "min": min_coherence,
                "max": max_coherence,
            },
            "exploration": {"average": avg_exploration},
            "trend": trend,
            "balance": balance,
            "delta_from_target": avg_coherence - self.TARGET_COHERENCE,
            "sources": list(set(r["source"] for r in readings)),
        }

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for improving system coherence

        Returns:
            List of recommendation objects
        """
        return self.recommendations

    def _extract_coherence_data(self, message: str) -> Optional[Dict[str, float]]:
        """Extract coherence data from log message if present"""
        # Pattern for coherence data in logs
        coherence_pattern = (
            r"(?:coherence|stability)[\s:]+(\d+\.\d+).*?(?:exploration)[\s:]+(\d+\.\d+)"
        )
        ratio_pattern = r"ratio[\s:\"]+(\d+):(\d+)"

        # Try to match patterns
        coherence_match = re.search(coherence_pattern, message, re.IGNORECASE)
        ratio_match = re.search(ratio_pattern, message, re.IGNORECASE)

        if coherence_match:
            try:
                coherence = float(coherence_match.group(1))
                exploration = float(coherence_match.group(2))
                return {"coherence": coherence, "exploration": exploration}
            except (ValueError, IndexError):
                pass

        if ratio_match:
            try:
                coherence_part = int(ratio_match.group(1))
                exploration_part = int(ratio_match.group(2))
                total = coherence_part + exploration_part
                if total > 0:
                    coherence = coherence_part / total
                    exploration = exploration_part / total
                    return {"coherence": coherence, "exploration": exploration}
            except (ValueError, IndexError):
                pass

        return None

    def _run_analysis(self) -> Dict[str, Any]:
        """
        Run periodic analysis of all collected data

        Returns:
            Dict with analysis results
        """
        logger.debug("Running MetaLens analysis")

        # Get recent coherence metrics
        recent_metrics = self.get_coherence_metrics(window_seconds=60)
        long_term_metrics = self.get_coherence_metrics(window_seconds=300)

        # Update system health based on metrics
        if recent_metrics["status"] == "success" and recent_metrics["sample_count"] > 0:
            # Base health on coherence balance and stability
            coherence_health = min(
                1.0, recent_metrics["coherence"]["average"] / self.TARGET_COHERENCE
            )
            stability_factor = 1.0 - min(
                0.5, recent_metrics["coherence"]["stddev"] * 5
            )  # Penalize high variability

            self.system_health = coherence_health * 0.7 + stability_factor * 0.3
            self.coherence_trend = recent_metrics["trend"]

        # Check for silence in all known modules
        for module in list(self.module_activity.keys()):
            self.check_silence(module)

        # Generate recommendations
        self._generate_recommendations(recent_metrics, long_term_metrics)

        # Return analysis summary
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "health": self.system_health,
            "trend": self.coherence_trend,
            "recent_metrics": recent_metrics,
            "long_term_metrics": long_term_metrics,
            "notifications": self.notifications[-5:] if self.notifications else [],
            "recommendations": self.recommendations,
        }

    def _generate_recommendations(
        self, recent_metrics: Dict[str, Any], long_term_metrics: Dict[str, Any]
    ) -> None:
        """
        Generate recommendations based on metrics

        Args:
            recent_metrics: Recent coherence metrics
            long_term_metrics: Longer term coherence metrics
        """
        # Clear old recommendations
        self.recommendations = []

        # Skip if not enough data
        if (
            recent_metrics["status"] != "success"
            or recent_metrics["sample_count"] < 3
            or long_term_metrics["status"] != "success"
        ):
            return

        # Check for coherence drift over time
        recent_coherence = recent_metrics["coherence"]["average"]
        long_term_coherence = long_term_metrics["coherence"]["average"]
        coherence_drift = recent_coherence - long_term_coherence

        # Recommendation for decreasing coherence trend
        if recent_metrics["trend"] == "decreasing" and recent_coherence < 0.7:
            self.recommendations.append(
                {
                    "type": "add_attractor",
                    "priority": "high" if recent_coherence < 0.65 else "medium",
                    "message": f"Add stronger attractor to stabilize decreasing coherence (current: {recent_coherence:.2f})",
                    "details": {
                        "suggested_action": "Add attractor point with higher strength",
                        "coherence_target": 0.75,
                        "strength": 0.9,
                    },
                }
            )

        # Recommendation for high variability
        if recent_metrics["coherence"]["stddev"] > 0.1:
            self.recommendations.append(
                {
                    "type": "apply_dampening",
                    "priority": "medium",
                    "message": f"Apply entropy dampening to reduce coherence variability (stddev: {recent_metrics['coherence']['stddev']:.2f})",
                    "details": {
                        "suggested_action": "Activate entropy dampening",
                        "strength": 0.5,
                        "duration": 10.0,
                    },
                }
            )

        # Recommendation if over-stabilized
        if recent_coherence > 0.85 and long_term_coherence > 0.8:
            self.recommendations.append(
                {
                    "type": "increase_exploration",
                    "priority": "medium",
                    "message": f"System is over-stabilized at {recent_coherence:.2f} coherence, increase exploration",
                    "details": {
                        "suggested_action": "Add exploration attractor",
                        "exploration_target": 0.25,
                        "strength": 0.7,
                    },
                }
            )

        # Recommendation for logging improvement if few coherence readings
        sources_with_data = set(r["source"] for r in self.coherence_readings)
        if len(self.log_buffer) > 20 and len(sources_with_data) < 3:
            self.recommendations.append(
                {
                    "type": "improve_logging",
                    "priority": "low",
                    "message": f"Only {len(sources_with_data)} modules are reporting coherence metrics",
                    "details": {
                        "suggested_action": "Add coherence logging to more modules",
                        "current_sources": list(sources_with_data),
                    },
                }
            )
