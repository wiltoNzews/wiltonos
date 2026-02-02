"""
Quantum Orchestrator Implementation
----------------------------------

This module orchestrates all quantum components to maintain the 3:1 coherence ratio,
providing a unified interface for system-wide quantum balance management.
"""

import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta

# Import quantum components
from wilton_core.core.coherence_attractor import CoherenceAttractor
from wilton_core.core.loop_memory import LoopMemory
from wilton_core.core.entropy_filter import EntropyFilter
from wilton_core.core.meta_lens import MetaLens

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[QUANTUM_STATE: %(levelname)s_FLOW] %(message)s"
)
logger = logging.getLogger("wilton_core.quantum_orchestrator")


class QuantumOrchestrator:
    """
    QuantumOrchestrator provides a unified interface for maintaining quantum balance
    across all system components, ensuring the 3:1 coherence ratio is maintained
    while detecting and preventing problematic patterns like resonance spirals.
    """

    # Singleton instance
    _instance = None

    # Class constants
    TARGET_COHERENCE = 0.75  # Target coherence (stability) - 75%
    TARGET_EXPLORATION = 0.25  # Target exploration (chaos) - 25%

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern"""
        if cls._instance is None:
            logger.info("Initializing QuantumOrchestrator singleton")
            cls._instance = super(QuantumOrchestrator, cls).__new__(cls)
            cls._instance._initialize(*args, **kwargs)
        return cls._instance

    def _initialize(self):
        """Initialize the orchestrator and its components"""
        # Initialize all components
        logger.info("Initializing quantum components...")

        # Initialize in the correct order
        self.attractor = CoherenceAttractor()  # Primary quantum balance maintainer
        self.entropy_filter = EntropyFilter()  # Monitors and dampens resonance patterns
        self.loop_memory = LoopMemory()  # Tracks execution patterns and loops
        self.meta_lens = MetaLens()  # Provides high-level system observation

        # System state
        self.current_coherence = self.TARGET_COHERENCE
        self.current_exploration = self.TARGET_EXPLORATION
        self.quantum_state = "balanced"
        self.last_adjustment = time.time()
        self.calibration_points = (
            set()
        )  # Set of context points where calibration has occurred

        logger.info(
            f"QuantumOrchestrator initialized with target 3:1 ratio "
            f"({self.TARGET_COHERENCE:.2f}:{self.TARGET_EXPLORATION:.2f})"
        )

        # Register initialization with meta_lens
        self.meta_lens.register_coherence_reading(
            self.TARGET_COHERENCE,
            self.TARGET_EXPLORATION,
            "quantum_orchestrator",
            {"event": "initialization"},
        )

    def stabilize(
        self,
        current_coherence: Optional[float] = None,
        current_exploration: Optional[float] = None,
        adjustment_strength: float = 1.0,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Apply quantum stabilization to maintain the 3:1 balance

        Args:
            current_coherence: Current coherence value (defaults to last value)
            current_exploration: Current exploration value (defaults to last value)
            adjustment_strength: Strength of stabilization adjustment [0-inf]
            execution_context: Optional execution context for tracing

        Returns:
            Dict with stabilization results
        """
        # Use provided values or defaults
        coherence = (
            current_coherence
            if current_coherence is not None
            else self.current_coherence
        )
        exploration = (
            current_exploration
            if current_exploration is not None
            else self.current_exploration
        )

        context = execution_context or {}
        start_time = time.time()

        # 1. Record state in loop_memory for pattern detection
        self.loop_memory.record_state(coherence, exploration, context)

        # 2. Check for loops or resonance patterns
        loop_info = self.loop_memory.detect_loop()

        # 3. Apply entropy filtering if needed
        if loop_info:
            # Detected a loop, apply entropy dampening
            logger.info(
                f"Detected execution loop: {', '.join(loop_info['sequence'])}. Applying entropy dampening."
            )
            self.entropy_filter.activate_dampening(strength=0.3, duration=3.0)

            # Register with meta_lens
            self.meta_lens.register_log(
                "WARNING",
                f"Execution loop detected with sequence: {', '.join(loop_info['sequence'])}",
                "quantum_orchestrator",
            )

        # 4. Apply entropy filter dampening if active
        coherence, exploration = self.entropy_filter.detect_and_break_resonance(
            coherence, exploration, context
        )

        # 5. Apply attractor to pull toward target balance
        new_coherence, new_exploration = self.attractor.attract(
            coherence, exploration, adjustment_strength
        )

        # 6. Detect significant drift for traceback
        coherence_change = abs(new_coherence - coherence)
        if coherence_change > 0.1:
            # Large shift, generate traceback
            self.attractor.coherence_traceback(
                f"Large coherence shift detected: {coherence:.2f} → {new_coherence:.2f}",
                depth=5,
            )

        # 7. Update state
        self.current_coherence = new_coherence
        self.current_exploration = new_exploration
        self.last_adjustment = time.time()

        # 8. Update quantum state description
        if abs(new_coherence - self.TARGET_COHERENCE) < 0.05:
            self.quantum_state = "balanced"
        elif new_coherence < self.TARGET_COHERENCE:
            self.quantum_state = "exploration_heavy"
        else:
            self.quantum_state = "stability_heavy"

        # 9. Register with meta_lens
        self.meta_lens.register_coherence_reading(
            new_coherence,
            new_exploration,
            "quantum_orchestrator",
            {"context": context, "adjustment_strength": adjustment_strength},
        )

        # 10. Check for recommendations
        recommendations = self.meta_lens.get_recommendations()
        for rec in recommendations:
            if rec["type"] == "add_attractor" and rec["priority"] == "high":
                # Automatically add attractor for high priority recommendations
                self.attractor.add_attractor(
                    coherence=rec["details"]["coherence_target"],
                    exploration=1.0 - rec["details"]["coherence_target"],
                    strength=rec["details"]["strength"],
                    lifetime=30.0,  # Temporary attractor
                )
                logger.info(
                    f"Added automatic attractor based on recommendation: {rec['message']}"
                )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Generate execution summary
        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "before": {"coherence": coherence, "exploration": exploration},
            "after": {"coherence": new_coherence, "exploration": new_exploration},
            "delta": {
                "coherence": new_coherence - coherence,
                "exploration": new_exploration - exploration,
            },
            "quantum_state": self.quantum_state,
            "loop_detected": loop_info is not None,
            "processing_time_ms": processing_time * 1000,
            "recommendations": recommendations[:3] if recommendations else [],
        }

        return result

    def calibrate_point(
        self,
        context_name: str,
        coherence_offset: float = 0.0,
        exploration_offset: float = 0.0,
        create_attractor: bool = True,
    ) -> Dict[str, Any]:
        """
        Calibrate a specific execution point for ideal quantum balance

        Args:
            context_name: Name of the execution context
            coherence_offset: Optional coherence offset from target
            exploration_offset: Optional exploration offset from target
            create_attractor: Whether to create an attractor at this point

        Returns:
            Dict with calibration results
        """
        # Calculate target values with offsets
        target_coherence = max(
            0.01, min(0.99, self.TARGET_COHERENCE + coherence_offset)
        )
        target_exploration = max(
            0.01, min(0.99, self.TARGET_EXPLORATION + exploration_offset)
        )

        # Normalize to ensure they sum to 1.0
        total = target_coherence + target_exploration
        target_coherence = target_coherence / total
        target_exploration = target_exploration / total

        # Create an attractor if requested
        attractor_id = None
        if create_attractor:
            attractor_id = self.attractor.add_attractor(
                coherence=target_coherence,
                exploration=target_exploration,
                strength=0.7,
                radius=0.15,
            )

        # Add to calibration points
        self.calibration_points.add(context_name)

        # Record with meta_lens
        self.meta_lens.register_log(
            "INFO",
            f"Calibrated quantum point '{context_name}' to {target_coherence:.2f}:{target_exploration:.2f}",
            "quantum_orchestrator",
        )

        logger.info(
            f"Calibrated quantum point '{context_name}' to {target_coherence:.2f}:{target_exploration:.2f}"
        )

        return {
            "context": context_name,
            "target": {
                "coherence": target_coherence,
                "exploration": target_exploration,
                "ratio": f"{int(target_coherence*100)}:{int(target_exploration*100)}",
            },
            "attractor_id": attractor_id,
            "success": True,
        }

    def inject_distortion(
        self, distortion_type: str, magnitude: float, duration: float
    ) -> Dict[str, Any]:
        """
        Inject a controlled quantum distortion for testing or creative exploration

        Args:
            distortion_type: Type of distortion to apply ('temporal', 'spatial', etc.)
            magnitude: Distortion magnitude [0-inf]
            duration: Duration in seconds

        Returns:
            Dict with distortion results
        """
        # Apply distortion to attractor
        success = self.attractor.add_distortion(distortion_type, magnitude, duration)

        if success:
            # Record distortion with meta_lens
            self.meta_lens.register_log(
                "WARNING",
                f"Injected {distortion_type} distortion with magnitude {magnitude} for {duration}s",
                "quantum_orchestrator",
            )

            # Apply corresponding jitter in entropy filter
            jitter_info = self.entropy_filter.apply_jitter(
                intensity=min(1.0, magnitude * 0.2),  # Scale down for jitter
                duration=duration * 0.5,  # Shorter duration for jitter
            )

            logger.warning(
                f"Injected {distortion_type} distortion with magnitude {magnitude}"
            )

            return {
                "distortion_type": distortion_type,
                "magnitude": magnitude,
                "duration": duration,
                "success": True,
                "jitter_seed": jitter_info["seed"],
            }
        else:
            logger.error(f"Failed to inject {distortion_type} distortion")
            return {
                "distortion_type": distortion_type,
                "success": False,
                "error": f"Unknown distortion type: {distortion_type}",
            }

    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status across all quantum components

        Returns:
            Dict with complete system status
        """
        # Get status from all components
        attractor_status = self.attractor.get_status()

        # Get coherence metrics from meta_lens
        recent_metrics = self.meta_lens.get_coherence_metrics(window_seconds=60)

        # Get coherence trend data
        coherence_trend = self.loop_memory.get_coherence_trend()

        # Get recommendations
        recommendations = self.meta_lens.get_recommendations()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "current_state": {
                "coherence": self.current_coherence,
                "exploration": self.current_exploration,
                "ratio": f"{int(self.current_coherence*100)}:{int(self.current_exploration*100)}",
                "quantum_state": self.quantum_state,
            },
            "target": {
                "coherence": self.TARGET_COHERENCE,
                "exploration": self.TARGET_EXPLORATION,
                "ratio": f"{int(self.TARGET_COHERENCE*100)}:{int(self.TARGET_EXPLORATION*100)}",
            },
            "attractor": {
                "field_strength": attractor_status["field"]["strength"],
                "field_radius": attractor_status["field"]["radius"],
                "active_points": attractor_status["field"]["active_points"],
            },
            "calibration_points": len(self.calibration_points),
            "metrics": recent_metrics,
            "trend": coherence_trend,
            "recommendations": recommendations[:5] if recommendations else [],
        }
