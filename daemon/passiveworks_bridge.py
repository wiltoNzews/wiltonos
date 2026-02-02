#!/usr/bin/env python3
"""
PassiveWorks Bridge
===================
Connects the PassiveWorks gold modules to the breathing daemon.

This bridges what was built (PassiveWorks) to what we have now (crystals + daemon).

December 2025 — Finally connected.
"""

import sys
from pathlib import Path

# Add passiveworks to path
PASSIVEWORKS_PATH = Path.home() / "wiltonos" / "core" / "passiveworks"
sys.path.insert(0, str(PASSIVEWORKS_PATH))
sys.path.insert(0, str(PASSIVEWORKS_PATH / "wilton_core"))
sys.path.insert(0, str(PASSIVEWORKS_PATH / "agents"))

import logging
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("passiveworks_bridge")


@dataclass
class PassiveWorksState:
    """State from PassiveWorks modules."""
    brazilian_wave_value: float = 0.75
    fractal_observer_state: str = "stability"
    lemniscate_state: str = "dormant"
    coherence_attractor_pull: float = 0.0
    qctf_value: float = 0.75


class PassiveWorksBridge:
    """
    Bridge between PassiveWorks modules and the breathing daemon.

    Provides a unified interface to:
    - Brazilian Wave Protocol (coherence transformation)
    - Fractal Observer (consciousness simulation)
    - Lemniscate Mode (transcendence detection)
    - Coherence Attractor (dynamic field)
    - QCTF (quantum coherence threshold)
    """

    def __init__(self):
        self.state = PassiveWorksState()
        self.modules_loaded = {}
        self._load_modules()

    def _load_modules(self):
        """Load available PassiveWorks modules."""

        # Try to load Brazilian Wave
        try:
            from brazilian_wave import BrazilianWaveTransformer
            self.brazilian_wave = BrazilianWaveTransformer
            self.modules_loaded['brazilian_wave'] = True
            logger.info("Brazilian Wave loaded")
        except Exception as e:
            logger.warning(f"Could not load Brazilian Wave: {e}")
            self.brazilian_wave = None
            self.modules_loaded['brazilian_wave'] = False

        # Try to load Coherence Attractor
        try:
            from core.coherence_attractor import CoherenceAttractor
            self.coherence_attractor = CoherenceAttractor()
            self.modules_loaded['coherence_attractor'] = True
            logger.info("Coherence Attractor loaded")
        except Exception as e:
            logger.warning(f"Could not load Coherence Attractor: {e}")
            self.coherence_attractor = None
            self.modules_loaded['coherence_attractor'] = False

        # Try to load QCTF
        try:
            from qctf.qctf_core import QCTF, QCTFData
            self.qctf_class = QCTF
            self.qctf_data = QCTFData()
            self.modules_loaded['qctf'] = True
            logger.info("QCTF loaded")
        except Exception as e:
            logger.warning(f"Could not load QCTF: {e}")
            self.qctf_class = None
            self.qctf_data = None
            self.modules_loaded['qctf'] = False

        # Try to load Lemniscate Mode
        try:
            from lemniscate_mode import Agent as LemniscateAgent
            self.lemniscate = LemniscateAgent()
            self.modules_loaded['lemniscate'] = True
            logger.info("Lemniscate Mode loaded")
        except Exception as e:
            logger.warning(f"Could not load Lemniscate Mode: {e}")
            self.lemniscate = None
            self.modules_loaded['lemniscate'] = False

        logger.info(f"Modules loaded: {sum(self.modules_loaded.values())}/{len(self.modules_loaded)}")

    def transform_coherence(self, current_value: float, sigma: float = 0.1) -> float:
        """
        Apply Brazilian Wave transformation to a coherence value.

        P_{t+1} = 0.75 · P_t + 0.25 · N(P_t, σ)
        """
        if self.brazilian_wave:
            try:
                # The transform_value is a static method but has bugs with 'self'
                # We'll implement the formula directly
                import random
                import math

                # Generate gaussian noise
                u = random.random()
                while u == 0:
                    u = random.random()
                v = random.random()
                while v == 0:
                    v = random.random()
                z = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)
                noise = current_value + (z * sigma)

                # Apply Brazilian Wave formula
                next_value = (0.75 * current_value) + (0.25 * noise)
                # Clamp to [0, 1] — coherence has no meaning outside this range
                next_value = max(0.0, min(1.0, next_value))
                self.state.brazilian_wave_value = next_value
                return next_value
            except Exception as e:
                logger.error(f"Brazilian Wave transform failed: {e}")

        return current_value

    def get_attractor_pull(self, current_coherence: float) -> float:
        """
        Get the attractor field's pull toward target coherence (0.75).

        Returns a value indicating how strongly the field is pulling.
        """
        if self.coherence_attractor:
            try:
                target = 0.75
                distance = abs(current_coherence - target)
                # Exponential attraction - stronger when far from target
                pull = 1.0 - math.exp(-distance * 2)
                self.state.coherence_attractor_pull = pull
                return pull
            except Exception as e:
                logger.error(f"Attractor pull calculation failed: {e}")

        return 0.0

    def get_lemniscate_state(self) -> str:
        """
        Get current lemniscate state (dormant, active, transcendent).
        """
        if self.lemniscate:
            try:
                self.state.lemniscate_state = self.lemniscate.get_lemniscate_state()
                return self.state.lemniscate_state
            except Exception as e:
                logger.error(f"Lemniscate state failed: {e}")

        return "dormant"

    def check_transcendence(self, coherence: float) -> bool:
        """
        Check if coherence level indicates transcendence potential.

        Transcendence threshold: coherence > 0.89
        Exits transcendence when coherence drops below 0.7 (hysteresis).
        """
        if self.lemniscate:
            try:
                # Update lemniscate with current coherence
                self.lemniscate.coherence = coherence

                # Check for transcendence
                if coherence > 0.89:
                    if self.lemniscate.lemniscate_state != "transcendent":
                        self.lemniscate.lemniscate_state = "transcendent"
                        logger.info("💫 TRANSCENDENCE DETECTED")
                        return True
                    return True  # already transcendent, still above threshold
                elif coherence < 0.7:
                    # Hysteresis: exit transcendence at 0.7 (not 0.89)
                    # to avoid flickering at the boundary
                    if self.lemniscate.lemniscate_state == "transcendent":
                        self.lemniscate.lemniscate_state = "active"
                        logger.info("Lemniscate: transcendent → active (coherence dropped below 0.7)")
                    self.state.lemniscate_state = self.lemniscate.lemniscate_state
                    return False
            except Exception as e:
                logger.error(f"Transcendence check failed: {e}")

        return False

    def get_qctf_value(self) -> float:
        """
        Get current QCTF (Quantum Coherence Threshold Function) value.
        """
        if self.qctf_data:
            try:
                # Simplified QCTF calculation
                gef = self.qctf_data.gef  # Global Entanglement Factor
                qeai = self.qctf_data.qeai  # Quantum Ethical Alignment Index
                ci = self.qctf_data.ci  # Coherence Index

                # QCTF = GEF × QEAI × CI (simplified)
                qctf = gef * qeai * ci
                self.state.qctf_value = qctf
                return qctf
            except Exception as e:
                logger.error(f"QCTF calculation failed: {e}")

        return 0.75

    def apply_fractal_oscillation(self, step: int) -> str:
        """
        Determine current state in fractal 3:1 oscillation.

        Returns: "stability" (75% of time) or "exploration" (25% of time)
        """
        # Fractal 3:1 pattern
        cycle_pos = step % 4
        if cycle_pos < 3:
            self.state.fractal_observer_state = "stability"
        else:
            self.state.fractal_observer_state = "exploration"

        return self.state.fractal_observer_state

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state from all PassiveWorks modules."""
        return {
            "brazilian_wave_value": self.state.brazilian_wave_value,
            "fractal_observer_state": self.state.fractal_observer_state,
            "lemniscate_state": self.state.lemniscate_state,
            "coherence_attractor_pull": self.state.coherence_attractor_pull,
            "qctf_value": self.state.qctf_value,
            "modules_loaded": self.modules_loaded
        }


# Singleton instance
_bridge_instance = None

def get_bridge() -> PassiveWorksBridge:
    """Get the singleton PassiveWorks bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = PassiveWorksBridge()
    return _bridge_instance


def test_bridge():
    """Test the PassiveWorks bridge."""
    import math

    print("\n" + "=" * 60)
    print("PASSIVEWORKS BRIDGE TEST")
    print("=" * 60)

    bridge = get_bridge()

    print(f"\nModules loaded: {bridge.modules_loaded}")

    # Test Brazilian Wave
    print("\n--- Brazilian Wave Transform ---")
    value = 0.5
    for i in range(5):
        value = bridge.transform_coherence(value)
        print(f"Step {i+1}: {value:.4f}")

    # Test Fractal Oscillation
    print("\n--- Fractal Oscillation ---")
    for step in range(8):
        state = bridge.apply_fractal_oscillation(step)
        print(f"Step {step}: {state}")

    # Test Attractor Pull
    print("\n--- Attractor Pull ---")
    for coherence in [0.3, 0.5, 0.7, 0.75, 0.9]:
        pull = bridge.get_attractor_pull(coherence)
        print(f"Coherence {coherence}: Pull = {pull:.4f}")

    # Test QCTF
    print("\n--- QCTF Value ---")
    qctf = bridge.get_qctf_value()
    print(f"QCTF: {qctf:.4f}")

    # Test Lemniscate
    print("\n--- Lemniscate State ---")
    lstate = bridge.get_lemniscate_state()
    print(f"State: {lstate}")

    # Test Transcendence
    print("\n--- Transcendence Check ---")
    for coherence in [0.7, 0.85, 0.9, 0.95]:
        is_trans = bridge.check_transcendence(coherence)
        print(f"Coherence {coherence}: Transcendent = {is_trans}")

    # Full state
    print("\n--- Full State ---")
    import json
    print(json.dumps(bridge.get_full_state(), indent=2))

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_bridge()
