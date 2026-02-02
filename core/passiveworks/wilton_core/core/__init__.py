"""
Wilton Core - Quantum Balance System
-----------------------------------

This package provides core components for maintaining quantum balance with the
3:1 coherence ratio (75% coherence, 25% exploration).

Main Components:
- CoherenceAttractor: Maintains quantum balance by attracting system state to target ratio
- LoopMemory: Tracks execution patterns and predicts coherence divergence
- EntropyFilter: Prevents resonance spirals and harmful feedback loops
- MetaLens: High-level monitoring for drift, entropy, and silence gaps
- QuantumOrchestrator: Unified interface for all quantum components
"""

from wilton_core.core.coherence_attractor import CoherenceAttractor
from wilton_core.core.loop_memory import LoopMemory
from wilton_core.core.entropy_filter import EntropyFilter
from wilton_core.core.meta_lens import MetaLens
from wilton_core.core.quantum_orchestrator import QuantumOrchestrator

# Define version
__version__ = "0.1.0"

# Export key constants
TARGET_COHERENCE = 0.75  # Target coherence (stability) - 75%
TARGET_EXPLORATION = 0.25  # Target exploration (chaos) - 25%
TARGET_RATIO = "75:25"  # Target ratio as string representation

__all__ = [
    "CoherenceAttractor",
    "LoopMemory",
    "EntropyFilter",
    "MetaLens",
    "QuantumOrchestrator",
    "TARGET_COHERENCE",
    "TARGET_EXPLORATION",
    "TARGET_RATIO",
]
