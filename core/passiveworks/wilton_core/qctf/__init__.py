"""
QCTF (Quantum Coherence Transfer Function) Module
-------------------------------------------------

This module transpiles the TypeScript QCTF implementation to Python,
providing the core functionality for calculating quantum coherence.

It implements the 3:1 coherence-to-exploration ratio (75% / 25%) as
specified in the original TypeScript implementation.
"""

from .qctf_core import QCTF
from .coherence_calculator import CoherenceCalculator

__all__ = ["QCTF", "CoherenceCalculator"]
