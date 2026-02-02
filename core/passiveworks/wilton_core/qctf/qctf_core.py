"""
QCTF Core Implementation
------------------------

This module provides the Python implementation of the core QCTF classes
transpiled from the TypeScript original source.
"""

import math
import time
from typing import Any, Union, Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ToggleState(BaseModel):
    """Toggle state - represents the state of a quantum-control protocol"""

    active: bool
    value: float = Field(ge=0.8, le=1.2)  # Changed lt to le (less than or equal)
    last_activated: Optional[str] = None  # ISO timestamp
    source_module: Optional[str] = None


class Toggles(BaseModel):
    """Toggles collection - holds all possible toggle states"""

    stop: ToggleState
    failsafe: ToggleState
    reroute: ToggleState
    wormhole: ToggleState


class ScalingMetrics(BaseModel):
    """Enhanced scaling metrics - parameters for Dimension Scaling Factor calculation"""

    # Core scaling factors
    parallel_tasks: int = Field(gt=0, default=1)
    modules: int = Field(gt=0, default=1)
    depth: int = Field(ge=0, default=0)  # Fractal depth

    # Operational constraints
    latency: float = Field(ge=0, le=1, default=0)  # Normalized latency [0,1]
    error_rate: float = Field(ge=0, le=1, default=0)  # Error rate [0,1]

    # Legacy fields (maintained for compatibility)
    hpc_scale_factor: float = Field(gt=0, default=1.0)
    timeline_branches: int = Field(gt=0, default=1)


class ModuleCoherence(BaseModel):
    """Module coherence - coherence values for each Oroboro module"""

    oracle: float = Field(ge=0, le=1)
    nova: float = Field(ge=0, le=1)
    gnosis: float = Field(ge=0, le=1)
    sanctum: float = Field(ge=0, le=1)
    halo: float = Field(ge=0, le=1)


class QCTFHistoryEntry(BaseModel):
    """Enhanced QCTF history entry - historical QCTF values with additional metrics"""

    timestamp: str  # ISO timestamp
    qctf: float  # Final QCTF value

    # Core components
    gef: Optional[float] = None  # Global Entanglement Factor
    qeai: Optional[float] = None  # Quantum Ethical Alignment Index
    ci: Optional[float] = None  # Coherence Index

    # Advanced components
    raw_qctf: Optional[float] = None  # Raw QCTF before smoothing
    smoothed_qctf: Optional[float] = None  # Smoothed QCTF before normalization
    dimension_factor: Optional[float] = None  # 𝓓
    toggle_function: Optional[float] = None  # 𝓣_toggles
    feedback_function: Optional[float] = None  # ℱ
    entropy: Optional[float] = None  # Ψ_entropy

    # Active toggles at this point
    active_toggles: Optional[List[str]] = None


class CycleDetection(BaseModel):
    """Cycle detection configuration"""

    enabled: bool = False
    period: float = 86400  # Default: 1 day in seconds
    phase: float = 0
    amplitude: float = 0.1


class QCTFData(BaseModel):
    """Complete QCTF data - all components needed for QCTF calculation"""

    # Basic QCTF components
    gef: float = Field(ge=0, le=1, default=0.85)  # Global Entanglement Factor
    qeai: float = Field(ge=0, le=1, default=0.9)  # Quantum Ethical Alignment Index
    ci: float = Field(ge=0, le=1, default=0.8)  # Coherence Index

    # QCTF parameters
    entropy_scale: float = Field(gt=0, default=10)
    entropy: float = Field(ge=0, le=1, default=0.15)  # Ψ_entropy (base)
    oroboro_constant: float = Field(gt=0, default=1.618)  # Ω (golden ratio)

    # Fixed constants - calibrated for QCTF v4.0
    epsilon: float = Field(gt=0, default=1e-6)  # Numerical safeguard
    kappa: float = Field(gt=0, default=0.05)  # Dimension scaling parameter
    epsilon_d: float = Field(gt=0, default=0.1)  # Latency impact factor
    gamma: float = Field(ge=0, le=1, default=0.5)  # Toggle conflict resolution
    lambda_: float = Field(ge=0, le=1, default=0.8)  # Temporal smoothing
    k: float = Field(gt=0, default=1)  # Tanh normalization steepness
    mu: float = Field(gt=0, default=0.1)  # Toggle decay rate
    alpha: float = Field(gt=0, default=0.2)  # Feedback current rate weight
    beta: float = Field(gt=0, default=0.1)  # Feedback prediction weight
    eta: float = Field(gt=0, default=0.03)  # Module coherence sensitivity
    max_error: float = Field(gt=0, default=0.2)  # Maximum error rate cap

    # Toggle weights
    toggle_weights: Dict[str, float] = Field(
        default={"stop": 0.6, "failsafe": 0.25, "reroute": 0.1, "wormhole": 0.05}
    )

    # Core QCTF v4.0 components
    dimension_scaling_factor: float = Field(ge=1, default=1.0)  # 𝓓
    feedback_function: float = Field(ge=0.5, le=1, default=1.0)  # ℱ
    toggles: Toggles = Field(
        default_factory=lambda: Toggles(
            stop=ToggleState(active=False, value=0.8, last_activated=None),
            failsafe=ToggleState(active=False, value=0.9, last_activated=None),
            reroute=ToggleState(active=False, value=1.1, last_activated=None),
            wormhole=ToggleState(active=False, value=1.2, last_activated=None),
        )
    )

    # Tracking and history
    scaling_metrics: ScalingMetrics = Field(default_factory=lambda: ScalingMetrics())
    module_coherence: ModuleCoherence = Field(
        default_factory=lambda: ModuleCoherence(
            oracle=0.85, nova=0.75, gnosis=0.8, sanctum=0.9, halo=0.82
        )
    )
    history: List[QCTFHistoryEntry] = Field(default_factory=list)

    # Previous state for smoothing and feedback
    previous_raw_qctf: Optional[float] = 0.5  # Initialized at 0.5 for neutral start
    previous_delta_qctf: Optional[float] = 0  # Change in last calculation

    # Optional configuration
    cycle_detection: Optional[CycleDetection] = None


class ToggleEvent(BaseModel):
    """Toggle event - represents a toggle activation/deactivation event"""

    id: Union[str, int]
    toggle_type: str  # Enum: 'stop', 'failsafe', 'reroute', 'wormhole'
    action: str  # Enum: 'activate', 'deactivate'
    source_module: str
    target_module: Optional[str] = None  # For reroute
    reason: str
    timestamp: str  # ISO timestamp
    impact: Optional[str] = None  # Enum: 'HIGH', 'MEDIUM', 'LOW'
    value: Optional[float] = None  # Toggle value when activated

    @validator("toggle_type")
    def validate_toggle_type(cls, v):
        allowed_types = ["stop", "failsafe", "reroute", "wormhole"]
        if v not in allowed_types:
            raise ValueError(f"toggle_type must be one of {allowed_types}")
        return v

    @validator("action")
    def validate_action(cls, v):
        allowed_actions = ["activate", "deactivate"]
        if v not in allowed_actions:
            raise ValueError(f"action must be one of {allowed_actions}")
        return v

    @validator("impact")
    def validate_impact(cls, v):
        if v is not None:
            allowed_impacts = ["HIGH", "MEDIUM", "LOW"]
            if v not in allowed_impacts:
                raise ValueError(f"impact must be one of {allowed_impacts}")
        return v


class QCTF:
    """
    Main QCTF class implementing the Quantum Coherence Transfer Function

    This class maintains key constants and provides access to the core
    QCTF functionality.
    """

    # Constants from the TypeScript implementation
    TOGGLE_AUTH_MATRIX = {
        "stop": ["Oracle", "Sanctum"],  # Only these can emergency stop
        "failsafe": ["Oracle", "Sanctum", "Halo"],  # These can trigger failsafe
        "reroute": ["Oracle", "Halo", "Nova"],  # These can reroute
        "wormhole": ["Oracle", "Halo"],  # Only these can create wormholes
    }

    # Default QCTF values - including THE CRITICAL 3:1 RATIO
    DEFAULT_COHERENCE_VALUE = 0.75  # 75% coherence (3:1 ratio)
    DEFAULT_EXPLORATION_VALUE = 0.25  # 25% exploration (1:3 ratio)
    GOLDEN_RATIO = 1.618  # Φ (phi)

    @classmethod
    def create_default_data(cls):
        """Create a default QCTFData instance with 3:1 coherence ratio values"""
        return QCTFData()

    @staticmethod
    def iso_timestamp():
        """Generate an ISO format timestamp string"""
        return datetime.utcnow().isoformat()

    @staticmethod
    def current_time_ms():
        """Get current time in milliseconds"""
        return int(time.time() * 1000)
