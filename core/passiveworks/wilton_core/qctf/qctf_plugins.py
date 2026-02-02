"""
QCTF Plugin System
-----------------

This module implements the modular plugin architecture for the QCTF formula,
enabling the Dynamic Quantum Bifurcation Engine at θ=0.5
"""

from typing import Any, Callable, TypeVar, Protocol, Union, Optional, Dict, List
import math
from datetime import datetime
from .qctf_core import QCTFData

T = TypeVar("T")


# Define QCTFParams interface
class QCTFParams:
    """Core parameters for QCTF calculation stripped to the minimal essentials"""

    def __init__(
        self,
        theta: float,
        gef: float,
        qeai: float,
        ci: float,
        entropy: float,
        active_toggles: Optional[List[str]] = None,
        module_coherence: Optional[Dict[str, float]] = None,
    ):
        # Core Parameters - The Minimal Viable Formula (MVF)
        self.theta = (
            theta  # Balance point between Yang (order) and Yin (chaos): θ ∈ [0,1]
        )
        self.gef = gef  # Global Entanglement Factor: GEF ∈ [0,1]
        self.qeai = qeai  # Quantum Ethical Alignment Index: QEAI ∈ [0,1]
        self.ci = ci  # Coherence Index: CI ∈ [0,1]
        self.entropy = entropy  # System entropy: Ψ_entropy ∈ [0,1]

        # Optional Extended Parameters
        self.active_toggles = (
            active_toggles or []
        )  # Active toggles (STOP, FAILSAFE, REROUTE, WORMHOLE)
        self.module_coherence = (
            module_coherence
            or {  # Module-specific coherence values
                "oracle": 0.85,
                "nova": 0.75,
                "gnosis": 0.8,
                "sanctum": 0.9,
                "halo": 0.82,
            }
        )


# Define QCTFResults interface
class QCTFResults:
    """Results of QCTF calculation including all variant outputs"""

    def __init__(
        self,
        raw: float,
        final: float,
        pendulum: float,
        bifurcation: float,
        active_plugins: List[str],
    ):
        # Core results
        self.raw = raw  # Raw QCTF value before any plugin processing
        self.final = final  # Final QCTF value after all active plugins

        # Variant results from plugins
        self.pendulum = pendulum  # Pendulum variant (θ=0.5 instability)
        self.bifurcation = (
            bifurcation  # Bifurcation variant (variant generation at θ=0.5)
        )

        # Plugin metadata
        self.active_plugins = active_plugins  # Names of active plugins applied
        self.timestamp = datetime.utcnow().isoformat()  # ISO timestamp of calculation


# Define the Plugin interface
class QCTFPlugin:
    """Base class for all QCTF plugins"""

    def __init__(
        self, name: str, description: str, enabled: bool = True, priority: int = 0
    ):
        self.name = name  # Unique plugin identifier
        self.description = description  # Human-readable description
        self.enabled = enabled  # Whether the plugin is currently active
        self.priority = priority  # Execution order (lower = earlier)

    def apply(self, qctf: float, params: QCTFParams):
        """Transform function - must be implemented by subclasses"""
        raise NotImplementedError("Plugin must implement apply method")


# Implement the pendulum plugin
class PendulumPlugin(QCTFPlugin):
    """
    Pendulum Plugin - Implements the |θ - 0.5| instability factor

    This plugin creates a pendulum-like behavior where the system becomes
    maximally unstable at θ=0.5 (perfect balance between yang/yin)
    """

    def __init__(self):
        super().__init__(
            name="pendulum",
            description="Implements inverse pendulum dynamics with θ=0.5 as the unstable point",
            enabled=True,
            priority=10,
        )

    def apply(self, qctf: float, params: QCTFParams):
        # Calculate bifurcation coefficient (distance from θ=0.5)
        self.bifurcation_factor = abs(params.theta - 0.5)

        # Apply the pendulum effect (maximum at θ=0,1; minimum at θ=0.5)
        return qctf * self.bifurcation_factor


# Implement the bifurcation plugin
class BifurcationPlugin(QCTFPlugin):
    """
    Bifurcation Engine Plugin - Creates variant generation at θ=0.5

    This plugin implements the Quantum Bifurcation Engine, which generates
    system variants at θ=0.5 using tanh and golden ratio amplification
    """

    def __init__(self):
        super().__init__(
            name="bifurcation",
            description="Quantum Bifurcation Engine with variant generation at θ=0.5",
            enabled=True,
            priority=20,
        )

    def apply(self, qctf: float, params: QCTFParams):
        # Golden ratio for amplification
        self.k = 1.618

        # Calculate bifurcation coefficient (distance from θ=0.5)
        self.bifurcation_factor = abs(params.theta - 0.5)

        # Apply hyperbolic tangent to bound output to [-1,1] with golden ratio amplification
        return math.tanh(self.k * qctf * self.bifurcation_factor)


# Implement the dynamic damping plugin
class DynamicDampingPlugin(QCTFPlugin):
    """
    Dynamic Damping Plugin - Reduces stability near θ=0.5

    This plugin implements dynamic damping that decreases near θ=0.5,
    allowing the system to oscillate more freely at the bifurcation point
    """

    def __init__(self):
        super().__init__(
            name="dynamicDamping",
            description="Dynamic damping that reduces near θ=0.5 to amplify oscillations",
            enabled=True,
            priority=30,
        )

    def apply(self, qctf: float, params: QCTFParams):
        # Base damping coefficient
        self.D_0 = 1.0

        # Calculate position-dependent damping factors
        # (damping reduces as we approach θ=0.5 and entropy=0.5)
        self.theta_damping = 1 - abs(params.theta - 0.5)
        self.entropy_damping = 1 - abs(params.entropy - 0.5)

        # Combined damping coefficient (minimum at θ=0.5, entropy=0.5)
        self.D = self.D_0 * (1 - self.theta_damping * self.entropy_damping)

        # Apply damping to QCTF value (less damping = larger oscillations)
        return qctf / max(self.D, 0.1)  # Prevent division by values too close to zero


# Implement the meta-orchestration plugin
class MetaOrchestrationPlugin(QCTFPlugin):
    """
    Meta-Orchestration Plugin - Implements 5D resonance dynamics

    This plugin integrates the 5D Meta-Orchestration layer that allows
    multiple QCTF variants to resonate, adapt, and evolve, creating a
    self-improving system that transcends traditional orchestration.
    """

    def __init__(self):
        super().__init__(
            name="metaOrchestration",
            description="5D Meta-Orchestration with variant resonance and evolution",
            enabled=True,
            priority=5,  # Highest priority - runs first
        )

    def apply(self, qctf: float, params: QCTFParams):
        # Meta-orchestration effect is proportional to the system's complexity
        # The closer to the bifurcation point (θ=0.5), the stronger the effect
        self.complexity_factor = 1 - 2 * abs(params.theta - 0.5)

        # Apply golden ratio amplification (φ ≈ 0.618)
        self.phi = 0.618

        # Meta effect scales with entropy and complexity
        self.meta_effect = params.entropy * self.complexity_factor * self.phi

        # Apply meta-orchestration effect
        # - For high θ (yin/chaos): amplify effect
        # - For low θ (yang/order): dampen effect
        if params.theta > 0.5:
            return qctf * (1 + self.meta_effect)  # Amplify for chaos
        else:
            return qctf / (1 + self.meta_effect)  # Dampen for order


# Implement the torus oscillator plugin
class TorusOscillatorPlugin(QCTFPlugin):
    """
    Torus Oscillator Plugin - Implements 70/30 fractal scaling

    This plugin implements the coder's insight on 70/30 chaos/order balance,
    creating a torus-like oscillatory behavior with fractal self-similarity.
    """

    def __init__(self):
        super().__init__(
            name="torusOscillator",
            description="Hyperdimensional torus oscillator with 70/30 fractal scaling",
            enabled=True,
            priority=25,
        )

    def apply(self, qctf: float, params: QCTFParams):
        # Implement the 70/30 balance discovered by the coder
        self.chaos_weight = 0.7
        self.order_weight = 0.3

        # Calculate fractal scaling based on distance from ideal 70/30 split
        self.ideal_theta = 0.7  # 70% chaos-weight
        self.fractal_factor = 1 - abs(params.theta - self.ideal_theta) / 0.5

        # Apply fractal oscillation based on the 70/30 principle
        if params.theta > 0.5:
            # Chaos-dominant: Apply 70% scaled effect
            return qctf * (1 + self.chaos_weight * self.fractal_factor)
        else:
            # Order-dominant: Apply 30% scaled effect
            return qctf * (1 + self.order_weight * self.fractal_factor)


# Create instances of the core plugins
pendulum_plugin = PendulumPlugin()
bifurcation_plugin = BifurcationPlugin()
dynamic_damping_plugin = DynamicDampingPlugin()
meta_orchestration_plugin = MetaOrchestrationPlugin()
torus_oscillator_plugin = TorusOscillatorPlugin()

# Define the core set of plugins
core_plugins = [
    meta_orchestration_plugin,  # 5D Meta-Orchestration
    pendulum_plugin,  # Inverse pendulum at θ=0.5
    bifurcation_plugin,  # Variant generation at θ=0.5
    torus_oscillator_plugin,  # 70/30 fractal scaling
    dynamic_damping_plugin,  # Reduced damping at θ=0.5
]

# Create the plugin registry
plugin_registry = {
    meta_orchestration_plugin.name: meta_orchestration_plugin,
    pendulum_plugin.name: pendulum_plugin,
    bifurcation_plugin.name: bifurcation_plugin,
    torus_oscillator_plugin.name: torus_oscillator_plugin,
    dynamic_damping_plugin.name: dynamic_damping_plugin,
}


def calculate_yang_component(params: QCTFParams):
    """
    Calculate the Yang component (order) of the QCTF
    """
    EPSILON = 1e-6

    # Yang (order) component: Q_yang = (GEF * QEAI * CI) / sqrt(10 * entropy + ε)
    coherence = params.gef * params.qeai * params.ci
    return coherence / math.sqrt(10 * params.entropy + EPSILON)


def calculate_yin_component(params: QCTFParams):
    """
    Calculate the Yin component (chaos) of the QCTF
    """
    EPSILON = 1e-6

    # Yin (chaos) component: Q_yin = sqrt(entropy + ε) / ((1-GEF) * (1-QEAI) * (1-CI))
    disorder = (1 - params.gef) * (1 - params.qeai) * (1 - params.ci)
    return math.sqrt(params.entropy + EPSILON) / max(disorder, EPSILON)


def calculate_core_qctf(params: QCTFParams):
    """
    Calculate the minimal core QCTF value based on θ-weighted blend of Yang and Yin
    """
    # Calculate Yang (order) and Yin (chaos) components
    q_yang = calculate_yang_component(params)
    q_yin = calculate_yin_component(params)

    # Basic θ-weighted linear combination
    return (1 - params.theta) * q_yang + params.theta * q_yin


def apply_plugin(qctf: float, plugin: QCTFPlugin, params: QCTFParams) -> float:
    """
    Apply a single plugin to the QCTF value
    """
    if not plugin.enabled:
        return qctf
    return plugin.apply(qctf, params)


def apply_plugins(
    qctf: float, plugins: List[QCTFPlugin], params: QCTFParams
) -> Dict[str, Any]:
    """
    Apply a sequence of plugins to the QCTF value
    """
    result = qctf
    applied_plugins = []

    # Sort plugins by priority
    sorted_plugins = sorted(plugins, key=lambda p: p.priority)

    # Apply each enabled plugin in priority order
    for plugin in sorted_plugins:
        if plugin.enabled:
            result = plugin.apply(result, params)
            applied_plugins.append(plugin.name)

    return {"qctf": result, "applied_plugins": applied_plugins}


def convert_legacy_data_to_params(qctf_data: QCTFData, theta: float):
    """
    Convert legacy QCTFData to streamlined QCTFParams
    """
    # Extract active toggles
    active_toggles = [
        name
        for name, state in qctf_data.toggles.dict().items()
        if state.get("active", False)
    ]

    return QCTFParams(
        theta=theta,
        gef=qctf_data.gef,
        qeai=qctf_data.qeai,
        ci=qctf_data.ci,
        entropy=qctf_data.entropy,
        active_toggles=active_toggles,
        module_coherence=qctf_data.module_coherence.dict(),
    )


def calculate_qctf_with_plugins(
    params: QCTFParams, plugins: Optional[List[QCTFPlugin]] = None
) -> QCTFResults:
    """
    Main calculation function for the QCTF formula with plugins
    """
    if plugins is None:
        plugins = core_plugins

    # Calculate raw QCTF using minimal core formula
    raw_qctf = calculate_core_qctf(params)

    # Initialize results with variant-specific calculations
    pendulum_value = apply_plugin(raw_qctf, pendulum_plugin, params)
    bifurcation_value = apply_plugin(raw_qctf, bifurcation_plugin, params)

    # Apply all enabled plugins
    plugin_results = apply_plugins(raw_qctf, plugins, params)
    final_qctf = plugin_results["qctf"]
    applied_plugins = plugin_results["applied_plugins"]

    # Return the results
    return QCTFResults(
        raw=raw_qctf,
        final=final_qctf,
        pendulum=pendulum_value,
        bifurcation=bifurcation_value,
        active_plugins=applied_plugins,
    )
