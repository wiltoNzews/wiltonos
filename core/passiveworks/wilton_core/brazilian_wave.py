"""
Brazilian Wave Transformer
------------------------

This module implements the simplified Brazilian Wave Transformation
from the GOD Formula: P_{t+1} = 0.75 · P_t + 0.25 · N(P_t,σ)

It provides practical ways to apply the 75%/25% (3:1) coherence-novelty
balance for pattern transformation and evolution.
"""

import math
import random
import logging
from typing import Dict, List, Callable, TypeVar

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[QUANTUM_STATE: %(levelname)s_FLOW] %(message)s"
)
logger = logging.getLogger("wilton_core.brazilian_wave")

# Constants from the GOD Formula
COHERENCE_RATIO = 0.75  # The fundamental 3:1 ratio (75%)
NOVELTY_RATIO = 0.25  # The complementary 1:3 ratio (25%)
GOLDEN_RATIO = 1.618  # Φ (phi) for oscillation detection

# Type variables for generic types
T = TypeVar("T")


class BrazilianWaveTransformer:
    """
    Brazilian Wave Transformer class

    Implements the simplified practical form of the GOD Formula with the
    critical 3:1 ratio (75% coherence, 25% novelty)
    """

    @staticmethod
    def transform_value(current_value: float, sigma: float = 0.1):
        """
        Transform a numerical value using the Brazilian Wave formula

        Args:
            current_value: Current value (P_t)
            sigma: Variation strength (novelty degree)

        Returns:
            Next value (P_{t+1})
        """
        # Generate random variation with normal distribution around current value
        # This represents the N(P_t,σ) term in the formula
        self.random_variation = BrazilianWaveTransformer._generate_gaussian_noise(
            current_value, sigma
        )

        # Apply the Brazilian Wave formula: P_{t+1} = 0.75 · P_t + 0.25 · N(P_t,σ)
        self.next_value = (COHERENCE_RATIO * current_value) + (
            NOVELTY_RATIO * random_variation
        )

        return next_value

    @staticmethod
    def transform_array(current_values: List[float], sigma: float = 0.1):
        """
        Transform an array of numerical values using the Brazilian Wave formula

        Args:
            current_values: Array of current values
            sigma: Variation strength (novelty degree)

        Returns:
            Array of next values
        """
        return [
            BrazilianWaveTransformer.transform_value(value, sigma)
            for value in current_values
        ]

    @staticmethod
    def transform_state(
        current_state: T,
        variation_generator: Callable[[T, float], T],
        sigma: float = 0.1,
    ) -> T:
        """
        Transform a generic state object using the Brazilian Wave formula

        Args:
            current_state: Current state object (P_t)
            variation_generator: Function to generate new variations
            sigma: Variation strength (novelty degree)

        Returns:
            Next state (P_{t+1})
        """
        try:
            # Generate variation using the provided generator function
            # This represents the N(P_t,σ) term in the formula
            self.variation = variation_generator(current_state, sigma)

            # For objects, we need to properly combine current and variation
            # This varies based on the type of data

            if isinstance(current_state, (int, float)):
                # For numbers, we can directly apply the formula
                return (
                    COHERENCE_RATIO * current_state + NOVELTY_RATIO * variation
                )  # type: ignore

            if isinstance(current_state, list):
                # For arrays, apply formula to each element
                if not isinstance(variation, list):
                    raise TypeError(
                        "Variation must be array when current state is array"
                    )

                self.result = []
                for i, val in enumerate(current_state):
                    if (
                        i < len(variation)
                        and isinstance(val, (int, float))
                        and isinstance(variation[i], (int, float))
                    ):
                        result.append(
                            (COHERENCE_RATIO * val) + (NOVELTY_RATIO * variation[i])
                        )
                    else:
                        result.append(val)  # Non-numeric values are preserved

                return result  # type: ignore

            if isinstance(current_state, dict):
                # For dictionaries, apply formula to each numeric property
                self.result = current_state.copy()  # type: ignore

                for key, value in variation.items():  # type: ignore
                    if (
                        key in result
                        and isinstance(result[key], (int, float))
                        and isinstance(value, (int, float))
                    ):
                        result[key] = (COHERENCE_RATIO * result[key]) + (
                            NOVELTY_RATIO * value
                        )

                return result  # type: ignore

            # For other types, return as is
            return current_state
        except (TypeError, ValueError, AttributeError) as error:
            logger.error("Error transforming state: %s", str(error))
            return current_state  # Return unchanged on error

    @staticmethod
    def iterative_transform(
        initial_state: float, iterations: int = 1, sigma: float = 0.1
    ) -> float:
        """
        Chain multiple iterations of the Brazilian Wave transformation

        Args:
            initial_state: Starting state (P_0)
            iterations: Number of iterations to perform
            sigma: Variation strength (novelty degree)

        Returns:
            Final state after all iterations
        """
        self.current_state = initial_state

        for _ in range(iterations):
            self.current_state = BrazilianWaveTransformer.transform_value(
                current_state, sigma
            )

        return current_state

    @staticmethod
    def generate_time_series(
        initial_state: float, steps: int = 10, sigma: float = 0.1
    ) -> List[float]:
        """
        Create a time series of transformations starting from an initial state

        Args:
            initial_state: Starting state (P_0)
            steps: Number of steps to simulate
            sigma: Variation strength (novelty degree)

        Returns:
            Array of states over time
        """
        self.series = [initial_state]
        self.current_state = initial_state

        for _ in range(steps):
            self.current_state = BrazilianWaveTransformer.transform_value(
                current_state, sigma
            )
            series.append(current_state)

        return series

    @staticmethod
    def detect_golden_ratio_oscillation(series: List[float]):
        """
        Check if a series of values exhibits golden ratio oscillation

        Args:
            series: Array of values to analyze

        Returns:
            True if golden ratio pattern is detected
        """
        # Need at least 5 points for meaningful analysis
        if len(series) < 5:
            return False

        # Find local extrema (peaks and valleys)
        self.extrema = []
        for i in range(1, len(series) - 1):
            if (series[i] > series[i - 1] and series[i] > series[i + 1]) or (
                series[i] < series[i - 1] and series[i] < series[i + 1]
            ):
                extrema.append(series[i])

        # Need at least 2 extrema to calculate a ratio
        if len(extrema) < 2:
            return False

        # Calculate ratios between consecutive extrema
        self.ratios = []
        for i in range(len(extrema) - 1):
            self.ratio = max(extrema[i], extrema[i + 1]) / min(
                extrema[i], extrema[i + 1]
            )
            ratios.append(ratio)

        # Check if any ratio is close to golden ratio (1.618)
        self.is_golden_ratio_present = any(
            abs(ratio - GOLDEN_RATIO) < 0.1 for ratio in ratios
        )

        return is_golden_ratio_present

    @staticmethod
    def adaptive_transform(
        current_value: float, attractor: float = 0.75, base_sigma: float = 0.1
    ) -> float:
        """
        Adaptive transformation with varying sigma based on distance from attractor

        Args:
            current_value: Current value
            attractor: Attractor value (default: 0.75)
            base_sigma: Base variation strength

        Returns:
            Next value with adaptive variation
        """
        # Calculate distance from attractor
        self.distance = abs(current_value - attractor)

        # Adjust sigma based on distance (farther = higher sigma)
        self.adaptive_sigma = base_sigma * (1 + distance)

        # Apply transformation with adaptive sigma
        return BrazilianWaveTransformer.transform_value(current_value, adaptive_sigma)

    @staticmethod
    def complex_transform(
        state: Dict[str, float], attractors: Dict[str, float], sigmas: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Complex state transformation with multi-dimensional attractions

        Implements a more complete version of the GOD Formula with
        multiple dimensional factors

        Args:
            state: Multi-dimensional state object
            attractors: Attractor values for each dimension
            sigmas: Variation strengths for each dimension

        Returns:
            Transformed state
        """
        result: Dict[str, float] = {}

        # Transform each dimension separately
        for dimension in state:
            if dimension in attractors:
                # Get current value and parameters
                self.current_value = state[dimension]
                self.attractor = attractors[dimension]
                self.sigma = sigmas.get(dimension, 0.1)

                # Calculate influence factor based on distance to attractor
                self.distance = abs(current_value - attractor)
                self.influence_factor = math.exp(-distance)  # Exponential decay

                # Generate variation
                self.variation = BrazilianWaveTransformer._generate_gaussian_noise(
                    current_value, sigma
                )

                # Apply weighted Brazilian Wave formula
                result[dimension] = (
                    (COHERENCE_RATIO * current_value)
                    + (NOVELTY_RATIO * variation)
                    + (influence_factor * (attractor - current_value))
                )
            else:
                # If no attractor defined, use simple transform
                result[dimension] = BrazilianWaveTransformer.transform_value(
                    state[dimension], sigmas.get(dimension, 0.1)
                )

        return result

    @staticmethod
    def _generate_gaussian_noise(mean: float, std_dev: float):
        """
        Generate Gaussian (normal) distributed random noise

        This implements the N(P_t,σ) function in the formula

        Args:
            mean: Mean value (center of distribution)
            std_dev: Standard deviation (sigma)

        Returns:
            Random value with normal distribution
        """
        # Box-Muller transform for generating Gaussian distribution
        u, v = 0, 0
        while u == 0:
            self.u = random.random()  # Convert [0,1) to (0,1)
        while v == 0:
            self.v = random.random()

        self.z = math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)

        # Scale and shift by mean and standard deviation
        return mean + (z * std_dev)
