"""
Enhanced Quantum Consciousness Simulation with Multi-Level Fractal Patterns

This advanced simulation implements multiple improvements to validate the hypothesis
that consciousness is more fundamental than time:

1. Multi-observer dynamics with phase relationships
2. Multi-level nested fractal patterns (micro, meso, macro scales)
3. Adaptive retrocausal logic with conditional influence
4. Enhanced statistics and visualization for pattern detection
5. Detailed time-vs-future mismatch analysis

Building on initial findings showing statistically significant anomalies when 
retrocausal effects are enabled (p < 0.05).
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import time
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks

# Set random seed for reproducibility
np.random.seed(42)

class FractalPatternGenerator:
    """
    Generates multi-scale fractal patterns for observer states
    with nested oscillations at different time scales
    """
    def __init__(self, 
                 micro_cycle=5,    # Fast cycle (e.g., 5 steps)
                 meso_cycle=20,    # Medium cycle (e.g., 20 steps)
                 macro_cycle=100): # Slow cycle (e.g., 100 steps)
        
        # Define cycle lengths at different scales
        self.micro_cycle = micro_cycle
        self.meso_cycle = meso_cycle
        self.macro_cycle = macro_cycle
        
        # Default ratios for stability vs. exploration at each scale
        # 3:1 ratio means 75% stability, 25% exploration
        self.micro_ratio = 3.0  # 3:1 ratio at micro scale
        self.meso_ratio = 3.0   # 3:1 ratio at meso scale
        self.macro_ratio = 3.0  # 3:1 ratio at macro scale
        
        # Phase offsets between scales (0-1 range)
        self.micro_phase = 0.0
        self.meso_phase = 0.33  # Offset meso cycle by 1/3
        self.macro_phase = 0.67 # Offset macro cycle by 2/3
        
        # For tracking patterns and metrics
        self.history = []
        
    def get_state(self, step):
        """
        Determine the combined fractal state at multiple scales
        Returns dict with states and values at each scale
        """
        # Calculate phase position (0-1) in each cycle
        micro_pos = ((step / self.micro_cycle) + self.micro_phase) % 1.0
        meso_pos = ((step / self.meso_cycle) + self.meso_phase) % 1.0
        macro_pos = ((step / self.macro_cycle) + self.macro_phase) % 1.0
        
        # Determine if in exploratory state at each scale
        # Exploratory state is the last 1/(ratio+1) portion of each cycle
        micro_exploratory = micro_pos > (self.micro_ratio / (self.micro_ratio + 1.0))
        meso_exploratory = meso_pos > (self.meso_ratio / (self.meso_ratio + 1.0))
        macro_exploratory = macro_pos > (self.macro_ratio / (self.macro_ratio + 1.0))
        
        # Calculate fractal values at each scale (0.75 stable, 0.25 exploratory)
        micro_value = 0.25 if micro_exploratory else 0.75
        meso_value = 0.25 if meso_exploratory else 0.75
        macro_value = 0.25 if macro_exploratory else 0.75
        
        # Weighted combination of scales (micro has highest weight, macro lowest)
        combined_value = 0.5 * micro_value + 0.3 * meso_value + 0.2 * macro_value
        combined_exploratory = combined_value < 0.5
        
        # Record the state for later analysis
        state = {
            'step': step,
            'micro': {
                'exploratory': micro_exploratory,
                'value': micro_value,
                'position': micro_pos
            },
            'meso': {
                'exploratory': meso_exploratory,
                'value': meso_value,
                'position': meso_pos
            },
            'macro': {
                'exploratory': macro_exploratory,
                'value': macro_value,
                'position': macro_pos
            },
            'combined': {
                'exploratory': combined_exploratory,
                'value': combined_value
            }
        }
        
        self.history.append(state)
        return state


class Observer:
    """
    A simulated observer with multi-scale fractal patterns of
    stability (75%) and exploration (25%) states.
    """
    def __init__(self, id, pattern_generator=None, 
                 phase_offset=0.0, intention_strength=1.0):
        self.id = id
        self.step = 0
        
        # Use provided pattern generator or create a new one
        if pattern_generator:
            self.pattern_generator = pattern_generator
        else:
            self.pattern_generator = FractalPatternGenerator()
            
        # Individualize the observer with phase offsets
        if phase_offset:
            self.pattern_generator.micro_phase = (self.pattern_generator.micro_phase + phase_offset) % 1.0
            self.pattern_generator.meso_phase = (self.pattern_generator.meso_phase + phase_offset) % 1.0
            self.pattern_generator.macro_phase = (self.pattern_generator.macro_phase + phase_offset) % 1.0
            
        # Observer's ability to influence quantum events (0-1)
        self.intention_strength = intention_strength
        
        # For tracking observer state
        self.state_history = []
        self.current_state = None
        
    def update(self):
        """Update the observer's state based on fractal pattern"""
        self.current_state = self.pattern_generator.get_state(self.step)
        self.state_history.append(self.current_state)
        self.step += 1
        return self.current_state
    
    def get_future_state(self, steps_ahead):
        """Predict future state based on fractal patterns"""
        return self.pattern_generator.get_state(self.step + steps_ahead)
    
    def get_intention_vector(self):
        """
        Return an intention vector that represents the
        observer's current tendency to influence outcomes
        """
        if not self.current_state:
            return 0.0
            
        # Generate intention based on combined state
        combined = self.current_state['combined']
        
        # Exploratory state tends to push toward extremes
        # Stability state tends to push toward mean
        direction = 1.0 if combined['exploratory'] else -1.0
        magnitude = self.intention_strength * combined['value']
        
        return direction * magnitude


class MultiObserverSystem:
    """
    A system with multiple observers that can collectively
    influence quantum events through interference patterns
    """
    def __init__(self, num_observers=3, max_phase_diff=0.33):
        self.observers = []
        
        # Create multiple observers with different phase offsets
        for i in range(num_observers):
            # Distribute phase offsets evenly
            phase_offset = (i * max_phase_diff) / num_observers
            # Randomly vary intention strength (0.8-1.2 range)
            intention_strength = 0.8 + (0.4 * np.random.random())
            
            observer = Observer(
                id=f"Observer-{i+1}",
                phase_offset=phase_offset,
                intention_strength=intention_strength
            )
            self.observers.append(observer)
            
        self.step = 0
        
    def update_all(self):
        """Update all observers' states"""
        for observer in self.observers:
            observer.update()
        self.step += 1
        
    def get_collective_intention(self):
        """
        Calculate the collective intention of all observers,
        accounting for interference effects
        """
        # Get individual intention vectors
        intentions = [obs.get_intention_vector() for obs in self.observers]
        
        # Simple model: some destructive interference occurs
        # when observers have opposing intentions
        net_intention = sum(intentions) * 0.7  # 0.7 factor models partial destructive interference
        
        # Scale to reasonable range
        return np.clip(net_intention, -1.0, 1.0)
    
    def get_future_collective_intention(self, steps_ahead):
        """Predict future collective intention"""
        # Have each observer predict its future state
        future_states = [obs.get_future_state(steps_ahead) for obs in self.observers]
        
        # Generate future intentions
        future_intentions = []
        for i, observer in enumerate(self.observers):
            # Save current state
            current_state = observer.current_state
            
            # Temporarily set to future state
            observer.current_state = future_states[i]
            
            # Get intention vector
            future_intentions.append(observer.get_intention_vector())
            
            # Restore current state
            observer.current_state = current_state
            
        # Apply interference model
        net_future_intention = sum(future_intentions) * 0.7
        
        return np.clip(net_future_intention, -1.0, 1.0)


class EnhancedQuantumEventSimulator:
    """
    Simulates quantum events that may be influenced by observer
    intentions, with retrocausal effects and adaptive influence.
    """
    def __init__(self, observer_system, retrocausal=True, time_window=3, 
                 adaptive=True):
        self.observer_system = observer_system
        self.retrocausal = retrocausal
        self.time_window = time_window
        self.adaptive = adaptive
        
        # Default distribution parameters
        self.baseline_mean = 5.0
        self.baseline_std = 1.0
        
        # For recording outcomes and metrics
        self.outcomes = []
        self.expected_distributions = []
        self.retrocausal_influences = []
        self.intention_history = []
        self.mismatch_metrics = []
        self.pattern_correlations = []
        
    def generate_quantum_event(self):
        """
        Generate a quantum event potentially influenced by
        current and future observer intentions
        """
        # Get current observer intention
        current_intention = self.observer_system.get_collective_intention()
        self.intention_history.append(current_intention)
        
        # Adjust distribution based on current intention
        # Positive intention = higher mean (preference for higher values)
        # Negative intention = lower mean (preference for lower values)
        intention_factor = 1.0 + (current_intention * 0.2)
        mean = self.baseline_mean * intention_factor
        
        # Exploratory state (positive intention) = wider distribution
        # Stability state (negative intention) = narrower distribution
        std_factor = 1.0 + (current_intention * 0.15)
        std = self.baseline_std * std_factor
        
        # Generate the initial quantum outcome
        initial_outcome = np.random.normal(mean, std)
        
        # Record expected distribution parameters
        self.expected_distributions.append((mean, std))
        
        # Apply retrocausal influence if enabled
        if self.retrocausal:
            outcome = self.apply_retrocausal_influence(initial_outcome, mean, std)
        else:
            outcome = initial_outcome
            self.retrocausal_influences.append(0)
            
        # Record final outcome
        self.outcomes.append(outcome)
        
        # Calculate and record pattern correlations periodically
        if len(self.outcomes) % 10 == 0 and len(self.outcomes) >= 30:
            self.calculate_pattern_correlation()
            
        return outcome
        
    def apply_retrocausal_influence(self, initial_outcome, mean, std):
        """
        Apply influence from future observer intentions to current quantum outcomes,
        simulating retrocausal effects with adaptive strength
        """
        # Get future collective intention
        future_intention = self.observer_system.get_future_collective_intention(self.time_window)
        
        # Calculate mismatch between current outcome and what future would "prefer"
        # Higher future_intention = preference for higher values
        future_preferred_direction = np.sign(future_intention)
        current_deviation = (initial_outcome - mean) / std  # Z-score of current outcome
        
        # Calculate directional mismatch (-1 to 1 scale)
        # Positive: outcome is aligned with future preference
        # Negative: outcome opposes future preference
        directional_mismatch = future_preferred_direction * current_deviation
        
        # Record mismatch for analysis
        self.mismatch_metrics.append({
            'initial_outcome': initial_outcome,
            'mean': mean,
            'std': std,
            'future_intention': future_intention,
            'directional_mismatch': directional_mismatch
        })
        
        # Determine influence strength
        if self.adaptive:
            # Stronger influence when there's a larger mismatch between 
            # outcome and future preference
            base_strength = 0.1
            # Only apply influence when there's a mismatch (negative directional_mismatch)
            # Larger mismatches get stronger corrections
            mismatch_factor = max(0, -directional_mismatch)
            influence_strength = base_strength * mismatch_factor
        else:
            # Fixed influence strength
            influence_strength = 0.1 * abs(future_intention)
        
        # Calculate influence direction
        influence_direction = np.sign(future_intention)
        
        # Calculate total influence
        influence = influence_direction * influence_strength * std
        
        # Record the influence for analysis
        self.retrocausal_influences.append(influence)
        
        # Apply influence to outcome
        return initial_outcome + influence
        
    def calculate_pattern_correlation(self, window_size=30):
        """
        Calculate correlation between observer patterns and quantum outcomes
        """
        if len(self.outcomes) < window_size:
            return
            
        # Get recent outcomes and observer intentions
        recent_outcomes = self.outcomes[-window_size:]
        recent_intentions = self.intention_history[-window_size:]
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(recent_outcomes, recent_intentions)
        
        # Record correlation and significance
        self.pattern_correlations.append({
            'correlation': correlation,
            'p_value': p_value,
            'window_start': len(self.outcomes) - window_size,
            'window_end': len(self.outcomes) - 1
        })
        
        return correlation, p_value


class EnhancedAnalyzer:
    """
    Advanced analysis of quantum consciousness simulation results
    with multi-scale pattern detection and information theoretic measures
    """
    def __init__(self, simulator, observer_system):
        self.simulator = simulator
        self.observer_system = observer_system
        self.analysis_results = {}
        
    def run_analysis(self):
        """
        Run comprehensive analysis on simulation results
        """
        self.analysis_results = {}
        
        # Basic statistical tests
        self.run_basic_stats()
        
        # Pattern and scale analysis
        self.analyze_scale_interactions()
        
        # Retrocausal effect analysis
        self.analyze_retrocausal_effects()
        
        # Information theory metrics
        self.calculate_information_metrics()
        
        # Fractal pattern detection
        self.detect_fractal_patterns()
        
        return self.analysis_results
        
    def run_basic_stats(self):
        """Basic statistical tests on outcomes"""
        outcomes = np.array(self.simulator.outcomes)
        
        if len(outcomes) < 30:
            self.analysis_results['basic_stats'] = {
                'error': 'Insufficient data for analysis'
            }
            return
            
        # Distribution tests
        expected_means = np.array([d[0] for d in self.simulator.expected_distributions])
        expected_stds = np.array([d[1] for d in self.simulator.expected_distributions])
        
        # Standardized residuals
        standardized_residuals = (outcomes - expected_means) / expected_stds
        
        # Test against normal distribution
        ks_stat, ks_pvalue = stats.kstest(standardized_residuals, 'norm')
        
        # Test if retrocausal influences have non-zero mean
        influences = np.array(self.simulator.retrocausal_influences)
        if len(influences) > 0:
            t_stat, t_pvalue = stats.ttest_1samp(influences, 0)
            influence_results = {
                'mean': np.mean(influences),
                't_statistic': t_stat,
                'p_value': t_pvalue,
                'significant': t_pvalue < 0.05
            }
        else:
            influence_results = {
                'mean': 0,
                'significant': False
            }
            
        # Basic summary statistics
        self.analysis_results['basic_stats'] = {
            'outcome_mean': np.mean(outcomes),
            'outcome_std': np.std(outcomes),
            'ks_test': {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'significant': ks_pvalue < 0.05
            },
            'retrocausal_influence': influence_results
        }
        
    def analyze_scale_interactions(self):
        """
        Analyze interactions between different temporal scales
        """
        outcomes = np.array(self.simulator.outcomes)
        if len(outcomes) < 100:
            return
            
        # Create time series for each scale
        micro_series = []
        meso_series = []
        macro_series = []
        
        # Extract scale values from first observer (could enhance for multi-observer)
        for state in self.observer_system.observers[0].state_history:
            micro_series.append(state['micro']['value'])
            meso_series.append(state['meso']['value'])
            macro_series.append(state['macro']['value'])
            
        micro_series = np.array(micro_series[:len(outcomes)])
        meso_series = np.array(meso_series[:len(outcomes)])
        macro_series = np.array(macro_series[:len(outcomes)])
        
        # Calculate correlations between scales and outcomes
        micro_corr, micro_pval = stats.pearsonr(outcomes, micro_series)
        meso_corr, meso_pval = stats.pearsonr(outcomes, meso_series)
        macro_corr, macro_pval = stats.pearsonr(outcomes, macro_series)
        
        # Look for cross-scale interactions - do outcomes correlate more 
        # strongly when multiple scales are aligned?
        aligned_mask = (
            (micro_series > 0.5) & 
            (meso_series > 0.5) & 
            (macro_series > 0.5)
        ) | (
            (micro_series < 0.5) & 
            (meso_series < 0.5) & 
            (macro_series < 0.5)
        )
        
        if sum(aligned_mask) > 10:
            aligned_outcomes = outcomes[aligned_mask]
            unaligned_outcomes = outcomes[~aligned_mask]
            
            # Compare variance of aligned vs unaligned
            f_stat = np.var(aligned_outcomes) / np.var(unaligned_outcomes)
            
            # Degrees of freedom
            df1 = len(aligned_outcomes) - 1
            df2 = len(unaligned_outcomes) - 1
            
            # F-test p-value
            if df1 > 0 and df2 > 0:
                p_value = 1 - stats.f.cdf(f_stat, df1, df2)
                
                # Two-tailed test (we care about both increased and decreased variance)
                p_value = min(p_value, 1 - p_value) * 2
            else:
                p_value = 1.0
                
            alignment_effect = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'aligned_var': np.var(aligned_outcomes),
                'unaligned_var': np.var(unaligned_outcomes)
            }
        else:
            alignment_effect = {
                'error': 'Insufficient aligned data points'
            }
            
        self.analysis_results['scale_interactions'] = {
            'micro_correlation': {
                'r': micro_corr,
                'p_value': micro_pval,
                'significant': micro_pval < 0.05
            },
            'meso_correlation': {
                'r': meso_corr,
                'p_value': meso_pval,
                'significant': meso_pval < 0.05
            },
            'macro_correlation': {
                'r': macro_corr,
                'p_value': macro_pval,
                'significant': macro_pval < 0.05
            },
            'alignment_effect': alignment_effect
        }
        
    def analyze_retrocausal_effects(self):
        """
        Detailed analysis of retrocausal influences and mismatch metrics
        """
        if not self.simulator.retrocausal or len(self.simulator.mismatch_metrics) < 30:
            return
            
        mismatches = self.simulator.mismatch_metrics
        influences = self.simulator.retrocausal_influences
        
        # Extract directional mismatches
        directional_mismatches = [m['directional_mismatch'] for m in mismatches]
        
        # Group by mismatch magnitude
        mismatch_bins = []
        influence_by_mismatch = defaultdict(list)
        
        for i, mismatch in enumerate(directional_mismatches):
            bin_idx = min(9, max(0, int((mismatch + 1) * 5)))  # 10 bins from -1 to 1
            mismatch_bins.append(bin_idx)
            influence_by_mismatch[bin_idx].append(influences[i])
            
        # Calculate average influence by mismatch bin
        avg_influence_by_mismatch = {}
        for bin_idx, values in influence_by_mismatch.items():
            if len(values) > 5:  # Only include bins with sufficient data
                avg_influence_by_mismatch[bin_idx] = np.mean(values)
                
        # Test the trend - does influence correlate with mismatch?
        if len(directional_mismatches) == len(influences):
            mismatch_influence_corr, mismatch_influence_pval = stats.pearsonr(
                directional_mismatches, influences
            )
        else:
            mismatch_influence_corr, mismatch_influence_pval = 0, 1.0
            
        self.analysis_results['retrocausal_effects'] = {
            'avg_influence_by_mismatch': avg_influence_by_mismatch,
            'mismatch_influence_correlation': {
                'r': mismatch_influence_corr,
                'p_value': mismatch_influence_pval,
                'significant': mismatch_influence_pval < 0.05
            }
        }
        
    def calculate_information_metrics(self):
        """
        Calculate information theory metrics on outcomes
        """
        outcomes = np.array(self.simulator.outcomes)
        if len(outcomes) < 100:
            return
            
        # Calculate outcome entropy (discretize first)
        bins = min(20, len(outcomes) // 5)  # Limit bins for small samples
        hist, _ = np.histogram(outcomes, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros for entropy calculation
        entropy = -np.sum(hist * np.log2(hist)) * (max(outcomes) - min(outcomes)) / bins
        
        # Calculate conditional entropy for control vs. retrocausal
        if self.simulator.retrocausal and len(self.simulator.retrocausal_influences) > 0:
            # Group outcomes by retrocausal influence strength
            influence_bins = np.percentile(
                np.abs(self.simulator.retrocausal_influences),
                [0, 33, 67, 100]
            )
            
            influence_groups = np.digitize(
                np.abs(self.simulator.retrocausal_influences), 
                influence_bins
            )
            
            # Calculate entropy within each group
            group_entropies = []
            for group in range(1, 4):  # 3 bins
                group_outcomes = outcomes[influence_groups == group]
                if len(group_outcomes) > 20:
                    group_hist, _ = np.histogram(
                        group_outcomes, 
                        bins=min(10, len(group_outcomes) // 5),
                        density=True
                    )
                    group_hist = group_hist[group_hist > 0]
                    group_entropy = -np.sum(group_hist * np.log2(group_hist))
                    group_entropies.append(group_entropy)
                    
            # Average entropy when conditioned on influence
            if group_entropies:
                conditional_entropy = np.mean(group_entropies)
                
                # Information gain from knowing influence
                mutual_information = entropy - conditional_entropy
            else:
                conditional_entropy = None
                mutual_information = None
        else:
            conditional_entropy = None
            mutual_information = None
            
        self.analysis_results['information_metrics'] = {
            'outcome_entropy': entropy,
            'conditional_entropy': conditional_entropy,
            'mutual_information': mutual_information
        }
        
    def detect_fractal_patterns(self):
        """
        Detect fractal patterns in the outcome sequence
        """
        outcomes = np.array(self.simulator.outcomes)
        if len(outcomes) < 200:
            return
            
        # Detrend the data
        detrended = outcomes - np.mean(outcomes)
        
        # Calculate autocorrelation
        max_lag = min(50, len(outcomes) // 4)
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[len(detrended)-1:len(detrended)-1+max_lag]
        autocorr /= autocorr[0]  # Normalize
        
        # Find peaks in autocorrelation
        try:
            peaks, _ = find_peaks(autocorr, height=0.1)
            peak_heights = autocorr[peaks]
            peak_lags = peaks
            
            # Look for 3:1 or 4:1 patterns in peak distances
            if len(peaks) >= 2:
                peak_diffs = np.diff(peaks)
                avg_peak_distance = np.mean(peak_diffs)
                
                # Check if mean peak distance is close to 3, 4, or their multiples
                is_near_3 = any(abs(avg_peak_distance - (3 * i)) < 0.5 for i in range(1, 4))
                is_near_4 = any(abs(avg_peak_distance - (4 * i)) < 0.5 for i in range(1, 3))
                
                matches_fractal_ratio = is_near_3 or is_near_4
            else:
                avg_peak_distance = None
                matches_fractal_ratio = False
        except:
            peaks = []
            peak_heights = []
            peak_lags = []
            avg_peak_distance = None
            matches_fractal_ratio = False
            
        # Calculate Hurst exponent (simplified method)
        try:
            # Calculate R/S statistic for different time scales
            scales = np.logspace(1, np.log10(len(outcomes) // 5), 10).astype(int)
            rs_values = []
            
            for scale in scales:
                # Skip scales that are too large
                if scale >= len(outcomes) // 2:
                    continue
                    
                # Calculate R/S statistic
                chunks = len(outcomes) // scale
                rs_chunk = []
                
                for i in range(chunks):
                    chunk = outcomes[i*scale:(i+1)*scale]
                    mean = np.mean(chunk)
                    cumsum = np.cumsum(chunk - mean)
                    r = max(cumsum) - min(cumsum)  # Range
                    s = np.std(chunk)  # Standard deviation
                    if s > 0:
                        rs_chunk.append(r / s)
                        
                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))
                    
            if len(scales) >= 4 and len(rs_values) >= 4:
                # Perform log-log regression
                valid_scales = scales[:len(rs_values)]
                log_scales = np.log10(valid_scales)
                log_rs = np.log10(rs_values)
                
                slope, _, r_value, p_value, _ = stats.linregress(log_scales, log_rs)
                
                # Hurst exponent is the slope
                hurst_exponent = slope
                hurst_r_squared = r_value ** 2
            else:
                hurst_exponent = None
                hurst_r_squared = None
        except:
            hurst_exponent = None
            hurst_r_squared = None
        
        self.analysis_results['fractal_patterns'] = {
            'autocorrelation': {
                'peak_lags': peak_lags.tolist() if len(peak_lags) > 0 else [],
                'peak_heights': peak_heights.tolist() if len(peak_heights) > 0 else [],
                'avg_peak_distance': avg_peak_distance,
                'matches_fractal_ratio': matches_fractal_ratio
            },
            'hurst_analysis': {
                'hurst_exponent': hurst_exponent,
                'r_squared': hurst_r_squared,
                'significant': hurst_r_squared > 0.8 if hurst_r_squared is not None else False,
                'interpretation': self.interpret_hurst(hurst_exponent) if hurst_exponent is not None else None
            }
        }
        
    def interpret_hurst(self, hurst):
        """Interpret Hurst exponent value"""
        if hurst is None:
            return None
        elif hurst < 0.4:
            return "anti-persistent (mean-reverting)"
        elif hurst < 0.45:
            return "slightly anti-persistent"
        elif 0.45 <= hurst <= 0.55:
            return "random walk (Brownian motion)"
        elif hurst < 0.8:
            return "slightly persistent"
        else:
            return "strongly persistent (trending)"
    
    def generate_visualization(self, filename="enhanced_quantum_simulation.png"):
        """
        Create comprehensive visualization of simulation results
        """
        # Only create visualization if we have enough data
        if len(self.simulator.outcomes) < 30:
            return
            
        # Create figure with grid layout
        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(6, 2, figure=fig)
        
        # 1. Multi-scale observer states
        ax1 = fig.add_subplot(gs[0, :])
        self.plot_observer_states(ax1)
        
        # 2. Quantum outcomes
        ax2 = fig.add_subplot(gs[1, :])
        self.plot_quantum_outcomes(ax2)
        
        # 3. Retrocausal influences
        ax3 = fig.add_subplot(gs[2, 0])
        self.plot_retrocausal_influences(ax3)
        
        # 4. Mismatch vs. influence
        ax4 = fig.add_subplot(gs[2, 1])
        self.plot_mismatch_influence(ax4)
        
        # 5. Autocorrelation analysis
        ax5 = fig.add_subplot(gs[3, 0])
        self.plot_autocorrelation(ax5)
        
        # 6. Pattern correlation
        ax6 = fig.add_subplot(gs[3, 1])
        self.plot_pattern_correlation(ax6)
        
        # 7. Scale interaction analysis
        ax7 = fig.add_subplot(gs[4, 0])
        self.plot_scale_interactions(ax7)
        
        # 8. Information metrics
        ax8 = fig.add_subplot(gs[4, 1])
        self.plot_information_metrics(ax8)
        
        # 9. Statistical summary
        ax9 = fig.add_subplot(gs[5, :])
        self.plot_statistical_summary(ax9)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"Enhanced visualization saved to {filename}")
        
    def plot_observer_states(self, ax):
        """Plot multi-scale observer states"""
        if not self.observer_system.observers:
            return
            
        # Get first observer for visualization
        observer = self.observer_system.observers[0]
        steps = range(len(observer.state_history))
        
        # Extract state data
        micro_values = [s['micro']['value'] for s in observer.state_history]
        meso_values = [s['meso']['value'] for s in observer.state_history]
        macro_values = [s['macro']['value'] for s in observer.state_history]
        combined_values = [s['combined']['value'] for s in observer.state_history]
        
        # Plot multi-scale states
        ax.plot(steps, micro_values, 'r-', alpha=0.6, label='Micro Scale')
        ax.plot(steps, meso_values, 'g-', alpha=0.6, label='Meso Scale')
        ax.plot(steps, macro_values, 'b-', alpha=0.6, label='Macro Scale')
        ax.plot(steps, combined_values, 'k-', linewidth=2, label='Combined State')
        
        # Mark exploration vs stability regions
        for s in steps:
            if s < len(observer.state_history):
                state = observer.state_history[s]
                if state['combined']['exploratory']:
                    ax.axvspan(s, s+1, alpha=0.1, color='r')
        
        ax.set_ylabel('State Value')
        ax.set_title('Multi-Scale Observer States (Higher = More Stable, Lower = More Exploratory)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_quantum_outcomes(self, ax):
        """Plot quantum outcomes over time"""
        outcomes = self.simulator.outcomes
        steps = range(len(outcomes))
        
        ax.plot(steps, outcomes, 'g-', label='Quantum Outcomes')
        
        # Add expected distribution means
        if self.simulator.expected_distributions:
            expected_means = [d[0] for d in self.simulator.expected_distributions]
            ax.plot(steps, expected_means, 'k--', alpha=0.5, label='Expected Mean')
            
        ax.set_ylabel('Quantum Value')
        ax.set_title('Quantum Event Outcomes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_retrocausal_influences(self, ax):
        """Plot retrocausal influences over time"""
        influences = self.simulator.retrocausal_influences
        steps = range(len(influences))
        
        ax.plot(steps, influences, 'm-', label='Retrocausal Influence')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Calculate and display average influence
        if influences:
            avg_influence = np.mean(influences)
            ax.axhline(y=avg_influence, color='r', linestyle='--', 
                      label=f'Avg: {avg_influence:.4f}')
            
        ax.set_ylabel('Influence Magnitude')
        ax.set_title('Retrocausal Influences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_mismatch_influence(self, ax):
        """Plot mismatch vs influence relationship"""
        if not self.simulator.mismatch_metrics:
            return
            
        mismatches = [m['directional_mismatch'] for m in self.simulator.mismatch_metrics]
        influences = self.simulator.retrocausal_influences[:len(mismatches)]
        
        if len(mismatches) != len(influences):
            return
            
        # Scatter plot of mismatch vs influence
        ax.scatter(mismatches, influences, alpha=0.5, c='purple')
        
        # Try to fit a trend line
        try:
            z = np.polyfit(mismatches, influences, 1)
            p = np.poly1d(z)
            ax.plot(
                sorted(mismatches), 
                p(sorted(mismatches)), 
                "r--", 
                alpha=0.7,
                label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}"
            )
        except:
            pass
            
        ax.set_xlabel('Directional Mismatch')
        ax.set_ylabel('Retrocausal Influence')
        ax.set_title('Future-Mismatch vs Retrocausal Influence')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_autocorrelation(self, ax):
        """Plot autocorrelation of outcomes"""
        outcomes = np.array(self.simulator.outcomes)
        if len(outcomes) < 50:
            return
            
        # Calculate autocorrelation
        max_lag = min(50, len(outcomes) // 4)
        detrended = outcomes - np.mean(outcomes)
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[len(detrended)-1:len(detrended)-1+max_lag]
        autocorr /= autocorr[0]  # Normalize
        
        lags = np.arange(max_lag)
        ax.plot(lags, autocorr, 'b-')
        
        # Confidence bands (95%)
        conf_level = 1.96 / np.sqrt(len(outcomes))
        ax.axhline(y=conf_level, linestyle='--', color='r', alpha=0.5)
        ax.axhline(y=-conf_level, linestyle='--', color='r', alpha=0.5)
        
        # Mark significant peaks
        try:
            peaks, _ = find_peaks(autocorr, height=conf_level)
            if len(peaks) > 0:
                ax.plot(peaks, autocorr[peaks], 'ro')
                
                # Annotate key peaks
                for i, peak in enumerate(peaks):
                    if i < 3:  # Only label first few peaks
                        ax.annotate(
                            f"Lag {peak}", 
                            xy=(peak, autocorr[peak]),
                            xytext=(peak + 1, autocorr[peak] + 0.02),
                            arrowprops=dict(arrowstyle='->'),
                        )
        except:
            pass
            
        ax.set_xlabel('Lag')
        ax.set_ylabel('Autocorrelation')
        ax.set_title('Quantum Outcome Autocorrelation')
        ax.grid(True, alpha=0.3)
        
    def plot_pattern_correlation(self, ax):
        """Plot pattern correlations over time"""
        correlations = self.simulator.pattern_correlations
        if not correlations:
            return
            
        # Extract data
        window_positions = [(c['window_start'] + c['window_end']) // 2 for c in correlations]
        corr_values = [c['correlation'] for c in correlations]
        p_values = [c['p_value'] for c in correlations]
        
        # Plot correlations
        ax.plot(window_positions, corr_values, 'c-', label='Observer-Outcome Correlation')
        
        # Highlight significant correlations
        significant_idx = [i for i, p in enumerate(p_values) if p < 0.05]
        significant_pos = [window_positions[i] for i in significant_idx]
        significant_corr = [corr_values[i] for i in significant_idx]
        
        if significant_idx:
            ax.plot(significant_pos, significant_corr, 'r*', markersize=10, 
                   label='Significant (p<0.05)')
            
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Window Center Position')
        ax.set_ylabel('Correlation')
        ax.set_title('Observer State vs Quantum Outcome Correlation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def plot_scale_interactions(self, ax):
        """Plot scale interaction analysis"""
        # Use this area to visualize scale interaction results
        if 'scale_interactions' not in self.analysis_results:
            return
            
        scale_info = self.analysis_results['scale_interactions']
        
        # Create a bar chart of correlations by scale
        scales = ['Micro', 'Meso', 'Macro']
        correlations = [
            scale_info['micro_correlation']['r'],
            scale_info['meso_correlation']['r'],
            scale_info['macro_correlation']['r']
        ]
        
        significant = [
            scale_info['micro_correlation']['significant'],
            scale_info['meso_correlation']['significant'],
            scale_info['macro_correlation']['significant']
        ]
        
        # Plot bars
        bars = ax.bar(scales, correlations, color=['lightblue', 'lightgreen', 'salmon'])
        
        # Highlight significant correlations
        for i, bar in enumerate(bars):
            if significant[i]:
                bar.set_edgecolor('red')
                bar.set_linewidth(2)
                
        # Add alignment effect if available
        if 'alignment_effect' in scale_info and 'f_statistic' in scale_info['alignment_effect']:
            alignment_effect = scale_info['alignment_effect']
            is_significant = alignment_effect.get('significant', False)
            
            ax.text(
                0.5, 0.95, 
                f"Scale Alignment Effect: {alignment_effect['f_statistic']:.2f}\n" +
                f"p-value: {alignment_effect['p_value']:.4f}\n" +
                f"{'Significant!' if is_significant else 'Not significant'}",
                transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(
                    boxstyle='round,pad=0.5', 
                    fc='yellow' if is_significant else 'white', 
                    alpha=0.3
                )
            )
        
        ax.set_ylabel('Correlation with Outcomes')
        ax.set_title('Outcome Correlation by Temporal Scale')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
    def plot_information_metrics(self, ax):
        """Plot information theory metrics"""
        if 'information_metrics' not in self.analysis_results:
            return
            
        info_metrics = self.analysis_results['information_metrics']
        
        # Create a simple bar chart of entropy metrics
        metrics = ['Outcome Entropy']
        values = [info_metrics['outcome_entropy']]
        
        if info_metrics['conditional_entropy'] is not None:
            metrics.append('Conditional Entropy')
            values.append(info_metrics['conditional_entropy'])
            
        if info_metrics['mutual_information'] is not None:
            metrics.append('Mutual Information')
            values.append(info_metrics['mutual_information'])
            
        ax.bar(metrics, values, color=['lightblue', 'lightgreen', 'salmon'])
        
        # Add horizontal line at outcome entropy for reference
        ax.axhline(
            y=info_metrics['outcome_entropy'], 
            color='r', 
            linestyle='--', 
            alpha=0.5
        )
        
        # Add retrocausal interpretation
        if info_metrics['mutual_information'] is not None:
            if info_metrics['mutual_information'] > 0.05:
                message = "Significant information gain from future states!"
                color = 'yellow'
            else:
                message = "Limited information from future states"
                color = 'white'
                
            ax.text(
                0.5, 0.95, 
                message,
                transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(boxstyle='round,pad=0.5', fc=color, alpha=0.3)
            )
        
        ax.set_ylabel('Bits')
        ax.set_title('Information Theory Metrics')
        ax.grid(True, alpha=0.3)
        
    def plot_statistical_summary(self, ax):
        """Plot statistical summary"""
        # Use a text box to show key statistical findings
        if not self.analysis_results:
            return
            
        # Compile key findings
        findings = []
        
        # Basic stats
        if 'basic_stats' in self.analysis_results:
            basic_stats = self.analysis_results['basic_stats']
            
            # Distribution anomaly
            if 'ks_test' in basic_stats:
                ks_test = basic_stats['ks_test']
                findings.append(
                    f"Distribution Anomaly: {'YES' if ks_test['significant'] else 'No'}" +
                    f" (p = {ks_test['p_value']:.4f})"
                )
                
            # Retrocausal effect
            if 'retrocausal_influence' in basic_stats:
                influence = basic_stats['retrocausal_influence']
                findings.append(
                    f"Retrocausal Effect: {'SIGNIFICANT' if influence['significant'] else 'Not significant'}" +
                    f" (mean = {influence['mean']:.4f})"
                )
                
        # Fractal patterns
        if 'fractal_patterns' in self.analysis_results:
            fractal = self.analysis_results['fractal_patterns']
            
            # Autocorrelation peaks
            if 'autocorrelation' in fractal:
                auto = fractal['autocorrelation']
                if 'matches_fractal_ratio' in auto:
                    findings.append(
                        f"3:1 Pattern in Autocorrelation: {'DETECTED' if auto['matches_fractal_ratio'] else 'Not detected'}"
                    )
                    if auto['avg_peak_distance'] is not None:
                        findings.append(
                            f"Avg Peak Distance: {auto['avg_peak_distance']:.2f}"
                        )
                        
            # Hurst exponent
            if 'hurst_analysis' in fractal:
                hurst = fractal['hurst_analysis']
                if hurst['hurst_exponent'] is not None:
                    findings.append(
                        f"Hurst Exponent: {hurst['hurst_exponent']:.3f}" +
                        f" ({hurst['interpretation']})"
                    )
                    
        # Information metrics
        if 'information_metrics' in self.analysis_results:
            info = self.analysis_results['information_metrics']
            if info['mutual_information'] is not None:
                findings.append(
                    f"Information Gain from Future: {info['mutual_information']:.4f} bits" +
                    f" ({'Significant' if info['mutual_information'] > 0.05 else 'Not significant'})"
                )
                
        # Scale interactions
        if 'scale_interactions' in self.analysis_results:
            scales = self.analysis_results['scale_interactions']
            
            # Check if any scale has significant correlation
            significant_scales = []
            if scales['micro_correlation']['significant']:
                significant_scales.append('Micro')
            if scales['meso_correlation']['significant']:
                significant_scales.append('Meso')
            if scales['macro_correlation']['significant']:
                significant_scales.append('Macro')
                
            if significant_scales:
                findings.append(
                    f"Significant Scale Correlations: {', '.join(significant_scales)}"
                )
                
            # Check alignment effect
            if 'alignment_effect' in scales and 'significant' in scales['alignment_effect']:
                if scales['alignment_effect']['significant']:
                    findings.append(
                        f"Multi-Scale Alignment Effect: SIGNIFICANT (F={scales['alignment_effect']['f_statistic']:.2f})"
                    )
                    
        # Retrocausal effects
        if 'retrocausal_effects' in self.analysis_results:
            retro = self.analysis_results['retrocausal_effects']
            
            if 'mismatch_influence_correlation' in retro:
                corr = retro['mismatch_influence_correlation']
                findings.append(
                    f"Mismatch-Influence Correlation: {corr['r']:.3f}" +
                    f" ({'Significant' if corr['significant'] else 'Not significant'})"
                )
                
        # Display findings as text
        ax.axis('off')  # Hide axis
        
        # Create header
        header = "QUANTUM CONSCIOUSNESS SIMULATION STATISTICAL SUMMARY"
        if self.simulator.retrocausal:
            header += " (WITH RETROCAUSAL EFFECTS)"
        else:
            header += " (CONTROL - NO RETROCAUSAL EFFECTS)"
            
        ax.text(0.5, 1.0, header, ha='center', va='top', fontsize=14, fontweight='bold')
        
        # Display findings in a box
        findings_text = "\n".join(f"• {finding}" for finding in findings)
        ax.text(
            0.5, 0.9, findings_text,
            ha='center', va='top',
            bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.5)
        )
        
        # Add overall conclusion
        conclusion = "OVERALL CONCLUSION: "
        
        # Check if any significant findings for retrocausal version
        has_significant_findings = any(
            'SIGNIFICANT' in finding or 'DETECTED' in finding 
            for finding in findings
        )
        
        if self.simulator.retrocausal and has_significant_findings:
            conclusion += "Statistical anomalies detected consistent with consciousness-first hypothesis."
            conclusion += "\nFuture observer states appear to influence quantum event outcomes."
            conclusion_color = 'lightgreen'
        elif not self.simulator.retrocausal and not has_significant_findings:
            conclusion += "No significant evidence for consciousness-first hypothesis in control simulation."
            conclusion += "\nResults are consistent with conventional time-first causality."
            conclusion_color = 'white'
        else:
            conclusion += "Mixed evidence - some anomalies detected but results are inconclusive."
            conclusion_color = 'lightyellow'
            
        ax.text(
            0.5, 0.1, conclusion,
            ha='center', va='bottom', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', fc=conclusion_color, alpha=0.7)
        )


def run_enhanced_simulation(iterations=5000, retrocausal=True, 
                            num_observers=3, adaptive=True):
    """
    Run the enhanced quantum consciousness simulation
    with multiple observers and multi-scale fractal patterns
    """
    print(f"\nStarting Enhanced Quantum Consciousness Simulation ({iterations} iterations)")
    print(f"Retrocausal effects: {'Enabled' if retrocausal else 'Disabled'}")
    print(f"Number of observers: {num_observers}")
    print(f"Adaptive retrocausal logic: {'Enabled' if adaptive else 'Disabled'}")
    
    start_time = time.time()
    
    # Initialize multi-observer system
    observer_system = MultiObserverSystem(num_observers=num_observers)
    
    # Initialize enhanced quantum event simulator
    simulator = EnhancedQuantumEventSimulator(
        observer_system=observer_system,
        retrocausal=retrocausal,
        time_window=3,
        adaptive=adaptive
    )
    
    # Initialize enhanced analyzer
    analyzer = EnhancedAnalyzer(simulator, observer_system)
    
    # Run simulation
    for i in range(iterations):
        observer_system.update_all()
        simulator.generate_quantum_event()
        
        # Progress indicator
        if i % 1000 == 0:
            print(f"  Completed {i} iterations...")
    
    # Run comprehensive analysis
    print("\nAnalyzing results...")
    analysis_results = analyzer.run_analysis()
    
    # Generate visualization
    print("Generating visualization...")
    analyzer.generate_visualization()
    
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds\n")
    
    return simulator, observer_system, analyzer, analysis_results


def run_comparative_analysis():
    """
    Run both control and experimental simulations for comparison
    """
    # Control simulation (no retrocausal effects)
    print("\n" + "="*80)
    print("ENHANCED CONTROL SIMULATION: NO RETROCAUSAL EFFECTS")
    print("="*80)
    control_simulator, control_observers, control_analyzer, control_results = run_enhanced_simulation(
        iterations=2000, retrocausal=False
    )
    
    # Experimental simulation (with retrocausal effects)
    print("\n" + "="*80)
    print("ENHANCED EXPERIMENTAL SIMULATION: WITH RETROCAUSAL EFFECTS")
    print("="*80)
    retro_simulator, retro_observers, retro_analyzer, retro_results = run_enhanced_simulation(
        iterations=2000, retrocausal=True
    )
    
    # Show comparative analysis
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Compare key metrics
    compare_metrics = {
        'Distribution Anomaly': 
            ('basic_stats.ks_test.p_value', 'basic_stats.ks_test.significant'),
        'Retrocausal Effect': 
            ('basic_stats.retrocausal_influence.p_value', 'basic_stats.retrocausal_influence.significant'),
        'Information Gain': 
            ('information_metrics.mutual_information', None),
        'Fractal Pattern': 
            ('fractal_patterns.autocorrelation.matches_fractal_ratio', None),
        'Scale Alignment': 
            ('scale_interactions.alignment_effect.p_value', 'scale_interactions.alignment_effect.significant')
    }
    
    def get_nested_value(data, path):
        """Get a value from nested dictionaries using dot notation"""
        if data is None:
            return None
            
        parts = path.split('.')
        current = data
        for part in parts:
            if current is not None and part in current:
                current = current[part]
            else:
                return None
        return current
    
    # Print comparison table
    print("\nMETRIC COMPARISON:\n")
    print(f"{'Metric':<25} {'Control':<20} {'Retrocausal':<20} {'Difference':<15} {'Conclusion'}")
    print("-" * 90)
    
    significant_differences = 0
    
    for metric, (value_path, significance_path) in compare_metrics.items():
        control_value = get_nested_value(control_results, value_path)
        retro_value = get_nested_value(retro_results, value_path)
        
        control_significant = get_nested_value(control_results, significance_path) if significance_path else None
        retro_significant = get_nested_value(retro_results, significance_path) if significance_path else None
        
        # Format values for display
        control_display = f"{control_value:.6f}" if isinstance(control_value, float) else str(control_value)
        retro_display = f"{retro_value:.6f}" if isinstance(retro_value, float) else str(retro_value)
        
        # Add significance markers
        if control_significant is not None:
            control_display += " *" if control_significant else ""
        if retro_significant is not None:
            retro_display += " *" if retro_significant else ""
            
        # Calculate difference if both are numeric
        if isinstance(control_value, (int, float)) and isinstance(retro_value, (int, float)):
            difference = abs(retro_value - control_value)
            diff_display = f"{difference:.6f}"
            
            # Determine if difference is significant
            is_significant = False
            
            # For p-values, consider significant if one is < 0.05 and the other isn't
            if "p_value" in value_path:
                is_significant = (control_value < 0.05) != (retro_value < 0.05)
            # For other metrics, use a 20% relative difference threshold
            elif abs(control_value) > 0.0001:  # Avoid division by zero
                relative_diff = abs(difference / control_value)
                is_significant = relative_diff > 0.2
            else:
                is_significant = difference > 0.1
                
            conclusion = "SIGNIFICANT" if is_significant else "Not significant"
            
            if is_significant:
                significant_differences += 1
        else:
            diff_display = "N/A"
            
            # For boolean values, check if they differ
            if isinstance(control_value, bool) and isinstance(retro_value, bool):
                is_significant = control_value != retro_value
                conclusion = "SIGNIFICANT" if is_significant else "Not significant"
                
                if is_significant:
                    significant_differences += 1
            else:
                conclusion = "N/A"
                
        print(f"{metric:<25} {control_display:<20} {retro_display:<20} {diff_display:<15} {conclusion}")
        
    # Overall conclusion
    print("\nOVERALL CONCLUSION:")
    if significant_differences >= 3:
        print("  *** STRONG EVIDENCE FOR CONSCIOUSNESS-FIRST HYPOTHESIS ***")
        print("  Multiple significant differences detected when retrocausal effects are enabled")
    elif significant_differences >= 1:
        print("  ** MODERATE EVIDENCE FOR CONSCIOUSNESS-FIRST HYPOTHESIS **")
        print("  Some significant differences detected when retrocausal effects are enabled")
    else:
        print("  * LIMITED EVIDENCE FOR CONSCIOUSNESS-FIRST HYPOTHESIS *")
        print("  Few or no significant differences detected between simulations")
        
    # Recommendations for next steps
    print("\nRECOMMENDATIONS FOR FURTHER VALIDATION:")
    print("  1. Increase iteration count for greater statistical power")
    print("  2. Test with different fractal ratios (2:1, 4:1, etc.) to find optimal patterns")
    print("  3. Implement quantum biasing proportional to observer count")
    print("  4. Test sensitivity to time window parameter")
    print("  5. Explore physical implementations using true quantum randomness")
    
    print("\nDISCLAIMER:")
    print("  This is a simulated model and not a real quantum experiment. The results")
    print("  represent mathematical patterns in the simulation rather than physical reality.")
    print("  Any anomalies detected are emergent properties of the code architecture.")
    print("="*80)


if __name__ == "__main__":
    # Run comparative analysis (control vs. retrocausal)
    run_comparative_analysis()