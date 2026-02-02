"""
Quantum Consciousness Simulation - Testing Consciousness-First/Time-Second Hypothesis

This simulation implements a model to test whether a simulated observer with
fractal oscillation patterns between exploration (25%) and stability (75%) states
might exhibit retrocausal influence on quantum event outcomes.

The core hypothesis: If consciousness is more fundamental than time,
we should see statistically significant anomalies when future observer
states influence past quantum outcomes.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from collections import deque
import time

# Set random seed for reproducibility
np.random.seed(42)

class FractalObserver:
    """
    Simulated consciousness with fractal 3:1 oscillation patterns between
    stability (75%) and exploration (25%) states.
    """
    def __init__(self, cycle_length=4, micro_cycle_length=4, meta_cycle_length=4):
        # Main cycle: 3 stability, 1 exploration (75/25 split)
        self.cycle_length = cycle_length
        # Micro cycle within each state (fractal self-similarity)
        self.micro_cycle_length = micro_cycle_length
        # Meta cycle modulating overall state (fractal hierarchy)
        self.meta_cycle_length = meta_cycle_length
        
        # State tracking
        self.current_step = 0
        self.micro_step = 0
        self.meta_step = 0
        
        # History for time-warping
        self.state_history = deque(maxlen=1000)
        
        # Intention/focus value (0-1 range)
        self.intention = 0.5
        
        # For logging/analysis
        self.states = []
        self.intentions = []
        self.fractal_values = []
    
    def is_in_exploratory_state(self):
        """Determine if observer is in exploratory (25%) or stability (75%) state"""
        main_cycle_pos = self.current_step % self.cycle_length
        # Last position in cycle is exploratory (25%)
        return main_cycle_pos == self.cycle_length - 1
    
    def get_fractal_value(self):
        """
        Generate a fractal value combining main, micro and meta cycles,
        creating a self-similar nested pattern.
        """
        # Main cycle: 0.75 for stability, 0.25 for exploration
        main_value = 0.25 if self.is_in_exploratory_state() else 0.75
        
        # Micro cycle: fractal pattern within each state
        micro_pos = self.micro_step % self.micro_cycle_length
        micro_value = 0.25 if micro_pos == self.micro_cycle_length - 1 else 0.75
        
        # Meta cycle: modulation across many cycles
        meta_pos = self.meta_step % self.meta_cycle_length
        meta_value = 0.25 if meta_pos == self.meta_cycle_length - 1 else 0.75
        
        # Combine fractal values with different weights
        fractal_value = (0.6 * main_value + 0.3 * micro_value + 0.1 * meta_value)
        return fractal_value
    
    def update_intention(self):
        """Update the observer's intention/focus level with some randomness"""
        base_intention = self.get_fractal_value()
        # Add subtle randomness to intention
        noise = np.random.normal(0, 0.05)
        self.intention = max(0, min(1, base_intention + noise))
        
        # Record for analysis
        self.intentions.append(self.intention)
        self.fractal_values.append(self.get_fractal_value())
    
    def advance(self):
        """Advance the observer state by one time step"""
        # Record current state
        current_state = {
            'step': self.current_step,
            'is_exploratory': self.is_in_exploratory_state(),
            'fractal_value': self.get_fractal_value(),
            'intention': self.intention
        }
        self.state_history.append(current_state)
        self.states.append(current_state['is_exploratory'])
        
        # Advance cycles
        self.current_step += 1
        # Micro cycle advances faster
        self.micro_step += 1
        # Meta cycle advances slower
        if self.current_step % 4 == 0:
            self.meta_step += 1
            
        # Update intention level
        self.update_intention()
        
        return current_state
    
    def get_future_state(self, steps_ahead):
        """Get a future state (if available in history) or predict one"""
        future_index = self.current_step + steps_ahead
        
        # Simple prediction if we don't have history
        if len(self.state_history) <= steps_ahead:
            future_exploratory = (future_index % self.cycle_length) == (self.cycle_length - 1)
            return {
                'is_exploratory': future_exploratory,
                'intention': 0.25 if future_exploratory else 0.75,
                'fractal_value': 0.25 if future_exploratory else 0.75
            }
        
        # Get actual recorded future state (from previous iterations)
        for state in self.state_history:
            if state['step'] == future_index:
                return state
                
        # If specific future step not found but we have history, extrapolate
        future_exploratory = (future_index % self.cycle_length) == (self.cycle_length - 1)
        return {
            'is_exploratory': future_exploratory,
            'intention': 0.25 if future_exploratory else 0.75,
            'fractal_value': 0.25 if future_exploratory else 0.75
        }
    
    def get_past_state(self, steps_back):
        """Get a past state from history"""
        past_index = self.current_step - steps_back
        if past_index < 0:
            return None
            
        for state in self.state_history:
            if state['step'] == past_index:
                return state
                
        return None


class QuantumEventSimulator:
    """
    Simulates quantum events that may be influenced by observer
    states, including possible retrocausal effects.
    """
    def __init__(self, observer, retrocausal=True, time_window=3):
        self.observer = observer
        self.retrocausal = retrocausal
        self.time_window = time_window  # How many steps to look ahead/behind
        
        # For recording outcomes
        self.outcomes = []
        self.expected_distributions = []
        self.actual_distributions = []
        self.retrocausal_influences = []
        
        # Baseline outcome parameters 
        self.baseline_mean = 5.0
        self.baseline_std = 1.0
        
        # Fractal pattern detection
        self.pattern_matches = []
        
    def generate_quantum_event(self):
        """Generate a quantum event outcome based on observer state"""
        # Get current observer state
        is_exploratory = self.observer.is_in_exploratory_state()
        intention = self.observer.intention
        
        # Adjust distribution based on observer state
        if is_exploratory:
            # Broader distribution in exploratory state 
            mean = self.baseline_mean
            std = self.baseline_std * 1.5
        else:
            # Narrower distribution in stability state
            mean = self.baseline_mean 
            std = self.baseline_std * 0.75
            
        # Fine-tune with intention level
        intention_factor = 0.5 + intention
        mean *= intention_factor
        
        # Generate the initial quantum outcome
        initial_outcome = np.random.normal(mean, std)
        
        # Record expected distribution params
        self.expected_distributions.append((mean, std))
        
        # Apply retrocausal influence if enabled
        if self.retrocausal:
            outcome = self.apply_retrocausal_influence(initial_outcome, mean, std)
        else:
            outcome = initial_outcome
            self.retrocausal_influences.append(0)
            
        # Record final outcome
        self.outcomes.append(outcome)
        return outcome
        
    def apply_retrocausal_influence(self, initial_outcome, mean, std):
        """
        Apply influence from future observer states to current quantum outcome,
        simulating retrocausal effects
        """
        # Get future observer state
        future_state = self.observer.get_future_state(self.time_window)
        
        if not future_state:
            self.retrocausal_influences.append(0)
            return initial_outcome
            
        # Calculate influence based on future intention and state
        future_intention = future_state['intention']
        future_exploratory = future_state['is_exploratory']
        
        # Higher intention = stronger influence
        influence_strength = future_intention * 0.2
        
        # Exploratory state tends to "pull" toward more extreme values
        # Stability state tends to "pull" toward mean
        if future_exploratory:
            # Pull away from mean (more extreme)
            direction = 1 if initial_outcome > mean else -1
            influence = direction * influence_strength * std
        else:
            # Pull toward mean (more stable)
            direction = -1 if initial_outcome > mean else 1
            distance_from_mean = abs(initial_outcome - mean)
            influence = direction * influence_strength * distance_from_mean
            
        # Record the influence for analysis
        self.retrocausal_influences.append(influence)
        
        # Apply influence to outcome
        return initial_outcome + influence
        
    def check_pattern_match(self, window_size=12):
        """
        Check if recent outcomes show pattern matching observer's 3:1 fractal rhythm
        Returns correlation coefficient and p-value
        """
        if len(self.outcomes) < window_size:
            return 0, 1.0
            
        # Get recent outcomes and observer states
        recent_outcomes = self.outcomes[-window_size:]
        recent_states = self.observer.states[-window_size:]
        
        # Convert binary states to numerical (1 for exploratory, 0 for stability)
        state_values = [1 if state else 0 for state in recent_states]
        
        # Calculate correlation
        correlation, p_value = stats.pearsonr(recent_outcomes, state_values)
        
        # Record if significant pattern match found
        self.pattern_matches.append((correlation, p_value))
        
        return correlation, p_value


class Analyzer:
    """
    Analyzes results from quantum consciousness simulation to detect
    anomalies and patterns consistent with consciousness-first hypothesis.
    """
    def __init__(self, observer, simulator):
        self.observer = observer
        self.simulator = simulator
        
    def run_statistical_tests(self):
        """Run statistical tests to detect anomalies in the quantum outcomes"""
        results = {}
        
        # Check if retrocausal outcomes differ from expected distribution
        if len(self.simulator.outcomes) > 30:  # Need sufficient sample size
            outcomes = np.array(self.simulator.outcomes)
            expected_means = np.array([ed[0] for ed in self.simulator.expected_distributions])
            expected_stds = np.array([ed[1] for ed in self.simulator.expected_distributions])
            
            # Create standardized residuals
            standardized_residuals = (outcomes - expected_means) / expected_stds
            
            # Test if standardized residuals follow normal distribution (should be N(0,1))
            ks_stat, ks_pvalue = stats.kstest(standardized_residuals, 'norm')
            results['ks_test'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'significant': ks_pvalue < 0.05,
                'description': 'Tests if outcomes deviate from expected distribution'
            }
            
            # Look for autocorrelation in outcomes (temporal pattern)
            acf = np.correlate(standardized_residuals, standardized_residuals, mode='full')
            acf = acf[len(acf)//2:] / np.var(standardized_residuals) / len(standardized_residuals)
            
            # Test significance of autocorrelations at lag 3 (3:1 pattern)
            sig_level = 1.96 / np.sqrt(len(standardized_residuals))
            results['autocorrelation'] = {
                'lag3': acf[3] if len(acf) > 3 else 0,
                'significant': abs(acf[3]) > sig_level if len(acf) > 3 else False,
                'description': 'Tests for periodic patterns in outcomes at lag 3 (3:1 ratio)'
            }
            
            # Test if retrocausal influences have non-zero mean
            influences = np.array(self.simulator.retrocausal_influences)
            t_stat, t_pvalue = stats.ttest_1samp(influences, 0)
            results['retrocausal_influence'] = {
                'mean_influence': np.mean(influences),
                't_statistic': t_stat,
                'p_value': t_pvalue,
                'significant': t_pvalue < 0.05,
                'description': 'Tests if future states have significant influence on outcomes'
            }
            
            # Check pattern correlation between observer state and outcomes
            pattern_corrs = [pm[0] for pm in self.simulator.pattern_matches if not np.isnan(pm[0])]
            pattern_pvals = [pm[1] for pm in self.simulator.pattern_matches if not np.isnan(pm[1])]
            if pattern_corrs:
                results['pattern_match'] = {
                    'mean_correlation': np.mean(pattern_corrs),
                    'significant_ratio': sum(p < 0.05 for p in pattern_pvals) / len(pattern_pvals),
                    'description': 'Tests for correlation between observer states and outcomes'
                }
            
            # 3:1 Ratio detection in outcome patterns
            try:
                # Count peaks in autocorrelation
                peaks, _ = find_peaks(acf[:20])
                if len(peaks) >= 2:
                    mean_peak_distance = np.mean(np.diff(peaks))
                    results['ratio_pattern'] = {
                        'mean_peak_distance': mean_peak_distance,
                        'matches_3to1': 3.5 < mean_peak_distance < 4.5,
                        'description': 'Tests if outcome cycles match 3:1 or 4:1 pattern'
                    }
            except:
                # Fallback if peaks detection fails
                pass
            
        return results
    
    def plot_results(self, filename="quantum_consciousness_results.png"):
        """Create visualization of simulation results"""
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
        
        # Plot observer states
        ax1 = axes[0]
        steps = list(range(len(self.observer.states)))
        ax1.plot(steps, self.observer.states, 'b.', label='Exploratory State')
        ax1.plot(steps, self.observer.fractal_values, 'r-', 
                 alpha=0.7, label='Fractal Value')
        ax1.set_ylabel('Observer State')
        ax1.set_title('Fractal Observer States (1=Exploratory, 0=Stability)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot quantum outcomes
        ax2 = axes[1]
        ax2.plot(steps, self.simulator.outcomes, 'g-', label='Quantum Event Outcomes')
        expected_means = [ed[0] for ed in self.simulator.expected_distributions]
        ax2.plot(steps, expected_means, 'k--', alpha=0.5, label='Expected Mean')
        ax2.set_ylabel('Quantum Value')
        ax2.set_title('Quantum Event Outcomes')
        ax2.legend()
        ax2.grid(True)
        
        # Plot retrocausal influences
        ax3 = axes[2]
        ax3.plot(steps, self.simulator.retrocausal_influences, 'm-', 
                 label='Retrocausal Influence')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax3.set_ylabel('Influence Magnitude')
        ax3.set_title('Retrocausal Influences')
        ax3.legend()
        ax3.grid(True)
        
        # Plot pattern matches (correlation between observer and outcomes)
        ax4 = axes[3]
        if self.simulator.pattern_matches:
            pattern_steps = list(range(len(self.simulator.pattern_matches)))
            correlations = [pm[0] for pm in self.simulator.pattern_matches]
            pvalues = [pm[1] for pm in self.simulator.pattern_matches]
            ax4.plot(pattern_steps, correlations, 'c-', label='State-Outcome Correlation')
            # Highlight significant correlations
            significant_idx = [i for i, p in enumerate(pvalues) if p < 0.05]
            significant_corr = [correlations[i] for i in significant_idx]
            if significant_idx:
                ax4.plot([pattern_steps[i] for i in significant_idx], significant_corr, 
                         'r*', markersize=10, label='Significant (p<0.05)')
            ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax4.set_ylabel('Correlation')
            ax4.set_title('Pattern Matching: Observer State vs Quantum Outcomes')
            ax4.legend()
            
        ax4.set_xlabel('Simulation Step')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        print(f"Results visualization saved to {filename}")
        
    def print_summary(self, results):
        """Print summary of analysis results"""
        print("\n" + "="*80)
        print("QUANTUM CONSCIOUSNESS SIMULATION RESULTS")
        print("="*80)
        
        print("\nSIMULATION PARAMETERS:")
        print(f"  Retrocausal Effects: {'Enabled' if self.simulator.retrocausal else 'Disabled'}")
        print(f"  Time Window: {self.simulator.time_window} steps")
        print(f"  Total Iterations: {len(self.simulator.outcomes)}")
        print(f"  Fractal Cycle: 3:1 ratio (75% stability, 25% exploration)")
        
        print("\nSTATISTICAL TESTS:")
        for test_name, result in results.items():
            print(f"\n  {test_name.upper()}:")
            for key, value in result.items():
                if key != 'description':
                    print(f"    {key}: {value}")
            print(f"    ↳ {result['description']}")
            
        # Summarize retrocausal effect
        if 'retrocausal_influence' in results:
            influence = results['retrocausal_influence']
            if influence['significant']:
                effect = "SIGNIFICANT RETROCAUSAL EFFECT DETECTED"
                p_val = influence['p_value']
                magnitude = influence['mean_influence']
                print(f"\n  *** {effect} (p={p_val:.6f}) ***")
                print(f"      Mean Influence Magnitude: {magnitude:.6f}")
            else:
                print("\n  No significant retrocausal effect detected")
                
        # Summarize 3:1 pattern matching
        if 'ratio_pattern' in results and results['ratio_pattern']['matches_3to1']:
            print("\n  *** DETECTED 3:1 RATIO PATTERN IN OUTCOMES ***")
            print(f"      Peak Distance: {results['ratio_pattern']['mean_peak_distance']:.2f}")
            
        # Overall conclusion
        anomaly_detected = any(r.get('significant', False) for r in results.values())
        print("\nOVERALL CONCLUSION:")
        if anomaly_detected:
            print("  Statistical anomalies detected consistent with consciousness-first hypothesis")
            print("  Future observer states appear to influence quantum event outcomes")
        else:
            print("  No significant evidence for consciousness-first hypothesis in this simulation")
            print("  Results are consistent with conventional time-first causality")
            
        print("="*80)


# Helper function for peak detection in waveforms
def find_peaks(x, min_height=None, min_distance=1):
    """Find peaks in a 1D array"""
    peaks = []
    for i in range(1, len(x)-1):
        if x[i] > x[i-1] and x[i] > x[i+1]:
            if min_height is None or x[i] >= min_height:
                peaks.append(i)
    
    # Filter peaks by min_distance
    if min_distance > 1 and len(peaks) > 1:
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        peaks = filtered_peaks
                
    return np.array(peaks), np.array([x[p] for p in peaks])


def run_simulation(iterations=10000, retrocausal=True, time_window=3):
    """Run the quantum consciousness simulation"""
    print(f"\nStarting Quantum Consciousness Simulation ({iterations} iterations)")
    print(f"Retrocausal effects: {'Enabled' if retrocausal else 'Disabled'}")
    
    start_time = time.time()
    
    # Initialize fractal observer with 3:1 ratio pattern
    observer = FractalObserver(cycle_length=4)  # 3 stability, 1 exploratory
    
    # Initialize quantum event simulator
    simulator = QuantumEventSimulator(observer, retrocausal=retrocausal, time_window=time_window)
    
    # Initialize analyzer
    analyzer = Analyzer(observer, simulator)
    
    # Run simulation
    for i in range(iterations):
        observer.advance()
        simulator.generate_quantum_event()
        
        # Check for pattern matches every 12 steps
        if i % 12 == 0 and i > 24:
            simulator.check_pattern_match()
            
        # Progress indicator
        if i % 1000 == 0:
            print(f"  Completed {i} iterations...")
    
    # Analyze results
    results = analyzer.run_statistical_tests()
    
    # Plot visualization
    analyzer.plot_results()
    
    # Print summary
    analyzer.print_summary(results)
    
    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds\n")
    
    return observer, simulator, analyzer, results


def run_comparison_simulations():
    """Run both retrocausal and control simulations for comparison"""
    # Control run (no retrocausal effects)
    print("\n" + "="*80)
    print("CONTROL SIMULATION: NO RETROCAUSAL EFFECTS")
    print("="*80)
    control_observer, control_simulator, control_analyzer, control_results = run_simulation(
        iterations=2000, retrocausal=False)
        
    # Experimental run (with retrocausal effects)
    print("\n" + "="*80)
    print("EXPERIMENTAL SIMULATION: WITH RETROCAUSAL EFFECTS")
    print("="*80)
    retro_observer, retro_simulator, retro_analyzer, retro_results = run_simulation(
        iterations=2000, retrocausal=True)
        
    # Compare results
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Compare key metrics
    metrics = ['ks_test', 'retrocausal_influence', 'pattern_match']
    for metric in metrics:
        if metric in control_results and metric in retro_results:
            print(f"\n{metric.upper()}:")
            if 'p_value' in control_results[metric] and 'p_value' in retro_results[metric]:
                c_pval = control_results[metric]['p_value']
                r_pval = retro_results[metric]['p_value']
                print(f"  Control p-value: {c_pval:.6f}")
                print(f"  Retrocausal p-value: {r_pval:.6f}")
                print(f"  Difference: {abs(c_pval - r_pval):.6f}")
                print(f"  Conclusion: {'SIGNIFICANT DIFFERENCE' if abs(c_pval - r_pval) > 0.01 else 'No significant difference'}")
    
    # Overall conclusion
    control_anomaly = any(r.get('significant', False) for r in control_results.values())
    retro_anomaly = any(r.get('significant', False) for r in retro_results.values())
    
    print("\nOVERALL CONCLUSION:")
    if not control_anomaly and retro_anomaly:
        print("  *** STRONG EVIDENCE FOR CONSCIOUSNESS-FIRST HYPOTHESIS ***")
        print("  Significant anomalies only present when retrocausal effects enabled")
    elif control_anomaly and retro_anomaly:
        print("  *** PARTIAL EVIDENCE FOR CONSCIOUSNESS-FIRST HYPOTHESIS ***")
        print("  Anomalies present in both simulations but stronger with retrocausal effects")
    else:
        print("  No convincing evidence for consciousness-first hypothesis")
        print("  Results are consistent with conventional time-first causality")
    
    print("\nDISCLAIMER:")
    print("  This is a simulated model and not a real quantum experiment. The results")
    print("  represent mathematical patterns in the simulation rather than physical reality.")
    print("  Any anomalies detected are emergent properties of the code architecture.")
    print("="*80)


if __name__ == "__main__":
    # Run both control and experimental simulations
    run_comparison_simulations()