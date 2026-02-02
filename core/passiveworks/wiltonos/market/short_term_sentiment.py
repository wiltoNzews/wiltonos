"""
Short Term Sentiment Module for WiltonOS
----------------------------------------
Analyzes short-term sentiment signals and narrative momentum.
Identifies inflection points, sentiment fatigue, and echo storms for tactical positioning.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import math

# Add paths for importing core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import entropy tracker and other modules if available
try:
    from wiltonos.market.entropy_tracker import get_entropy_tracker
    from wilton_core.memory.quantum_diary import add_diary_entry, register_insight
    has_quantum_modules = True
except ImportError:
    has_quantum_modules = False

# Default paths
DEFAULT_SENTIMENT_LOG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'sentiment_logs')

class SentimentEngine:
    """
    Short-term sentiment engine for tactical market positioning.
    """
    
    # Sentiment signal types
    SIGNAL_TYPES = {
        "narrative_inflection": "Change in narrative direction or tone",
        "sentiment_fatigue": "Exhaustion in sentiment after sustained trend",
        "echo_storm": "High-intensity narrative propagation",
        "silence_signal": "Significant drop in engagement after high intensity",
        "sarcasm_spike": "Increase in sarcastic/contrarian comments",
        "conviction_divergence": "Retail vs institutional narrative divergence",
        "tribal_activation": "Group identity-driven sentiment surge",
        "fud_cascade": "Fear, uncertainty, doubt spreading rapidly",
        "fomo_surge": "Fear of missing out driving rapid sentiment shift"
    }
    
    # Signal strength levels
    STRENGTH_LEVELS = {
        "extreme": 0.9,   # Strongest signal - rare but highly reliable
        "strong": 0.75,   # Strong signal - reliable with good history
        "moderate": 0.6,  # Moderate signal - worth acting on
        "mild": 0.45,     # Mild signal - directional hint
        "weak": 0.3       # Weak signal - early pattern only
    }
    
    def __init__(self, 
                log_dir: Optional[str] = None,
                use_entropy_tracker: bool = True,
                debug_mode: bool = False):
        """
        Initialize the sentiment engine.
        
        Args:
            log_dir: Directory to store sentiment logs
            use_entropy_tracker: Whether to use the entropy tracker
            debug_mode: Enable debug logging
        """
        self.log_dir = log_dir or DEFAULT_SENTIMENT_LOG_PATH
        self.debug_mode = debug_mode
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize entropy tracker if requested
        self.entropy_tracker = get_entropy_tracker() if use_entropy_tracker and 'get_entropy_tracker' in globals() else None
        
        # Active sentiment trackers
        self.sentiment_trackers = {}
        
        # Trading signal parameters
        self.signal_params = {
            "min_confidence": 0.6,        # Minimum confidence to generate a signal
            "lookback_days": 10,          # Days to look back for pattern detection
            "signal_expiry_hours": 48,    # How long a signal remains active
            "consecutive_points": 3,      # Minimum consecutive points for trend
            "threshold_cross_pct": 0.15   # Percent change to consider threshold crossed
        }
        
        print(f"SentimentEngine initialized. Log directory: {self.log_dir}")
    
    def track_asset(self, 
                  symbol: str,
                  description: Optional[str] = None,
                  initial_sentiment: Optional[float] = None,
                  initial_sources: Optional[Dict[str, float]] = None,
                  tags: Optional[List[str]] = None) -> Dict:
        """
        Start tracking an asset for short-term sentiment.
        
        Args:
            symbol: Asset symbol (e.g., "NVDA", "BTC", "ETH")
            description: Asset description
            initial_sentiment: Initial sentiment score (-1.0 to 1.0)
            initial_sources: Dict mapping sources to sentiment scores
            tags: List of tags for categorization
            
        Returns:
            The newly created sentiment tracker
        """
        symbol = symbol.upper()
        
        # Check if already tracking
        if symbol in self.sentiment_trackers:
            print(f"Already tracking asset {symbol}")
            return self.sentiment_trackers[symbol]
        
        # Initialize entropy tracker for this symbol if available
        if self.entropy_tracker:
            try:
                self.entropy_tracker.track_symbol(symbol, description)
            except Exception as e:
                print(f"Error initializing entropy tracking for {symbol}: {e}")
        
        # Create new tracker
        tracker = {
            "symbol": symbol,
            "description": description or f"Short-term sentiment tracker for {symbol}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "sentiment_history": [],
            "current_sentiment": initial_sentiment or 0.0,
            "sentiment_sources": initial_sources or {},
            "tags": tags or [],
            "active_signals": [],
            "expired_signals": [],
            "metrics": {
                "sentiment_volatility": 0.0,
                "sentiment_momentum": 0.0,
                "days_tracked": 0,
                "signal_accuracy": {
                    "correct": 0,
                    "incorrect": 0,
                    "unresolved": 0
                }
            }
        }
        
        # Add initial sentiment point if provided
        if initial_sentiment is not None:
            self._add_sentiment_point(
                tracker=tracker,
                sentiment=initial_sentiment,
                sources=initial_sources or {},
                volume=None,
                volatility=None
            )
        
        # Add to sentiment trackers
        self.sentiment_trackers[symbol] = tracker
        
        # Save to disk
        self._save_tracker(symbol)
        
        print(f"Started tracking asset {symbol} for short-term sentiment")
        return tracker
    
    def update_sentiment(self,
                        symbol: str,
                        sentiment: float,
                        sources: Optional[Dict[str, float]] = None,
                        volume: Optional[int] = None,
                        volatility: Optional[float] = None,
                        narrative_state: Optional[str] = None) -> Dict:
        """
        Update sentiment for a tracked asset.
        
        Args:
            symbol: Asset symbol
            sentiment: New sentiment score (-1.0 to 1.0)
            sources: Dict mapping sources to sentiment scores
            volume: Trading or mention volume
            volatility: Price or sentiment volatility
            narrative_state: Narrative state descriptor
            
        Returns:
            Updated sentiment tracker with any new signals
        """
        symbol = symbol.upper()
        
        if symbol not in self.sentiment_trackers:
            raise ValueError(f"Asset {symbol} not being tracked")
        
        tracker = self.sentiment_trackers[symbol]
        
        # Add sentiment point
        self._add_sentiment_point(
            tracker=tracker,
            sentiment=sentiment,
            sources=sources or {},
            volume=volume,
            volatility=volatility,
            narrative_state=narrative_state
        )
        
        # Update current sentiment
        old_sentiment = tracker["current_sentiment"]
        tracker["current_sentiment"] = sentiment
        tracker["updated_at"] = datetime.now().isoformat()
        
        # Update sources
        if sources:
            for source, score in sources.items():
                tracker["sentiment_sources"][source] = score
        
        # Update days tracked
        created = datetime.fromisoformat(tracker["created_at"])
        days_tracked = (datetime.now() - created).days
        tracker["metrics"]["days_tracked"] = days_tracked
        
        # Update metrics
        self._update_metrics(tracker)
        
        # Detect any new signals
        new_signals = self._detect_signals(tracker)
        if new_signals:
            print(f"Detected {len(new_signals)} new signals for {symbol}")
            tracker["active_signals"].extend(new_signals)
            
            # Register in quantum diary if available
            if has_quantum_modules:
                try:
                    for signal in new_signals:
                        register_insight(
                            label=f"{symbol} {signal['type'].replace('_', ' ').title()} Signal",
                            summary=f"Detected {signal['strength']} {signal['type']} signal for {symbol}\n{signal['description']}",
                            phi_impact=0.05,
                            tags=["sentiment", "short-term", symbol, signal["type"]] + tracker["tags"]
                        )
                except Exception as e:
                    print(f"Error registering sentiment signal in quantum diary: {e}")
        
        # Update expired signals
        now = datetime.now()
        still_active = []
        for signal in tracker["active_signals"]:
            created = datetime.fromisoformat(signal["created_at"])
            expiry_hours = signal.get("expiry_hours", self.signal_params["signal_expiry_hours"])
            
            if (now - created).total_seconds() / 3600 > expiry_hours:
                # Move to expired signals
                signal["expired_at"] = now.isoformat()
                if signal.get("resolution") is None:
                    signal["resolution"] = "expired"
                    tracker["metrics"]["signal_accuracy"]["unresolved"] += 1
                
                tracker["expired_signals"].append(signal)
            else:
                still_active.append(signal)
        
        tracker["active_signals"] = still_active
        
        # Save changes
        self._save_tracker(symbol)
        
        print(f"Updated {symbol} sentiment to {sentiment:.2f} (from {old_sentiment:.2f})")
        if new_signals:
            for signal in new_signals:
                print(f"  New signal: {signal['type']} ({signal['strength']})")
                
        return {
            "tracker": tracker,
            "new_signals": new_signals
        }
    
    def resolve_signal(self,
                      symbol: str,
                      signal_id: str,
                      resolution: str,
                      notes: Optional[str] = None) -> Dict:
        """
        Resolve a trading signal with outcome.
        
        Args:
            symbol: Asset symbol
            signal_id: ID of the signal to resolve
            resolution: Signal resolution ("correct", "incorrect", "partial", "invalid")
            notes: Additional notes on resolution
            
        Returns:
            Updated signal
        """
        symbol = symbol.upper()
        
        if symbol not in self.sentiment_trackers:
            raise ValueError(f"Asset {symbol} not being tracked")
        
        tracker = self.sentiment_trackers[symbol]
        
        # Find signal in active signals
        signal = None
        for s in tracker["active_signals"]:
            if s["id"] == signal_id:
                signal = s
                break
        
        if not signal:
            # Check expired signals
            for s in tracker["expired_signals"]:
                if s["id"] == signal_id:
                    signal = s
                    break
        
        if not signal:
            raise ValueError(f"Signal with ID {signal_id} not found")
        
        # Update signal resolution
        signal["resolution"] = resolution
        signal["resolved_at"] = datetime.now().isoformat()
        if notes:
            signal["resolution_notes"] = notes
        
        # Update accuracy metrics
        if resolution == "correct":
            tracker["metrics"]["signal_accuracy"]["correct"] += 1
        elif resolution == "incorrect":
            tracker["metrics"]["signal_accuracy"]["incorrect"] += 1
        elif resolution == "invalid":
            # Invalid signals don't count for accuracy
            pass
        else:  # partial or other outcomes
            # Count as half correct
            tracker["metrics"]["signal_accuracy"]["correct"] += 0.5
            tracker["metrics"]["signal_accuracy"]["incorrect"] += 0.5
        
        # If signal is active, move to expired
        if signal in tracker["active_signals"]:
            tracker["active_signals"].remove(signal)
            tracker["expired_signals"].append(signal)
        
        # Save changes
        self._save_tracker(symbol)
        
        print(f"Resolved signal {signal_id} as {resolution}")
        return signal
    
    def get_active_signals(self, 
                         symbol: Optional[str] = None, 
                         min_strength: Optional[float] = None) -> List[Dict]:
        """
        Get active trading signals.
        
        Args:
            symbol: Optional filter for specific asset
            min_strength: Minimum signal strength
            
        Returns:
            List of active signals
        """
        active_signals = []
        
        if symbol:
            symbol = symbol.upper()
            if symbol not in self.sentiment_trackers:
                return []
            
            trackers = {symbol: self.sentiment_trackers[symbol]}
        else:
            trackers = self.sentiment_trackers
        
        for symbol, tracker in trackers.items():
            for signal in tracker["active_signals"]:
                if min_strength is None or signal["strength_value"] >= min_strength:
                    signal_copy = signal.copy()
                    signal_copy["symbol"] = symbol
                    active_signals.append(signal_copy)
        
        # Sort by strength descending
        active_signals = sorted(active_signals, key=lambda x: x["strength_value"], reverse=True)
        
        return active_signals
    
    def get_sentiment_breakdown(self, symbol: str) -> Dict:
        """
        Get detailed sentiment breakdown for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Sentiment breakdown details
        """
        symbol = symbol.upper()
        
        if symbol not in self.sentiment_trackers:
            raise ValueError(f"Asset {symbol} not being tracked")
        
        tracker = self.sentiment_trackers[symbol]
        
        # Get sentiment history and sources
        history = tracker["sentiment_history"]
        
        # Get additional data from entropy tracker if available
        entropy_data = {}
        if self.entropy_tracker:
            try:
                # Get metrics, patterns, and latest event
                entropy_data["metrics"] = self.entropy_tracker.compute_metrics(symbol)
                
                # Get most recent event
                active_tracker = self.entropy_tracker.active_trackers.get(symbol, {})
                events = active_tracker.get("events", [])
                if events:
                    latest_event = max(events, key=lambda e: e.get("timestamp", ""))
                    entropy_data["latest_event"] = latest_event
            except Exception as e:
                print(f"Error getting entropy data for {symbol}: {e}")
        
        # Calculate signal accuracy if we have resolved signals
        accuracy = 0.0
        accuracy_data = tracker["metrics"]["signal_accuracy"]
        total_resolved = accuracy_data["correct"] + accuracy_data["incorrect"]
        if total_resolved > 0:
            accuracy = accuracy_data["correct"] / total_resolved
        
        # Create sentiment breakdown
        breakdown = {
            "symbol": symbol,
            "current_sentiment": tracker["current_sentiment"],
            "sentiment_momentum": tracker["metrics"]["sentiment_momentum"],
            "sentiment_volatility": tracker["metrics"]["sentiment_volatility"],
            "sentiment_history": sorted(history, key=lambda x: x["timestamp"])[-20:],  # Last 20 points
            "sentiment_by_source": tracker["sentiment_sources"],
            "active_signals_count": len(tracker["active_signals"]),
            "active_signals": tracker["active_signals"],
            "signal_accuracy": accuracy,
            "days_tracked": tracker["metrics"]["days_tracked"],
            "entropy_data": entropy_data
        }
        
        return breakdown
    
    def get_narrative_progression(self, symbol: str) -> Dict:
        """
        Get narrative progression analysis for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Narrative progression analysis
        """
        symbol = symbol.upper()
        
        if symbol not in self.sentiment_trackers:
            raise ValueError(f"Asset {symbol} not being tracked")
        
        tracker = self.sentiment_trackers[symbol]
        
        # Get sentiment history with narrative states
        history = sorted(tracker["sentiment_history"], key=lambda x: x["timestamp"])
        narrative_states = [h for h in history if "narrative_state" in h]
        
        # Extract narrative progression
        progression = []
        last_state = None
        
        for point in narrative_states:
            state = point["narrative_state"]
            if state != last_state:
                progression.append({
                    "timestamp": point["timestamp"],
                    "state": state,
                    "sentiment": point["sentiment"]
                })
                last_state = state
        
        # Get current active signals by type
        signal_types = {}
        for signal in tracker["active_signals"]:
            signal_type = signal["type"]
            if signal_type not in signal_types:
                signal_types[signal_type] = []
            signal_types[signal_type].append(signal)
        
        # Create narrative analysis
        analysis = {
            "symbol": symbol,
            "narrative_progression": progression,
            "current_state": narrative_states[-1]["narrative_state"] if narrative_states else None,
            "state_transitions": len(progression),
            "active_signal_types": signal_types,
            "sarcasm_detected": any(s["type"] == "sarcasm_spike" for s in tracker["active_signals"]),
            "tribal_activation": any(s["type"] == "tribal_activation" for s in tracker["active_signals"]),
            "echo_chamber_activity": any(s["type"] == "echo_storm" for s in tracker["active_signals"])
        }
        
        return analysis
    
    def _add_sentiment_point(self, 
                           tracker: Dict, 
                           sentiment: float,
                           sources: Dict[str, float],
                           volume: Optional[int] = None,
                           volatility: Optional[float] = None,
                           narrative_state: Optional[str] = None) -> None:
        """Add a sentiment history point to a tracker."""
        # Ensure sentiment is within range
        sentiment = max(-1.0, min(1.0, sentiment))
        
        point = {
            "timestamp": datetime.now().isoformat(),
            "sentiment": sentiment,
            "sources": sources
        }
        
        # Add optional fields if provided
        if volume is not None:
            point["volume"] = volume
        
        if volatility is not None:
            point["volatility"] = volatility
            
        if narrative_state is not None:
            point["narrative_state"] = narrative_state
        
        tracker["sentiment_history"].append(point)
    
    def _update_metrics(self, tracker: Dict) -> None:
        """Update calculated metrics for a tracker."""
        # Need at least 2 points for metrics
        history = tracker["sentiment_history"]
        if len(history) < 2:
            return
        
        # Sort history by timestamp
        history = sorted(history, key=lambda x: x["timestamp"])
        
        # Calculate sentiment volatility (standard deviation of changes)
        sentiment_values = [h["sentiment"] for h in history]
        changes = [abs(sentiment_values[i] - sentiment_values[i-1]) 
                  for i in range(1, len(sentiment_values))]
        
        if changes:
            sentiment_volatility = sum(changes) / len(changes)
            tracker["metrics"]["sentiment_volatility"] = sentiment_volatility
        
        # Calculate momentum (recent trend direction and strength)
        # Use last 5 points or all if less
        window_size = min(5, len(history))
        recent_points = history[-window_size:]
        
        # Simple linear regression slope as momentum
        if window_size >= 3:
            x = list(range(window_size))
            y = [p["sentiment"] for p in recent_points]
            
            # Calculate slope
            x_mean = sum(x) / window_size
            y_mean = sum(y) / window_size
            
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(window_size))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(window_size))
            
            momentum = numerator / denominator if denominator != 0 else 0
            
            # Normalize to -1.0 to 1.0 range
            max_expected_slope = 0.5  # Typical max slope we expect to see
            normalized_momentum = momentum / max_expected_slope
            normalized_momentum = max(-1.0, min(1.0, normalized_momentum))
            
            tracker["metrics"]["sentiment_momentum"] = normalized_momentum
    
    def _detect_signals(self, tracker: Dict) -> List[Dict]:
        """Detect trading signals from sentiment patterns."""
        # Need minimum history for signal detection
        history = tracker["sentiment_history"]
        if len(history) < self.signal_params["consecutive_points"]:
            return []
        
        # Sort history by timestamp
        history = sorted(history, key=lambda x: x["timestamp"])
        
        # Get recent points within lookback period
        lookback = datetime.now() - timedelta(days=self.signal_params["lookback_days"])
        recent_points = [p for p in history if datetime.fromisoformat(p["timestamp"]) >= lookback]
        
        if len(recent_points) < self.signal_params["consecutive_points"]:
            return []
        
        # List to store detected signals
        signals = []
        
        # Detect narrative inflection
        inflection_signal = self._detect_narrative_inflection(tracker, recent_points)
        if inflection_signal:
            signals.append(inflection_signal)
        
        # Detect sentiment fatigue
        fatigue_signal = self._detect_sentiment_fatigue(tracker, recent_points)
        if fatigue_signal:
            signals.append(fatigue_signal)
        
        # Detect echo storm
        echo_signal = self._detect_echo_storm(tracker, recent_points)
        if echo_signal:
            signals.append(echo_signal)
        
        # Detect silence after noise
        silence_signal = self._detect_silence_signal(tracker, recent_points)
        if silence_signal:
            signals.append(silence_signal)
        
        # Detect source divergence
        divergence_signal = self._detect_source_divergence(tracker, recent_points)
        if divergence_signal:
            signals.append(divergence_signal)
        
        # Detect tribal activation
        tribal_signal = self._detect_tribal_activation(tracker, recent_points)
        if tribal_signal:
            signals.append(tribal_signal)
        
        return signals
    
    def _detect_narrative_inflection(self, tracker: Dict, points: List[Dict]) -> Optional[Dict]:
        """Detect narrative inflection point."""
        # Need at least 5 points
        if len(points) < 5:
            return None
        
        # Look at trend before and after the middle point
        mid_idx = len(points) // 2
        before_points = points[:mid_idx]
        after_points = points[mid_idx:]
        
        # Calculate average sentiment before and after
        before_avg = sum(p["sentiment"] for p in before_points) / len(before_points)
        after_avg = sum(p["sentiment"] for p in after_points) / len(after_points)
        
        # Check if there's a significant change in sentiment direction
        threshold = self.signal_params["threshold_cross_pct"]
        if (before_avg < 0 and after_avg > 0) or (before_avg > 0 and after_avg < 0):
            # Direction change
            strength = min(1.0, abs(after_avg - before_avg) * 2)
            if strength >= self.STRENGTH_LEVELS["weak"]:
                return self._create_signal(
                    signal_type="narrative_inflection",
                    strength=self._get_strength_label(strength),
                    strength_value=strength,
                    description=f"Sentiment direction change from {before_avg:.2f} to {after_avg:.2f}",
                    direction="positive" if after_avg > before_avg else "negative"
                )
        elif abs(after_avg - before_avg) > threshold:
            # Significant acceleration or deceleration
            strength = min(1.0, abs(after_avg - before_avg) * 1.5)
            if strength >= self.STRENGTH_LEVELS["weak"]:
                direction = "acceleration" if abs(after_avg) > abs(before_avg) else "deceleration"
                return self._create_signal(
                    signal_type="narrative_inflection",
                    strength=self._get_strength_label(strength),
                    strength_value=strength,
                    description=f"Sentiment {direction} from {before_avg:.2f} to {after_avg:.2f}",
                    direction=direction
                )
        
        return None
    
    def _detect_sentiment_fatigue(self, tracker: Dict, points: List[Dict]) -> Optional[Dict]:
        """Detect sentiment fatigue pattern."""
        # Need volume data for fatigue detection
        if len(points) < 5 or not any("volume" in p for p in points):
            return None
        
        # Filter points with volume data
        volume_points = [p for p in points if "volume" in p]
        if len(volume_points) < 4:
            return None
        
        # Check for decrease in volume while sentiment remains extreme
        for i in range(len(volume_points) - 3):
            window = volume_points[i:i+4]
            
            # Calculate average volume and sentiment
            avg_volume = sum(p["volume"] for p in window) / len(window)
            avg_sentiment = sum(p["sentiment"] for p in window) / len(window)
            
            # Check if sentiment is extreme
            if abs(avg_sentiment) > 0.6:
                # Check for volume decline
                start_volume = window[0]["volume"]
                end_volume = window[-1]["volume"]
                
                if end_volume < start_volume * 0.6:  # 40% drop in volume
                    strength = min(1.0, abs(avg_sentiment) * (start_volume / (end_volume + 1)) * 0.3)
                    if strength >= self.STRENGTH_LEVELS["weak"]:
                        sentiment_dir = "positive" if avg_sentiment > 0 else "negative"
                        return self._create_signal(
                            signal_type="sentiment_fatigue",
                            strength=self._get_strength_label(strength),
                            strength_value=strength,
                            description=f"Volume declining ({end_volume/start_volume:.1%} of initial) while {sentiment_dir} sentiment remains extreme at {avg_sentiment:.2f}",
                            direction="reversal" if avg_sentiment > 0 else "bounce"
                        )
        
        return None
    
    def _detect_echo_storm(self, tracker: Dict, points: List[Dict]) -> Optional[Dict]:
        """Detect echo storm pattern (resonance intensification)."""
        # Need source data for echo detection
        source_points = [p for p in points if p.get("sources") and len(p["sources"]) >= 2]
        if len(source_points) < 3:
            return None
        
        # Check for increasing alignment in sentiment across sources
        for i in range(len(source_points) - 2):
            window = source_points[i:i+3]
            
            # Calculate source alignment for each point (standard deviation of source sentiments)
            alignment_scores = []
            for point in window:
                source_values = list(point["sources"].values())
                mean = sum(source_values) / len(source_values)
                variance = sum((v - mean) ** 2 for v in source_values) / len(source_values)
                std_dev = math.sqrt(variance)
                # Lower std_dev means higher alignment
                alignment = 1.0 - min(1.0, std_dev)
                alignment_scores.append(alignment)
            
            # Check if alignment is increasing
            if alignment_scores[0] < alignment_scores[1] < alignment_scores[2]:
                # Significant increasing alignment
                avg_sentiment = sum(p["sentiment"] for p in window) / len(window)
                strength = min(1.0, alignment_scores[2] * abs(avg_sentiment) * 1.2)
                
                if strength >= self.STRENGTH_LEVELS["weak"]:
                    sentiment_dir = "positive" if avg_sentiment > 0 else "negative"
                    return self._create_signal(
                        signal_type="echo_storm",
                        strength=self._get_strength_label(strength),
                        strength_value=strength,
                        description=f"Increasing sentiment alignment across sources with {sentiment_dir} bias at {avg_sentiment:.2f}",
                        direction=sentiment_dir
                    )
        
        return None
    
    def _detect_silence_signal(self, tracker: Dict, points: List[Dict]) -> Optional[Dict]:
        """Detect silence after high engagement pattern."""
        # Need volume data for silence detection
        if len(points) < 5 or not any("volume" in p for p in points):
            return None
        
        # Filter points with volume data
        volume_points = [p for p in points if "volume" in p]
        if len(volume_points) < 5:
            return None
        
        # Check the most recent 3 points
        recent = volume_points[-3:]
        prior = volume_points[-6:-3] if len(volume_points) >= 6 else volume_points[:-3]
        
        # Calculate average volume in recent and prior periods
        recent_avg_volume = sum(p["volume"] for p in recent) / len(recent)
        prior_avg_volume = sum(p["volume"] for p in prior) / len(prior)
        
        # Check for significant volume drop
        if recent_avg_volume < prior_avg_volume * 0.25:  # 75% drop in volume
            # This is our silence signal
            # Look at sentiment before silence
            prior_sentiment = sum(p["sentiment"] for p in prior) / len(prior)
            
            # Signal strength based on volume drop and sentiment extremity
            volume_ratio = recent_avg_volume / prior_avg_volume
            strength = min(1.0, (1.0 - volume_ratio) * abs(prior_sentiment) * 1.5)
            
            if strength >= self.STRENGTH_LEVELS["weak"]:
                sentiment_dir = "positive" if prior_sentiment > 0 else "negative"
                reversal_hint = "potential reversal" if prior_sentiment < 0 else "potential pullback"
                
                return self._create_signal(
                    signal_type="silence_signal",
                    strength=self._get_strength_label(strength),
                    strength_value=strength,
                    description=f"Significant volume decline ({volume_ratio:.1%} of prior) after {sentiment_dir} sentiment phase. Indicates {reversal_hint}.",
                    direction="reversal" if prior_sentiment < 0 else "pullback"
                )
        
        return None
    
    def _detect_source_divergence(self, tracker: Dict, points: List[Dict]) -> Optional[Dict]:
        """Detect divergence between sentiment sources."""
        # Need source data for divergence detection
        source_points = [p for p in points if p.get("sources") and len(p["sources"]) >= 2]
        if len(source_points) < 3:
            return None
        
        # Only use most recent point
        point = source_points[-1]
        sources = point["sources"]
        
        # Need at least 2 sources
        if len(sources) < 2:
            return None
        
        # Look for institutional vs retail divergence if those sources exist
        institutional_sources = ["institutional", "analyst", "fund", "expert"]
        retail_sources = ["retail", "social", "twitter", "reddit"]
        
        inst_values = []
        retail_values = []
        
        for source, value in sources.items():
            if any(inst in source.lower() for inst in institutional_sources):
                inst_values.append(value)
            elif any(retail in source.lower() for retail in retail_sources):
                retail_values.append(value)
        
        # If we have both types of sources, check for divergence
        if inst_values and retail_values:
            inst_avg = sum(inst_values) / len(inst_values)
            retail_avg = sum(retail_values) / len(retail_values)
            
            # Check for significant divergence
            divergence = abs(inst_avg - retail_avg)
            if divergence > 0.5:  # Significant sentiment gap
                # Direction is from perspective of institutional view
                direction = "positive" if inst_avg > retail_avg else "negative"
                
                # Signal strength based on divergence magnitude
                strength = min(1.0, divergence * 1.2)
                
                if strength >= self.STRENGTH_LEVELS["weak"]:
                    return self._create_signal(
                        signal_type="conviction_divergence",
                        strength=self._get_strength_label(strength),
                        strength_value=strength,
                        description=f"Institutional sentiment ({inst_avg:.2f}) diverges from retail ({retail_avg:.2f}). Institutional bias is {direction}.",
                        direction=direction
                    )
        
        # General source divergence if no inst/retail
        if not (inst_values and retail_values):
            # Calculate standard deviation of sources
            values = list(sources.values())
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance)
            
            # Check for high standard deviation
            if std_dev > 0.5:  # High divergence
                # Find min and max sources
                min_source = min(sources.items(), key=lambda x: x[1])
                max_source = max(sources.items(), key=lambda x: x[1])
                
                # Signal strength based on standard deviation
                strength = min(1.0, std_dev * 1.1)
                
                if strength >= self.STRENGTH_LEVELS["weak"]:
                    return self._create_signal(
                        signal_type="conviction_divergence",
                        strength=self._get_strength_label(strength),
                        strength_value=strength,
                        description=f"High source divergence (σ={std_dev:.2f}). {max_source[0]} most positive at {max_source[1]:.2f}, {min_source[0]} most negative at {min_source[1]:.2f}.",
                        direction="mixed"
                    )
        
        return None
    
    def _detect_tribal_activation(self, tracker: Dict, points: List[Dict]) -> Optional[Dict]:
        """Detect tribal activation pattern (group identity-driven sentiment)."""
        # Check for narrative state to detect tribal activation
        narrative_points = [p for p in points if "narrative_state" in p]
        if not narrative_points:
            return None
        
        # Check recent points for tribal activation state
        tribal_states = ["tribal_activation", "group_identity", "echo_chamber"]
        
        for point in reversed(narrative_points):  # Start with most recent
            state = point["narrative_state"]
            
            if any(tribal in state.lower() for tribal in tribal_states):
                # Calculate signal strength based on sentiment extremity
                sentiment = point["sentiment"]
                strength = min(1.0, abs(sentiment) * 1.3)
                
                if strength >= self.STRENGTH_LEVELS["weak"]:
                    sentiment_dir = "positive" if sentiment > 0 else "negative"
                    return self._create_signal(
                        signal_type="tribal_activation",
                        strength=self._get_strength_label(strength),
                        strength_value=strength,
                        description=f"Group identity-driven sentiment detected in '{state}' state with {sentiment_dir} bias at {sentiment:.2f}.",
                        direction=sentiment_dir
                    )
        
        return None
    
    def _create_signal(self, 
                     signal_type: str,
                     strength: str,
                     strength_value: float,
                     description: str,
                     direction: str,
                     expiry_hours: Optional[int] = None) -> Dict:
        """Create a trading signal object."""
        signal_id = f"signal_{signal_type}_{int(time.time())}_{str(strength_value)[:4]}"
        
        signal = {
            "id": signal_id,
            "type": signal_type,
            "strength": strength,
            "strength_value": strength_value,
            "description": description,
            "direction": direction,
            "created_at": datetime.now().isoformat(),
            "expiry_hours": expiry_hours or self.signal_params["signal_expiry_hours"],
            "resolution": None,
            "confidence": strength_value  # Use strength as initial confidence
        }
        
        return signal
    
    def _get_strength_label(self, strength_value: float) -> str:
        """Get strength label based on strength value."""
        if strength_value >= self.STRENGTH_LEVELS["extreme"]:
            return "extreme"
        elif strength_value >= self.STRENGTH_LEVELS["strong"]:
            return "strong"
        elif strength_value >= self.STRENGTH_LEVELS["moderate"]:
            return "moderate"
        elif strength_value >= self.STRENGTH_LEVELS["mild"]:
            return "mild"
        else:
            return "weak"
    
    def _save_tracker(self, symbol: str) -> bool:
        """Save a tracker to disk."""
        symbol = symbol.upper()
        
        if symbol not in self.sentiment_trackers:
            return False
        
        tracker_path = os.path.join(self.log_dir, f"{symbol.lower()}_sentiment.json")
        
        try:
            with open(tracker_path, 'w', encoding='utf-8') as f:
                json.dump(self.sentiment_trackers[symbol], f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving tracker for {symbol}: {e}")
            return False
    
    def _load_tracker(self, symbol: str) -> Optional[Dict]:
        """Load a tracker from disk."""
        symbol = symbol.upper()
        
        tracker_path = os.path.join(self.log_dir, f"{symbol.lower()}_sentiment.json")
        
        if not os.path.exists(tracker_path):
            return None
        
        try:
            with open(tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            return tracker
        except Exception as e:
            print(f"Error loading tracker for {symbol}: {e}")
            return None
    
    def load_all_trackers(self) -> int:
        """
        Load all saved trackers from disk.
        
        Returns:
            Number of trackers loaded
        """
        count = 0
        for filename in os.listdir(self.log_dir):
            if filename.endswith("_sentiment.json"):
                symbol = filename.split("_sentiment.json")[0].upper()
                tracker = self._load_tracker(symbol)
                
                if tracker:
                    self.sentiment_trackers[symbol] = tracker
                    count += 1
        
        print(f"Loaded {count} sentiment trackers from disk")
        return count

# Singleton pattern
_sentiment_engine_instance = None

def get_sentiment_engine(log_dir: Optional[str] = None,
                        use_entropy_tracker: bool = True,
                        debug_mode: bool = False) -> SentimentEngine:
    """
    Get or create singleton instance of the sentiment engine.
    
    Args:
        log_dir: Directory to store sentiment logs
        use_entropy_tracker: Whether to use the entropy tracker
        debug_mode: Enable debug logging
        
    Returns:
        SentimentEngine instance
    """
    global _sentiment_engine_instance
    
    if _sentiment_engine_instance is None:
        _sentiment_engine_instance = SentimentEngine(
            log_dir=log_dir,
            use_entropy_tracker=use_entropy_tracker,
            debug_mode=debug_mode
        )
    
    return _sentiment_engine_instance

if __name__ == "__main__":
    # Example usage
    engine = get_sentiment_engine(debug_mode=True)
    
    # Track assets
    # Track NVDA with different sources
    engine.track_asset(
        symbol="NVDA",
        description="NVIDIA Corporation - AI and GPU leader",
        initial_sentiment=0.4,
        initial_sources={
            "institutional": 0.6,
            "retail": 0.2,
            "analyst": 0.5,
            "twitter": 0.1,
            "reddit": 0.3
        },
        tags=["tech", "ai", "semiconductor"]
    )
    
    # Update with a narrative state
    engine.update_sentiment(
        symbol="NVDA",
        sentiment=0.3,
        sources={
            "institutional": 0.7,
            "retail": 0.1,
            "analyst": 0.6,
            "twitter": -0.1,
            "reddit": 0.1
        },
        volume=12500,
        narrative_state="shock_adjustment"
    )
    
    # Another update with increasing divergence
    engine.update_sentiment(
        symbol="NVDA",
        sentiment=0.2,
        sources={
            "institutional": 0.8,
            "retail": -0.2,
            "analyst": 0.7,
            "twitter": -0.3,
            "reddit": -0.1
        },
        volume=9800,
        narrative_state="tribal_activation"
    )
    
    # Final update with decreasing volume
    engine.update_sentiment(
        symbol="NVDA",
        sentiment=0.1,
        sources={
            "institutional": 0.7,
            "retail": -0.4,
            "analyst": 0.6,
            "twitter": -0.5,
            "reddit": -0.3
        },
        volume=4200,
        narrative_state="tribal_activation"
    )
    
    # Get active signals
    active_signals = engine.get_active_signals()
    print(f"Active signals: {len(active_signals)}")
    for signal in active_signals:
        print(f"  - [{signal['symbol']}] {signal['type']} ({signal['strength']}): {signal['description']}")
    
    # Get sentiment breakdown
    breakdown = engine.get_sentiment_breakdown("NVDA")
    print(f"\nNVDA sentiment: {breakdown['current_sentiment']:.2f}")
    print(f"Sentiment momentum: {breakdown['sentiment_momentum']:.2f}")
    print(f"Source sentiment:")
    for source, value in breakdown["sentiment_by_source"].items():
        print(f"  - {source}: {value:.2f}")
    
    # Get narrative progression
    progression = engine.get_narrative_progression("NVDA")
    print(f"\nNarrative progression:")
    for state in progression["narrative_progression"]:
        print(f"  - {state['timestamp']}: {state['state']} ({state['sentiment']:.2f})")
    
    print(f"Current state: {progression['current_state']}")
    print(f"Echo chamber activity: {'Yes' if progression['echo_chamber_activity'] else 'No'}")
    print(f"Tribal activation: {'Yes' if progression['tribal_activation'] else 'No'}")
    
    # Resolve a signal if we have any
    if active_signals:
        engine.resolve_signal(
            symbol=active_signals[0]["symbol"],
            signal_id=active_signals[0]["id"],
            resolution="correct",
            notes="Signal correctly predicted institutional/retail divergence"
        )