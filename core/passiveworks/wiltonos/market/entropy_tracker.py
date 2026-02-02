"""
Entropy Tracker for WiltonOS
----------------------------
Tracks narrative entropy, sentiment dynamics, and social resonance across market tickers.
Identifies narrative collision points, sentiment fatigue, and echo storms.
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union

# Add paths for importing core modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import WiltonOS core modules if available
try:
    from wilton_core.memory.quantum_diary import add_diary_entry, register_insight
    from wilton_core.memory.semantic_tagger import get_semantic_tagger
    from wilton_core.memory.thread_map import get_thread_map, ThreadMap
    has_quantum_modules = True
except ImportError:
    has_quantum_modules = False

# Default paths
DEFAULT_ENTROPY_LOG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'entropy_logs')
DEFAULT_NARRATIVE_BLOOM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                          'wilton_core', 'memory', 'narrative_bloom.json')

class EntropyTracker:
    """
    Tracks narrative entropy and sentiment across financial and social media data.
    """
    
    # Sentiment states in escalating order
    SENTIMENT_STATES = [
        "neutral",
        "interest", 
        "concern", 
        "shock", 
        "fear", 
        "confusion", 
        "anger", 
        "tribal_activation", 
        "collapse_of_voice",
        "narrative_reset"
    ]
    
    # Event types
    EVENT_TYPES = {
        "narrative_echo_storm": "High-intensity narrative propagation with echo chambers",
        "sentiment_fatigue": "Rapid drop in engagement after high-intensity period",
        "institutional_divergence": "Different narrative tracks between retail and institutional voices",
        "alpha_attention_shift": "Influencer-driven spike in attention",
        "narrative_collision": "Two competing narratives creating friction and confusion",
        "sentiment_volatility_spike": "Rapid fluctuations in sentiment without clear direction"
    }
    
    def __init__(self, 
                 log_dir: Optional[str] = None, 
                 narrative_bloom_path: Optional[str] = None,
                 debug_mode: bool = False):
        """
        Initialize the entropy tracker.
        
        Args:
            log_dir: Directory to store entropy logs
            narrative_bloom_path: Path to the narrative bloom JSON file
            debug_mode: Enable debug logging
        """
        self.log_dir = log_dir or DEFAULT_ENTROPY_LOG_PATH
        self.narrative_bloom_path = narrative_bloom_path or DEFAULT_NARRATIVE_BLOOM_PATH
        self.debug_mode = debug_mode
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Load or initialize the narrative bloom data
        self.narrative_bloom = self._load_narrative_bloom()
        
        # Thread map for social media integration
        self.thread_map = get_thread_map() if has_quantum_modules else None
        
        # Active trackers
        self.active_trackers = {}
        
        print(f"EntropyTracker initialized. Log directory: {self.log_dir}")
        
    def _load_narrative_bloom(self) -> Dict:
        """Load the narrative bloom data or create it if it doesn't exist."""
        if os.path.exists(self.narrative_bloom_path):
            try:
                with open(self.narrative_bloom_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading narrative bloom: {e}")
        
        # Create default structure
        default_bloom = {
            "meta": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0"
            },
            "word_clusters": {},
            "quote_spikes": [],
            "tracked_symbols": {},
            "events": []
        }
        
        # Save default structure
        os.makedirs(os.path.dirname(self.narrative_bloom_path), exist_ok=True)
        with open(self.narrative_bloom_path, 'w', encoding='utf-8') as f:
            json.dump(default_bloom, f, indent=2)
        
        return default_bloom
    
    def _save_narrative_bloom(self) -> bool:
        """Save the narrative bloom data."""
        try:
            # Update last updated timestamp
            self.narrative_bloom["meta"]["last_updated"] = datetime.now().isoformat()
            
            with open(self.narrative_bloom_path, 'w', encoding='utf-8') as f:
                json.dump(self.narrative_bloom, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving narrative bloom: {e}")
            return False
    
    def track_symbol(self, 
                   symbol: str, 
                   description: Optional[str] = None,
                   initial_data: Optional[Dict] = None) -> Dict:
        """
        Start tracking a new symbol for narrative entropy.
        
        Args:
            symbol: The ticker or topic symbol (e.g., "NVDA", "BTC", "AI_ETHICS")
            description: Optional description of what this symbol represents
            initial_data: Optional initial data points for this symbol
            
        Returns:
            The newly created tracker
        """
        symbol = symbol.upper()  # Normalize symbol
        
        # Check if already tracking
        if symbol in self.active_trackers:
            print(f"Already tracking symbol {symbol}")
            return self.active_trackers[symbol]
        
        # Create new tracker
        tracker = {
            "symbol": symbol,
            "description": description or f"Narrative tracker for {symbol}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "data_points": [],
            "events": [],
            "sentiment_arc": {},
            "current_resonance": 0.0,
            "metrics": {
                "max_resonance": 0.0,
                "sentiment_volatility": 0.0,
                "narrative_entropy": 0.0
            }
        }
        
        # Add initial data if provided
        if initial_data:
            for point in initial_data.get("data_points", []):
                self.add_data_point(symbol, **point)
                
            for event in initial_data.get("events", []):
                self.add_event(symbol, **event)
        
        # Add to active trackers
        self.active_trackers[symbol] = tracker
        
        # Register in narrative bloom tracked symbols
        if symbol not in self.narrative_bloom["tracked_symbols"]:
            self.narrative_bloom["tracked_symbols"][symbol] = {
                "first_tracked": datetime.now().isoformat(),
                "description": description or f"Narrative tracker for {symbol}",
                "events_count": 0,
                "max_resonance": 0.0
            }
            self._save_narrative_bloom()
            
        print(f"Started tracking symbol: {symbol}")
        return tracker
    
    def add_data_point(self,
                      symbol: str,
                      timestamp: Optional[str] = None,
                      mention_count: Optional[int] = None,
                      sentiment_score: Optional[float] = None,
                      sentiment_state: Optional[str] = None,
                      price: Optional[float] = None,
                      volume: Optional[int] = None,
                      top_terms: Optional[List[str]] = None,
                      sources: Optional[Dict[str, int]] = None,
                      resonance_score: Optional[float] = None) -> Dict:
        """
        Add a new data point for a tracked symbol.
        
        Args:
            symbol: The ticker or topic symbol
            timestamp: ISO format timestamp (defaults to now)
            mention_count: Number of mentions in the time period
            sentiment_score: Sentiment score from -1.0 to 1.0
            sentiment_state: Named sentiment state
            price: Current price if applicable
            volume: Trading volume if applicable
            top_terms: List of top associated terms
            sources: Dict mapping source platforms to mention counts
            resonance_score: Narrative resonance score from 0.0 to 1.0
            
        Returns:
            The added data point
        """
        symbol = symbol.upper()
        
        # Start tracking if not already tracked
        if symbol not in self.active_trackers:
            self.track_symbol(symbol)
        
        # Create data point
        point = {
            "timestamp": timestamp or datetime.now().isoformat(),
            "mention_count": mention_count,
            "sentiment_score": sentiment_score,
            "sentiment_state": sentiment_state,
            "price": price,
            "volume": volume,
            "top_terms": top_terms or [],
            "sources": sources or {},
            "resonance_score": resonance_score
        }
        
        # Remove None values
        point = {k: v for k, v in point.items() if v is not None}
        
        # Add to tracker
        self.active_trackers[symbol]["data_points"].append(point)
        self.active_trackers[symbol]["updated_at"] = datetime.now().isoformat()
        
        # Update sentiment arc if sentiment state is provided
        if sentiment_state:
            date_key = datetime.fromisoformat(point["timestamp"]).strftime("%b %d")
            self.active_trackers[symbol]["sentiment_arc"][date_key] = sentiment_state
        
        # Update current resonance if provided
        if resonance_score is not None:
            self.active_trackers[symbol]["current_resonance"] = resonance_score
            
            # Update max resonance if applicable
            if resonance_score > self.active_trackers[symbol]["metrics"]["max_resonance"]:
                self.active_trackers[symbol]["metrics"]["max_resonance"] = resonance_score
                
                # Also update in narrative bloom
                if symbol in self.narrative_bloom["tracked_symbols"]:
                    self.narrative_bloom["tracked_symbols"][symbol]["max_resonance"] = resonance_score
                    self._save_narrative_bloom()
        
        # Save to disk if we have enough data points (every 10)
        if len(self.active_trackers[symbol]["data_points"]) % 10 == 0:
            self._save_tracker(symbol)
        
        if self.debug_mode:
            print(f"Added data point for {symbol}: {point}")
            
        return point
    
    def add_event(self,
                 symbol: str,
                 event_type: str,
                 label: str,
                 timestamp: Optional[str] = None,
                 description: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 sentiment_arc: Optional[Dict[str, str]] = None,
                 resonance_score: Optional[float] = None,
                 actionable: Optional[bool] = None,
                 watch_for: Optional[List[str]] = None) -> Dict:
        """
        Add a significant event for a symbol.
        
        Args:
            symbol: The ticker or topic symbol
            event_type: Type of event (e.g., "narrative_echo_storm")
            label: Short label for the event
            timestamp: ISO format timestamp (defaults to now)
            description: Detailed description of the event
            tags: List of tags for the event
            sentiment_arc: Dict mapping dates to sentiment states
            resonance_score: Event resonance score from 0.0 to 1.0
            actionable: Whether the event suggests actionable insights
            watch_for: List of things to watch for following this event
            
        Returns:
            The added event
        """
        symbol = symbol.upper()
        
        # Start tracking if not already tracked
        if symbol not in self.active_trackers:
            self.track_symbol(symbol)
        
        # Validate event type
        if event_type not in self.EVENT_TYPES and not event_type.startswith("custom_"):
            event_type = "custom_" + event_type
            print(f"Warning: Unknown event type. Using {event_type}")
            
        # Create event
        timestamp_val = timestamp or datetime.now().isoformat()
        event = {
            "event_id": f"{symbol}_{int(time.time())}_{label.replace(' ', '_')}",
            "event_type": event_type,
            "label": label,
            "timestamp": timestamp_val,
            "description": description,
            "tags": tags or [],
            "sentiment_arc": sentiment_arc or {},
            "resonance_score": resonance_score or 0.0,
            "actionable": actionable or False,
            "watch_for": watch_for or []
        }
        
        # Add symbol tag if not present
        if symbol not in event["tags"]:
            event["tags"].append(symbol)
        
        # Add to tracker
        self.active_trackers[symbol]["events"].append(event)
        self.active_trackers[symbol]["updated_at"] = timestamp_val
        
        # Add to narrative bloom events
        bloom_event = event.copy()
        bloom_event["symbol"] = symbol
        self.narrative_bloom["events"].append(bloom_event)
        
        # Update event count in tracked symbols
        if symbol in self.narrative_bloom["tracked_symbols"]:
            self.narrative_bloom["tracked_symbols"][symbol]["events_count"] += 1
        
        # Save changes
        self._save_tracker(symbol)
        self._save_narrative_bloom()
        
        # Register in quantum diary if available
        if has_quantum_modules:
            try:
                add_diary_entry(
                    entry_type="external_trigger",
                    label=f"{symbol} Event: {label}",
                    summary=description or f"{event_type} event detected for {symbol}",
                    tags=["market", "entropy", symbol] + (tags or []),
                    phi_impact=min(resonance_score or 0.0, 0.1),  # Cap impact at 0.1
                    significance_score=resonance_score,
                    source={
                        "type": "market_entropy",
                        "symbol": symbol,
                        "event_type": event_type
                    },
                    emotional_valence=self._determine_emotional_valence(sentiment_arc or {})
                )
            except Exception as e:
                print(f"Error registering event in quantum diary: {e}")
        
        if self.debug_mode:
            print(f"Added event for {symbol}: {label}")
            
        return event
    
    def compute_metrics(self, symbol: str) -> Dict[str, float]:
        """
        Compute current metrics for a tracked symbol.
        
        Args:
            symbol: The ticker or topic symbol
            
        Returns:
            Dict with updated metrics
        """
        symbol = symbol.upper()
        
        if symbol not in self.active_trackers:
            print(f"Symbol {symbol} not tracked")
            return {}
        
        tracker = self.active_trackers[symbol]
        
        # Need multiple data points to compute metrics
        if len(tracker["data_points"]) < 2:
            return tracker["metrics"]
        
        # Extract data for calculations
        data_points = sorted(tracker["data_points"], key=lambda x: x["timestamp"])
        
        # Get sentiment scores and resonance scores where available
        sentiment_scores = [p.get("sentiment_score") for p in data_points if "sentiment_score" in p]
        resonance_scores = [p.get("resonance_score") for p in data_points if "resonance_score" in p]
        
        # Calculate sentiment volatility (standard deviation of changes)
        if len(sentiment_scores) >= 2:
            changes = [abs(sentiment_scores[i] - sentiment_scores[i-1]) for i in range(1, len(sentiment_scores))]
            sentiment_volatility = sum(changes) / len(changes) if changes else 0
            tracker["metrics"]["sentiment_volatility"] = sentiment_volatility
        
        # Calculate narrative entropy based on sentiment state transitions
        sentiment_states = [p.get("sentiment_state") for p in data_points if "sentiment_state" in p]
        if len(sentiment_states) >= 2:
            # Count transitions between states
            transitions = sum(1 for i in range(1, len(sentiment_states)) if sentiment_states[i] != sentiment_states[i-1])
            # Normalize by maximum possible transitions
            max_transitions = len(sentiment_states) - 1
            narrative_entropy = transitions / max_transitions if max_transitions > 0 else 0
            tracker["metrics"]["narrative_entropy"] = narrative_entropy
        
        # Update max resonance
        if resonance_scores:
            max_resonance = max(resonance_scores)
            tracker["metrics"]["max_resonance"] = max(tracker["metrics"]["max_resonance"], max_resonance)
        
        return tracker["metrics"]
    
    def detect_narrative_patterns(self, symbol: str) -> List[Dict]:
        """
        Detect narrative patterns in a symbol's data.
        
        Args:
            symbol: The ticker or topic symbol
            
        Returns:
            List of detected patterns
        """
        symbol = symbol.upper()
        
        if symbol not in self.active_trackers:
            print(f"Symbol {symbol} not tracked")
            return []
        
        tracker = self.active_trackers[symbol]
        
        # Need sufficient data points
        if len(tracker["data_points"]) < 3:
            return []
        
        # Sort data points by timestamp
        data_points = sorted(tracker["data_points"], key=lambda x: x["timestamp"])
        
        patterns = []
        
        # Detect sentiment fatigue (rapid drop in mentions after a peak)
        mentions = [p.get("mention_count") for p in data_points if "mention_count" in p]
        if len(mentions) >= 3:
            for i in range(1, len(mentions)-1):
                # Check for peak followed by significant drop
                if mentions[i] > mentions[i-1] * 1.5 and mentions[i+1] < mentions[i] * 0.6:
                    pattern = {
                        "type": "sentiment_fatigue",
                        "timestamp": data_points[i+1]["timestamp"],
                        "confidence": 0.7,
                        "details": {
                            "previous_mentions": mentions[i-1],
                            "peak_mentions": mentions[i],
                            "drop_mentions": mentions[i+1],
                            "drop_percentage": (mentions[i] - mentions[i+1]) / mentions[i] * 100
                        }
                    }
                    patterns.append(pattern)
                    
                    # Create an event if drop is severe (>70%)
                    if mentions[i+1] < mentions[i] * 0.3:
                        self.add_event(
                            symbol=symbol,
                            event_type="sentiment_fatigue",
                            label=f"{symbol} Sentiment Fatigue Detected",
                            timestamp=data_points[i+1]["timestamp"],
                            description=f"Significant drop in engagement ({pattern['details']['drop_percentage']:.1f}%) after high-intensity period",
                            resonance_score=0.65,
                            actionable=True,
                            watch_for=["narrative reset", "price lag", "institutional moves"]
                        )
        
        # Detect narrative echo storms (sustained high resonance)
        resonance = [p.get("resonance_score") for p in data_points if "resonance_score" in p]
        if len(resonance) >= 3:
            # Check for sustained high resonance
            high_resonance_streak = 0
            for i in range(len(resonance)):
                if resonance[i] and resonance[i] > 0.6:
                    high_resonance_streak += 1
                    if high_resonance_streak >= 3:
                        pattern = {
                            "type": "narrative_echo_storm",
                            "timestamp": data_points[i]["timestamp"],
                            "confidence": 0.8,
                            "details": {
                                "duration_points": high_resonance_streak,
                                "avg_resonance": sum(resonance[i-high_resonance_streak+1:i+1]) / high_resonance_streak
                            }
                        }
                        patterns.append(pattern)
                        
                        # Only create event if we haven't already for this streak
                        if not any(p["type"] == "narrative_echo_storm" for p in patterns[:-1]):
                            self.add_event(
                                symbol=symbol,
                                event_type="narrative_echo_storm",
                                label=f"{symbol} Narrative Echo Storm",
                                timestamp=data_points[i]["timestamp"],
                                description=f"Sustained high resonance ({pattern['details']['avg_resonance']:.2f}) detected over {high_resonance_streak} data points",
                                resonance_score=pattern['details']['avg_resonance'],
                                actionable=True,
                                watch_for=["sentiment extremes", "volatility", "momentum shift"]
                            )
                else:
                    high_resonance_streak = 0
        
        return patterns
    
    def _save_tracker(self, symbol: str) -> bool:
        """Save a tracker to disk."""
        symbol = symbol.upper()
        
        if symbol not in self.active_trackers:
            return False
        
        tracker_path = os.path.join(self.log_dir, f"{symbol.lower()}_entropy.json")
        
        try:
            with open(tracker_path, 'w', encoding='utf-8') as f:
                json.dump(self.active_trackers[symbol], f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving tracker for {symbol}: {e}")
            return False
    
    def _load_tracker(self, symbol: str) -> Optional[Dict]:
        """Load a tracker from disk."""
        symbol = symbol.upper()
        
        tracker_path = os.path.join(self.log_dir, f"{symbol.lower()}_entropy.json")
        
        if not os.path.exists(tracker_path):
            return None
        
        try:
            with open(tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            return tracker
        except Exception as e:
            print(f"Error loading tracker for {symbol}: {e}")
            return None
    
    def _determine_emotional_valence(self, sentiment_arc: Dict[str, str]) -> str:
        """Determine overall emotional valence from sentiment arc."""
        if not sentiment_arc:
            return "neutral"
        
        # Get the most recent sentiment
        sorted_dates = sorted(sentiment_arc.keys())
        if not sorted_dates:
            return "neutral"
        
        latest_sentiment = sentiment_arc[sorted_dates[-1]]
        
        # Map sentiment states to emotional valence
        negative_states = ["fear", "confusion", "anger", "collapse_of_voice"]
        positive_states = ["interest", "narrative_reset"]
        complex_states = ["shock", "tribal_activation"]
        
        if latest_sentiment in negative_states:
            return "negative"
        elif latest_sentiment in positive_states:
            return "positive"
        elif latest_sentiment in complex_states:
            return "complex"
        else:
            return "neutral"
            
    def create_resonance_curves(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Create resonance curves comparing narrative arc with price movements.
        
        Args:
            symbols: List of ticker symbols to analyze
            
        Returns:
            Dict with resonance curve data for each symbol
        """
        results = {}
        
        for symbol in symbols:
            symbol = symbol.upper()
            
            if symbol not in self.active_trackers:
                print(f"Symbol {symbol} not tracked")
                continue
                
            tracker = self.active_trackers[symbol]
            
            # Get data points with both price and sentiment
            data_points = [p for p in tracker["data_points"] 
                          if "price" in p and ("sentiment_score" in p or "resonance_score" in p)]
            
            if len(data_points) < 2:
                print(f"Insufficient data for {symbol} resonance curve")
                continue
                
            # Sort by timestamp
            data_points = sorted(data_points, key=lambda x: x["timestamp"])
            
            # Extract dates, prices, and narrative metrics
            dates = [datetime.fromisoformat(p["timestamp"]).strftime("%Y-%m-%d") for p in data_points]
            prices = [p["price"] for p in data_points]
            
            # Use resonance_score if available, otherwise sentiment_score
            narrative_scores = []
            for p in data_points:
                if "resonance_score" in p:
                    narrative_scores.append(p["resonance_score"])
                elif "sentiment_score" in p:
                    # Transform sentiment (-1 to 1) to resonance (0 to 1) scale
                    # High absolute sentiment = high resonance
                    narrative_scores.append(abs(p["sentiment_score"]))
                else:
                    narrative_scores.append(0)
            
            # Calculate price changes
            price_changes = [0]
            for i in range(1, len(prices)):
                change = (prices[i] - prices[i-1]) / prices[i-1] if prices[i-1] != 0 else 0
                price_changes.append(change)
            
            # Calculate narrative leads/lags
            leads_lags = []
            if len(dates) >= 3:
                for i in range(1, len(dates)-1):
                    # Consider narrative leading price if high resonance precedes price move
                    if narrative_scores[i] > 0.6 and abs(price_changes[i+1]) > 0.02:
                        leads_lags.append({
                            "date": dates[i],
                            "type": "narrative_lead",
                            "resonance": narrative_scores[i],
                            "subsequent_price_change": price_changes[i+1]
                        })
                    # Consider price leading narrative if price move precedes resonance spike
                    elif abs(price_changes[i]) > 0.02 and narrative_scores[i+1] > 0.6:
                        leads_lags.append({
                            "date": dates[i],
                            "type": "price_lead",
                            "price_change": price_changes[i],
                            "subsequent_resonance": narrative_scores[i+1]
                        })
            
            # Calculate correlation between narrative and price
            if len(narrative_scores) >= 3:
                # Simple correlation (not perfect but indicative)
                try:
                    # Align lengths for lagged correlation
                    narrative_t0 = narrative_scores[:-1]  # Narrative at t
                    price_t1 = price_changes[1:]  # Price change at t+1
                    
                    # Calculate correlation
                    mean_narrative = sum(narrative_t0) / len(narrative_t0)
                    mean_price = sum(price_t1) / len(price_t1)
                    
                    numerator = sum((narrative_t0[i] - mean_narrative) * (price_t1[i] - mean_price) 
                                    for i in range(len(narrative_t0)))
                    
                    denominator_narrative = sum((x - mean_narrative) ** 2 for x in narrative_t0)
                    denominator_price = sum((x - mean_price) ** 2 for x in price_t1)
                    denominator = (denominator_narrative * denominator_price) ** 0.5
                    
                    correlation = numerator / denominator if denominator != 0 else 0
                except Exception as e:
                    print(f"Error calculating correlation for {symbol}: {e}")
                    correlation = 0
            else:
                correlation = 0
            
            # Store results
            results[symbol] = {
                "dates": dates,
                "prices": prices,
                "narrative_scores": narrative_scores,
                "price_changes": price_changes,
                "leads_lags": leads_lags,
                "narrative_price_correlation": correlation,
                "curve_data": {
                    "dates": dates,
                    "narrative_line": narrative_scores,
                    "price_line": [p / max(prices) for p in prices] if max(prices) > 0 else prices
                }
            }
            
            # Log insights to diary if correlation is strong
            if has_quantum_modules and abs(correlation) > 0.6:
                direction = "positive" if correlation > 0 else "negative"
                try:
                    register_insight(
                        label=f"{symbol} Strong Narrative-Price Correlation",
                        summary=f"Detected {direction} correlation ({correlation:.2f}) between narrative resonance and price movements",
                        phi_impact=min(abs(correlation) * 0.1, 0.1),
                        tags=["market", "resonance-curve", symbol, f"{direction}-correlation"]
                    )
                except Exception as e:
                    print(f"Error registering correlation insight: {e}")
        
        return results
    
    def tag_sentiment_inflection(self, symbol: str) -> List[Dict]:
        """
        Identify and tag sentiment inflection points.
        
        Args:
            symbol: The ticker or topic symbol
            
        Returns:
            List of identified inflection points
        """
        symbol = symbol.upper()
        
        if symbol not in self.active_trackers:
            print(f"Symbol {symbol} not tracked")
            return []
            
        tracker = self.active_trackers[symbol]
        
        # Need sufficient data
        if len(tracker["data_points"]) < 5:
            return []
            
        # Sort data points by timestamp
        data_points = sorted(tracker["data_points"], key=lambda x: x["timestamp"])
        
        # Extract sentiment values
        sentiment_data = [(p["timestamp"], p.get("sentiment_score", 0)) 
                         for p in data_points if "sentiment_score" in p]
        
        if len(sentiment_data) < 5:
            return []
            
        inflection_points = []
        
        # Find potential inflection points using windowed analysis
        window_size = 3
        for i in range(window_size, len(sentiment_data) - window_size):
            # Get current point and surrounding windows
            pre_window = [s for _, s in sentiment_data[i-window_size:i]]
            post_window = [s for _, s in sentiment_data[i+1:i+1+window_size]]
            current = sentiment_data[i][1]
            
            pre_avg = sum(pre_window) / len(pre_window)
            post_avg = sum(post_window) / len(post_window)
            
            # Check if this is a sentiment reversal point
            is_reversal = (pre_avg < current and post_avg < current) or (pre_avg > current and post_avg > current)
            
            # Check if this is a trend acceleration point
            pre_trend = pre_window[-1] - pre_window[0]
            post_trend = post_window[-1] - post_window[0]
            is_acceleration = abs(post_trend) > abs(pre_trend) * 1.5
            
            # Check if this is a trend deceleration/exhaustion point
            is_exhaustion = abs(pre_trend) > abs(post_trend) * 1.5
            
            # If any of these conditions is true, we have an inflection point
            if is_reversal or is_acceleration or is_exhaustion:
                # Determine the type
                if is_reversal:
                    inflection_type = "reversal"
                    if current > pre_avg:
                        subtype = "bottom"
                    else:
                        subtype = "top"
                elif is_acceleration:
                    inflection_type = "acceleration"
                    if post_trend > 0:
                        subtype = "bullish"
                    else:
                        subtype = "bearish"
                else:  # is_exhaustion
                    inflection_type = "exhaustion"
                    if pre_trend > 0:
                        subtype = "bullish_exhaustion"
                    else:
                        subtype = "bearish_exhaustion"
                
                # Create inflection point record
                inflection = {
                    "timestamp": sentiment_data[i][0],
                    "type": inflection_type,
                    "subtype": subtype,
                    "sentiment_value": current,
                    "pre_window_avg": pre_avg,
                    "post_window_avg": post_avg,
                    "pre_trend": pre_trend,
                    "post_trend": post_trend,
                    "confidence": 0.7
                }
                
                # Get price at inflection if available
                for p in data_points:
                    if p["timestamp"] == sentiment_data[i][0] and "price" in p:
                        inflection["price"] = p["price"]
                        break
                
                inflection_points.append(inflection)
                
                # For significant inflection points, create an event
                if (is_reversal and abs(current - pre_avg) > 0.3) or \
                   (is_acceleration and abs(post_trend) > abs(pre_trend) * 2):
                    # Create descriptive label and text
                    if inflection_type == "reversal":
                        label = f"{symbol} Sentiment {subtype.capitalize()}"
                        description = f"Detected potential sentiment {subtype} at {current:.2f}"
                    elif inflection_type == "acceleration":
                        label = f"{symbol} {subtype.capitalize()} Acceleration"
                        description = f"Sentiment trend accelerating in {subtype} direction"
                    else:
                        label = f"{symbol} {subtype.capitalize()}"
                        description = f"Sentiment trend showing signs of exhaustion after {pre_trend:.2f} move"
                    
                    # Add event
                    self.add_event(
                        symbol=symbol,
                        event_type="sentiment_inflection",
                        label=label,
                        timestamp=sentiment_data[i][0],
                        description=description,
                        tags=[inflection_type, subtype],
                        resonance_score=0.6,
                        actionable=True,
                        watch_for=["confirmation", "price reaction", "volume spike"]
                    )
        
        return inflection_points
    
    def build_conviction_index(self, symbol: str) -> Dict[str, Any]:
        """
        Build a conviction index based on sentiment divergence and agreement.
        
        Args:
            symbol: The ticker or topic symbol
            
        Returns:
            Dict with conviction index data
        """
        symbol = symbol.upper()
        
        if symbol not in self.active_trackers:
            print(f"Symbol {symbol} not tracked")
            return {}
            
        tracker = self.active_trackers[symbol]
        
        # Need sufficient data
        if len(tracker["data_points"]) < 3:
            return {"conviction_score": 0, "confidence": 0, "signal_type": "insufficient_data"}
            
        # Get recent data points
        data_points = sorted(tracker["data_points"], key=lambda x: x["timestamp"])[-10:]
        
        # Extract sentiment and source data
        sentiment_by_source = {}
        for point in data_points:
            if "sources" in point:
                timestamp = point["timestamp"]
                date_key = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
                
                for source, count in point["sources"].items():
                    if source not in sentiment_by_source:
                        sentiment_by_source[source] = {}
                    
                    # Store sentiment if available for this source
                    if "sentiment_score" in point:
                        sentiment_by_source[source][date_key] = point["sentiment_score"]
        
        # Need at least 2 sources for divergence analysis
        if len(sentiment_by_source) < 2:
            return {"conviction_score": 0, "confidence": 0, "signal_type": "insufficient_sources"}
            
        # Calculate divergence between sources
        source_pairs = []
        for i, source1 in enumerate(sentiment_by_source.keys()):
            for source2 in list(sentiment_by_source.keys())[i+1:]:
                # Find common dates
                common_dates = set(sentiment_by_source[source1].keys()) & set(sentiment_by_source[source2].keys())
                
                if common_dates:
                    # Calculate average divergence
                    divergences = [abs(sentiment_by_source[source1][date] - sentiment_by_source[source2][date]) 
                                 for date in common_dates]
                    avg_divergence = sum(divergences) / len(divergences)
                    
                    source_pairs.append({
                        "source1": source1,
                        "source2": source2,
                        "divergence": avg_divergence,
                        "common_dates_count": len(common_dates)
                    })
        
        if not source_pairs:
            return {"conviction_score": 0, "confidence": 0, "signal_type": "no_common_dates"}
            
        # Calculate average divergence across all pairs
        avg_divergence = sum(pair["divergence"] for pair in source_pairs) / len(source_pairs)
        
        # Assess latest sentiment direction
        latest_sentiments = []
        for source, dates in sentiment_by_source.items():
            if dates:
                latest_date = max(dates.keys())
                latest_sentiments.append(dates[latest_date])
        
        if not latest_sentiments:
            return {"conviction_score": 0, "confidence": 0, "signal_type": "no_recent_sentiment"}
            
        avg_sentiment = sum(latest_sentiments) / len(latest_sentiments)
        sentiment_agreement = len([s for s in latest_sentiments if (s > 0 and avg_sentiment > 0) or 
                                  (s < 0 and avg_sentiment < 0)]) / len(latest_sentiments)
        
        # Calculate conviction score
        # High divergence + high agreement = strong signal
        # Low divergence + high agreement = likely noise (herd behavior)
        conviction_score = avg_divergence * sentiment_agreement
        
        # Determine signal type
        if avg_divergence > 0.3:  # High divergence
            if sentiment_agreement > 0.7:  # High agreement
                signal_type = "strong_signal"
            else:
                signal_type = "mixed_signal"
        else:  # Low divergence
            if sentiment_agreement > 0.7:  # High agreement
                signal_type = "potential_herd_behavior"
            else:
                signal_type = "noise"
        
        # Calculate confidence based on data quality
        confidence = min(1.0, (len(source_pairs) / 10) * (sum(p["common_dates_count"] for p in source_pairs) / 
                                                       (len(source_pairs) * 10)))
        
        result = {
            "conviction_score": conviction_score,
            "avg_divergence": avg_divergence,
            "sentiment_agreement": sentiment_agreement,
            "avg_sentiment": avg_sentiment,
            "signal_type": signal_type,
            "confidence": confidence,
            "source_pair_count": len(source_pairs),
            "source_details": [{
                "sources": f"{p['source1']} vs {p['source2']}",
                "divergence": p["divergence"]
            } for p in source_pairs]
        }
        
        # Create event for strong signals
        if signal_type == "strong_signal" and conviction_score > 0.3:
            sentiment_direction = "positive" if avg_sentiment > 0 else "negative"
            self.add_event(
                symbol=symbol,
                event_type="conviction_signal",
                label=f"{symbol} Strong {sentiment_direction.capitalize()} Conviction Signal",
                description=f"High divergence ({avg_divergence:.2f}) with strong {sentiment_direction} sentiment agreement ({sentiment_agreement:.2f})",
                tags=["conviction", sentiment_direction, signal_type],
                resonance_score=min(conviction_score + 0.3, 0.9),
                actionable=True,
                watch_for=["price confirmation", "volume surge", "institutional moves"]
            )
        
        return result

# Singleton pattern
_entropy_tracker_instance = None

def get_entropy_tracker(log_dir: Optional[str] = None, 
                      narrative_bloom_path: Optional[str] = None,
                      debug_mode: bool = False) -> EntropyTracker:
    """
    Get or create singleton instance of the entropy tracker.
    
    Args:
        log_dir: Directory to store entropy logs
        narrative_bloom_path: Path to the narrative bloom JSON file
        debug_mode: Enable debug logging
        
    Returns:
        EntropyTracker instance
    """
    global _entropy_tracker_instance
    
    if _entropy_tracker_instance is None:
        _entropy_tracker_instance = EntropyTracker(
            log_dir=log_dir,
            narrative_bloom_path=narrative_bloom_path,
            debug_mode=debug_mode
        )
    
    return _entropy_tracker_instance

if __name__ == "__main__":
    # Example usage
    tracker = get_entropy_tracker(debug_mode=True)
    
    # Create or load NVDA tracker
    nvda_data = {
        "data_points": [
            {
                "timestamp": (datetime.now() - timedelta(days=6)).isoformat(),
                "mention_count": 1200,
                "sentiment_score": 0.2,
                "sentiment_state": "interest",
                "price": 950.25,
                "resonance_score": 0.3
            },
            {
                "timestamp": (datetime.now() - timedelta(days=5)).isoformat(),
                "mention_count": 4500,
                "sentiment_score": -0.3,
                "sentiment_state": "shock",
                "price": 840.10,
                "resonance_score": 0.65
            },
            {
                "timestamp": (datetime.now() - timedelta(days=4)).isoformat(),
                "mention_count": 7800,
                "sentiment_score": -0.6,
                "sentiment_state": "fear",
                "price": 801.50,
                "resonance_score": 0.75
            },
            {
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat(),
                "mention_count": 2100,
                "sentiment_score": -0.4,
                "sentiment_state": "confusion",
                "price": 788.20,
                "resonance_score": 0.45
            },
            {
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "mention_count": 5400,
                "sentiment_score": -0.5,
                "sentiment_state": "anger",
                "price": 790.80,
                "resonance_score": 0.70
            },
            {
                "timestamp": datetime.now().isoformat(),
                "mention_count": 800,
                "sentiment_score": -0.2,
                "sentiment_state": "collapse_of_voice",
                "price": 805.30,
                "resonance_score": 0.20
            }
        ]
    }
    
    # Create and populate NVDA tracker
    nvda = tracker.track_symbol("NVDA", "NVIDIA Corporation")
    for point in nvda_data["data_points"]:
        tracker.add_data_point("NVDA", **point)
    
    # Compute metrics
    metrics = tracker.compute_metrics("NVDA")
    print(f"NVDA Metrics: {metrics}")
    
    # Detect patterns
    patterns = tracker.detect_narrative_patterns("NVDA")
    print(f"Detected {len(patterns)} patterns")
    for pattern in patterns:
        print(f"  - {pattern['type']}: {pattern['confidence']:.2f} confidence")
    
    # Example event
    tracker.add_event(
        symbol="NVDA",
        event_type="narrative_echo_storm",
        label="NVDA_Narrative_Collision_Apr21",
        description="Market-disruptive news resulted in narrative collision between AI hype and market reality",
        tags=["NVDA", "short", "AI bubble", "retail sentiment", "echo fatigue"],
        sentiment_arc={
            "Apr 15": "shock",
            "Apr 16": "fear",
            "Apr 18": "confusion",
            "Apr 20": "anger",
            "Apr 21": "collapse_of_voice"
        },
        resonance_score=0.82,
        actionable=True,
        watch_for=["volatility", "media reframing", "price lag"]
    )
    
    print("EntropyTracker example completed")