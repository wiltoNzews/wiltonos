"""
Long Term Conviction Module for WiltonOS
----------------------------------------
Analyzes and builds conviction signals for long-term investment decisions.
Uses narrative patterns, language fractals, and emotional volatility arcs to find high-conviction opportunities.
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
DEFAULT_CONVICTION_LOG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'conviction_logs')

class ConvictionEngine:
    """
    Long-term conviction engine for investment decisions.
    """
    
    # Conviction level descriptors
    CONVICTION_LEVELS = {
        0.9: "Highest Conviction - Narrative Foundation Shift",
        0.8: "Very Strong Conviction - Emotional-Logical Alignment",
        0.7: "Strong Conviction - Institutional-Retail Alignment",
        0.6: "Moderate-High Conviction - Clear Directional Bias",
        0.5: "Moderate Conviction - Directional With Caveats",
        0.4: "Low-Moderate Conviction - Watching Closely",
        0.3: "Low Conviction - Early Pattern Detection",
        0.2: "Very Low Conviction - Potential Signal Formation",
        0.1: "Minimal Conviction - Observation Mode"
    }
    
    # Time horizon mapping
    TIME_HORIZONS = {
        0.9: "3-5 years",
        0.8: "2-3 years",
        0.7: "1-2 years",
        0.6: "9-12 months",
        0.5: "6-9 months",
        0.4: "3-6 months",
        0.3: "1-3 months",
        0.2: "2-4 weeks",
        0.1: "1-2 weeks"
    }
    
    def __init__(self, 
                log_dir: Optional[str] = None,
                use_entropy_tracker: bool = True,
                debug_mode: bool = False):
        """
        Initialize the conviction engine.
        
        Args:
            log_dir: Directory to store conviction logs
            use_entropy_tracker: Whether to use the entropy tracker
            debug_mode: Enable debug logging
        """
        self.log_dir = log_dir or DEFAULT_CONVICTION_LOG_PATH
        self.debug_mode = debug_mode
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize entropy tracker if requested
        self.entropy_tracker = get_entropy_tracker() if use_entropy_tracker and 'get_entropy_tracker' in globals() else None
        
        # Active conviction trackers
        self.conviction_trackers = {}
        
        # Position sizing model parameters
        self.position_sizing = {
            "max_position_size": 0.10,  # Maximum position size as percentage of portfolio
            "min_position_size": 0.01,  # Minimum position size as percentage of portfolio
            "conviction_scaling": 0.8,   # How much conviction affects position size
            "max_single_asset_class": 0.40,  # Maximum exposure to a single asset class
            "volatility_adjustment": 0.5  # How much volatility affects position size
        }
        
        print(f"ConvictionEngine initialized. Log directory: {self.log_dir}")
    
    def track_asset(self, 
                  symbol: str,
                  asset_class: str, 
                  description: Optional[str] = None,
                  initial_conviction: Optional[float] = None,
                  thesis: Optional[str] = None,
                  catalysts: Optional[List[Dict]] = None,
                  tags: Optional[List[str]] = None) -> Dict:
        """
        Start tracking an asset for long-term conviction.
        
        Args:
            symbol: Asset symbol (e.g., "NVDA", "BTC", "ETH")
            asset_class: Asset class (e.g., "equity", "crypto", "commodity")
            description: Asset description
            initial_conviction: Initial conviction score (0.0 to 1.0)
            thesis: Investment thesis
            catalysts: List of potential catalysts
            tags: List of tags for categorization
            
        Returns:
            The newly created conviction tracker
        """
        symbol = symbol.upper()
        
        # Check if already tracking
        if symbol in self.conviction_trackers:
            print(f"Already tracking asset {symbol}")
            return self.conviction_trackers[symbol]
        
        # Initialize entropy tracker for this symbol if available
        if self.entropy_tracker:
            try:
                self.entropy_tracker.track_symbol(symbol, description)
            except Exception as e:
                print(f"Error initializing entropy tracking for {symbol}: {e}")
        
        # Create new tracker
        tracker = {
            "symbol": symbol,
            "asset_class": asset_class,
            "description": description or f"Long-term conviction tracker for {symbol}",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "conviction_history": [],
            "current_conviction": initial_conviction or 0.3,  # Start with low-moderate conviction
            "thesis": thesis or f"Initial tracking for {symbol}",
            "catalysts": catalysts or [],
            "tags": tags or [asset_class],
            "metrics": {
                "max_conviction": initial_conviction or 0.3,
                "conviction_volatility": 0.0,
                "days_tracked": 0,
                "thesis_revisions": 0
            },
            "position_sizing": {
                "current_target": 0.0,
                "last_adjustment": datetime.now().isoformat(),
                "sizing_history": []
            }
        }
        
        # Add initial conviction point
        self._add_conviction_point(
            tracker=tracker,
            conviction=initial_conviction or 0.3,
            reasoning="Initial tracking setup",
            catalysts_updated=bool(catalysts)
        )
        
        # Add to conviction trackers
        self.conviction_trackers[symbol] = tracker
        
        # Save to disk
        self._save_tracker(symbol)
        
        print(f"Started tracking asset {symbol} with conviction level: {tracker['current_conviction']}")
        return tracker
    
    def update_conviction(self,
                        symbol: str,
                        new_conviction: float,
                        reasoning: str,
                        thesis_update: Optional[str] = None,
                        new_catalysts: Optional[List[Dict]] = None,
                        remove_catalysts: Optional[List[str]] = None) -> Dict:
        """
        Update conviction for a tracked asset.
        
        Args:
            symbol: Asset symbol
            new_conviction: New conviction score (0.0 to 1.0)
            reasoning: Reasoning for conviction change
            thesis_update: Updated investment thesis
            new_catalysts: New potential catalysts
            remove_catalysts: IDs of catalysts to remove
            
        Returns:
            Updated conviction tracker
        """
        symbol = symbol.upper()
        
        if symbol not in self.conviction_trackers:
            raise ValueError(f"Asset {symbol} not being tracked")
        
        tracker = self.conviction_trackers[symbol]
        
        # Enforce conviction range
        new_conviction = max(0.0, min(1.0, new_conviction))
        
        # Update conviction history
        old_conviction = tracker["current_conviction"]
        self._add_conviction_point(
            tracker=tracker,
            conviction=new_conviction,
            reasoning=reasoning,
            thesis_updated=bool(thesis_update),
            catalysts_updated=bool(new_catalysts) or bool(remove_catalysts)
        )
        
        # Update thesis if provided
        if thesis_update:
            tracker["thesis"] = thesis_update
            tracker["metrics"]["thesis_revisions"] += 1
        
        # Process catalysts
        if new_catalysts:
            for catalyst in new_catalysts:
                # Generate ID if not provided
                if "id" not in catalyst:
                    catalyst["id"] = f"catalyst_{len(tracker['catalysts'])}_{int(time.time())}"
                
                catalyst["added_at"] = datetime.now().isoformat()
                tracker["catalysts"].append(catalyst)
        
        if remove_catalysts:
            tracker["catalysts"] = [c for c in tracker["catalysts"] if c.get("id") not in remove_catalysts]
        
        # Update current conviction
        tracker["current_conviction"] = new_conviction
        tracker["updated_at"] = datetime.now().isoformat()
        
        # Update max conviction if applicable
        if new_conviction > tracker["metrics"]["max_conviction"]:
            tracker["metrics"]["max_conviction"] = new_conviction
        
        # Update days tracked
        created = datetime.fromisoformat(tracker["created_at"])
        days_tracked = (datetime.now() - created).days
        tracker["metrics"]["days_tracked"] = days_tracked
        
        # Update conviction volatility (standard deviation of changes)
        if len(tracker["conviction_history"]) >= 3:
            changes = [
                abs(
                    tracker["conviction_history"][i]["conviction"] - 
                    tracker["conviction_history"][i-1]["conviction"]
                ) for i in range(1, len(tracker["conviction_history"]))
            ]
            tracker["metrics"]["conviction_volatility"] = sum(changes) / len(changes)
        
        # Recalculate position sizing
        self._calculate_position_size(tracker)
        
        # Save changes
        self._save_tracker(symbol)
        
        # Log significant conviction changes
        if abs(new_conviction - old_conviction) >= 0.2 and has_quantum_modules:
            direction = "increase" if new_conviction > old_conviction else "decrease"
            try:
                register_insight(
                    label=f"{symbol} Significant Conviction {direction.capitalize()}",
                    summary=f"Conviction {'increased' if direction == 'increase' else 'decreased'} from {old_conviction:.2f} to {new_conviction:.2f}\nReasoning: {reasoning}",
                    phi_impact=0.07,
                    tags=["conviction", direction, symbol] + tracker["tags"]
                )
            except Exception as e:
                print(f"Error registering conviction change in quantum diary: {e}")
        
        print(f"Updated {symbol} conviction to {new_conviction:.2f} (from {old_conviction:.2f})")
        return tracker
    
    def get_conviction_details(self, symbol: str) -> Dict:
        """
        Get detailed conviction information for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Detailed conviction information
        """
        symbol = symbol.upper()
        
        if symbol not in self.conviction_trackers:
            raise ValueError(f"Asset {symbol} not being tracked")
        
        tracker = self.conviction_trackers[symbol]
        
        # Get additional data from entropy tracker if available
        entropy_data = {}
        if self.entropy_tracker:
            try:
                # Get metrics, patterns, and latest event
                entropy_data["metrics"] = self.entropy_tracker.compute_metrics(symbol)
                patterns = self.entropy_tracker.detect_narrative_patterns(symbol)
                if patterns:
                    entropy_data["patterns"] = patterns
                
                # Get most recent event
                active_tracker = self.entropy_tracker.active_trackers.get(symbol, {})
                events = active_tracker.get("events", [])
                if events:
                    latest_event = max(events, key=lambda e: e.get("timestamp", ""))
                    entropy_data["latest_event"] = latest_event
            except Exception as e:
                print(f"Error getting entropy data for {symbol}: {e}")
        
        # Compute time horizon based on conviction
        conviction = tracker["current_conviction"]
        time_horizon = self._get_time_horizon(conviction)
        
        # Determine conviction level descriptor
        conviction_level = self._get_conviction_level(conviction)
        
        # Create detailed response
        details = {
            "symbol": symbol,
            "asset_class": tracker["asset_class"],
            "current_conviction": conviction,
            "conviction_level": conviction_level,
            "time_horizon": time_horizon,
            "thesis": tracker["thesis"],
            "target_position_size": tracker["position_sizing"]["current_target"],
            "catalysts": tracker["catalysts"],
            "days_tracked": tracker["metrics"]["days_tracked"],
            "conviction_history": sorted(tracker["conviction_history"], key=lambda x: x["timestamp"])[-10:],
            "entropy_data": entropy_data
        }
        
        return details
    
    def get_portfolio_recommendations(self, asset_classes: Optional[List[str]] = None) -> Dict:
        """
        Get portfolio recommendations based on current convictions.
        
        Args:
            asset_classes: Optional filter for specific asset classes
            
        Returns:
            Portfolio recommendations
        """
        if not self.conviction_trackers:
            return {"status": "no_assets", "allocations": [], "summary": "No assets being tracked"}
        
        # Filter by asset class if specified
        trackers = self.conviction_trackers
        if asset_classes:
            trackers = {k: v for k, v in trackers.items() if v["asset_class"] in asset_classes}
        
        # Get current date for reference
        today = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate total conviction-weighted allocation
        total_weight = sum(t["position_sizing"]["current_target"] for t in trackers.values())
        
        # Handle case where total weight exceeds 1.0 (100%)
        scaling_factor = 1.0
        if total_weight > 1.0:
            scaling_factor = 1.0 / total_weight
        
        # Build allocation recommendations
        allocations = []
        for symbol, tracker in trackers.items():
            conviction = tracker["current_conviction"]
            raw_allocation = tracker["position_sizing"]["current_target"]
            scaled_allocation = raw_allocation * scaling_factor
            
            allocations.append({
                "symbol": symbol,
                "asset_class": tracker["asset_class"],
                "conviction": conviction,
                "conviction_level": self._get_conviction_level(conviction),
                "allocation": scaled_allocation,
                "time_horizon": self._get_time_horizon(conviction),
            })
        
        # Sort by conviction descending
        allocations = sorted(allocations, key=lambda x: x["conviction"], reverse=True)
        
        # Calculate asset class diversification
        asset_class_allocations = {}
        for alloc in allocations:
            asset_class = alloc["asset_class"]
            if asset_class not in asset_class_allocations:
                asset_class_allocations[asset_class] = 0
            asset_class_allocations[asset_class] += alloc["allocation"]
        
        # Extract high, medium and low conviction assets
        high_conviction = [a for a in allocations if a["conviction"] >= 0.7]
        medium_conviction = [a for a in allocations if 0.4 <= a["conviction"] < 0.7]
        low_conviction = [a for a in allocations if a["conviction"] < 0.4]
        
        # Build recommendation summary
        cash_allocation = max(0, 1.0 - sum(a["allocation"] for a in allocations))
        summary = {
            "date": today,
            "total_assets_tracked": len(allocations),
            "high_conviction_count": len(high_conviction),
            "medium_conviction_count": len(medium_conviction),
            "low_conviction_count": len(low_conviction),
            "cash_allocation": cash_allocation,
            "asset_class_diversification": asset_class_allocations,
            "overall_conviction": sum(a["conviction"] * a["allocation"] for a in allocations) / (1 - cash_allocation) if (1 - cash_allocation) > 0 else 0
        }
        
        return {
            "status": "success",
            "allocations": allocations,
            "summary": summary
        }
    
    def _add_conviction_point(self, 
                            tracker: Dict, 
                            conviction: float,
                            reasoning: str,
                            thesis_updated: bool = False,
                            catalysts_updated: bool = False) -> None:
        """Add a conviction history point to a tracker."""
        point = {
            "timestamp": datetime.now().isoformat(),
            "conviction": conviction,
            "reasoning": reasoning,
            "thesis_updated": thesis_updated,
            "catalysts_updated": catalysts_updated
        }
        
        tracker["conviction_history"].append(point)
    
    def _calculate_position_size(self, tracker: Dict) -> float:
        """Calculate position sizing based on conviction and other factors."""
        conviction = tracker["current_conviction"]
        
        # Base calculation: scales with conviction
        base_size = (
            self.position_sizing["min_position_size"] + 
            (self.position_sizing["max_position_size"] - self.position_sizing["min_position_size"]) * 
            conviction ** self.position_sizing["conviction_scaling"]
        )
        
        # Apply volatility adjustment if available
        volatility_adj = 1.0
        if tracker["metrics"]["conviction_volatility"] > 0:
            volatility_adj = math.exp(-self.position_sizing["volatility_adjustment"] * tracker["metrics"]["conviction_volatility"])
        
        # Calculate final position size
        position_size = base_size * volatility_adj
        
        # Record in position sizing history
        sizing_point = {
            "timestamp": datetime.now().isoformat(),
            "position_size": position_size,
            "conviction": conviction,
            "volatility_adjustment": volatility_adj
        }
        
        tracker["position_sizing"]["current_target"] = position_size
        tracker["position_sizing"]["last_adjustment"] = datetime.now().isoformat()
        tracker["position_sizing"]["sizing_history"].append(sizing_point)
        
        return position_size
    
    def _get_time_horizon(self, conviction: float) -> str:
        """Get time horizon based on conviction level."""
        # Find the closest conviction level
        levels = sorted(self.TIME_HORIZONS.keys())
        closest_level = min(levels, key=lambda x: abs(x - conviction))
        
        return self.TIME_HORIZONS[closest_level]
    
    def _get_conviction_level(self, conviction: float) -> str:
        """Get conviction level descriptor based on conviction score."""
        levels = sorted(self.CONVICTION_LEVELS.keys())
        closest_level = min(levels, key=lambda x: abs(x - conviction))
        
        return self.CONVICTION_LEVELS[closest_level]
    
    def _save_tracker(self, symbol: str) -> bool:
        """Save a tracker to disk."""
        symbol = symbol.upper()
        
        if symbol not in self.conviction_trackers:
            return False
        
        tracker_path = os.path.join(self.log_dir, f"{symbol.lower()}_conviction.json")
        
        try:
            with open(tracker_path, 'w', encoding='utf-8') as f:
                json.dump(self.conviction_trackers[symbol], f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving tracker for {symbol}: {e}")
            return False
    
    def _load_tracker(self, symbol: str) -> Optional[Dict]:
        """Load a tracker from disk."""
        symbol = symbol.upper()
        
        tracker_path = os.path.join(self.log_dir, f"{symbol.lower()}_conviction.json")
        
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
            if filename.endswith("_conviction.json"):
                symbol = filename.split("_conviction.json")[0].upper()
                tracker = self._load_tracker(symbol)
                
                if tracker:
                    self.conviction_trackers[symbol] = tracker
                    count += 1
        
        print(f"Loaded {count} conviction trackers from disk")
        return count

# Singleton pattern
_conviction_engine_instance = None

def get_conviction_engine(log_dir: Optional[str] = None,
                        use_entropy_tracker: bool = True,
                        debug_mode: bool = False) -> ConvictionEngine:
    """
    Get or create singleton instance of the conviction engine.
    
    Args:
        log_dir: Directory to store conviction logs
        use_entropy_tracker: Whether to use the entropy tracker
        debug_mode: Enable debug logging
        
    Returns:
        ConvictionEngine instance
    """
    global _conviction_engine_instance
    
    if _conviction_engine_instance is None:
        _conviction_engine_instance = ConvictionEngine(
            log_dir=log_dir,
            use_entropy_tracker=use_entropy_tracker,
            debug_mode=debug_mode
        )
    
    return _conviction_engine_instance

if __name__ == "__main__":
    # Example usage
    engine = get_conviction_engine(debug_mode=True)
    
    # Track Bitcoin with initial conviction
    engine.track_asset(
        symbol="BTC",
        asset_class="crypto",
        description="Bitcoin - Digital gold",
        initial_conviction=0.6,
        thesis="Bitcoin is positioned as digital gold and inflation hedge with growing institutional adoption",
        catalysts=[
            {
                "id": "btc_cat_1",
                "title": "Halving Event",
                "description": "Bitcoin halving event typically reduces new supply and has historically led to price increases",
                "expected_date": "April 2024",
                "significance": "high",
                "impact_conviction": 0.2
            },
            {
                "id": "btc_cat_2",
                "title": "Institutional Adoption",
                "description": "Increasing institutional adoption through ETFs and corporate treasury allocations",
                "expected_date": "Ongoing",
                "significance": "high",
                "impact_conviction": 0.25
            }
        ],
        tags=["crypto", "bitcoin", "inflation-hedge", "digital-gold"]
    )
    
    # Track Ethereum with initial conviction
    engine.track_asset(
        symbol="ETH",
        asset_class="crypto",
        description="Ethereum - Programmable money and smart contract platform",
        initial_conviction=0.7,
        thesis="Ethereum's position as the leading smart contract platform with EIP-1559 reducing supply",
        catalysts=[
            {
                "id": "eth_cat_1",
                "title": "ETH 2.0 Final Phase",
                "description": "Completion of ETH 2.0 upgrade",
                "expected_date": "2025",
                "significance": "high",
                "impact_conviction": 0.3
            }
        ],
        tags=["crypto", "ethereum", "smart-contracts", "defi"]
    )
    
    # Track NVDA with initial conviction
    engine.track_asset(
        symbol="NVDA",
        asset_class="equity",
        description="NVIDIA Corporation - AI and GPU leader",
        initial_conviction=0.75,
        thesis="Leading position in AI chip market with strong moat through CUDA and developer ecosystem",
        catalysts=[
            {
                "id": "nvda_cat_1",
                "title": "New AI Chip Release",
                "description": "Next-generation AI architecture release",
                "expected_date": "2024-Q4",
                "significance": "high",
                "impact_conviction": 0.2
            }
        ],
        tags=["equity", "tech", "ai", "semiconductors"]
    )
    
    # Get portfolio recommendations
    recommendations = engine.get_portfolio_recommendations()
    print("\nPortfolio Recommendations:")
    print(f"Overall conviction: {recommendations['summary']['overall_conviction']:.2f}")
    print(f"Asset class diversification: {recommendations['summary']['asset_class_diversification']}")
    print("\nAllocations:")
    for alloc in recommendations["allocations"]:
        print(f"  {alloc['symbol']}: {alloc['allocation']*100:.1f}% (Conviction: {alloc['conviction']:.2f}, {alloc['conviction_level']})")
    
    print(f"\nCash allocation: {recommendations['summary']['cash_allocation']*100:.1f}%")
    
    # Update conviction based on new thesis
    engine.update_conviction(
        symbol="BTC",
        new_conviction=0.68,
        reasoning="Increasing institutional adoption through ETFs shows stronger momentum than expected",
        thesis_update="Bitcoin's role as a non-sovereign store of value is being validated through increased institutional acceptance"
    )
    
    # Get updated details
    btc_details = engine.get_conviction_details("BTC")
    print(f"\nUpdated BTC conviction: {btc_details['current_conviction']:.2f}")
    print(f"Time horizon: {btc_details['time_horizon']}")
    print(f"Target position size: {btc_details['target_position_size']*100:.1f}%")