"""
Investment Engine for WiltonOS
------------------------------
Master module that integrates long-term conviction and short-term sentiment signals.
Provides a unified interface for investment decisions while maintaining the 3:1 ratio 
of conviction to exploration (75% conviction-based, 25% sentiment-driven).
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

# Import core modules if available
try:
    from wilton_core.memory.quantum_diary import add_diary_entry, register_insight
    from wiltonos.market.entropy_tracker import get_entropy_tracker
    from wiltonos.market.long_term_conviction import get_conviction_engine
    from wiltonos.market.short_term_sentiment import get_sentiment_engine
    has_market_modules = True
except ImportError as e:
    print(f"Error importing market modules: {e}")
    has_market_modules = False

# Default paths
DEFAULT_ENGINE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'data', 'investment_engine_config.json')
DEFAULT_STATE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'investment_engine_state.json')

class InvestmentEngine:
    """
    Master investment engine that integrates conviction and sentiment signals.
    """
    
    # Constants
    QUANTUM_RATIO = 0.75  # 3:1 ratio (75% conviction / 25% exploration)
    
    # Investment modes
    INVESTMENT_MODES = {
        "balanced": "Default 75/25 conviction/exploration ratio",
        "conviction_heavy": "85/15 conviction/exploration ratio",
        "exploration_heavy": "60/40 conviction/exploration ratio",
        "enjoyment_mode": "Special mode with dynamic conviction/exploration ratio",
        "pause": "Pause automatic investment decisions"
    }
    
    def __init__(self, 
                config_path: Optional[str] = None,
                state_path: Optional[str] = None,
                debug_mode: bool = False):
        """
        Initialize the investment engine.
        
        Args:
            config_path: Path to configuration file
            state_path: Path to engine state file
            debug_mode: Enable debug logging
        """
        self.config_path = config_path or DEFAULT_ENGINE_CONFIG_PATH
        self.state_path = state_path or DEFAULT_STATE_PATH
        self.debug_mode = debug_mode
        
        # Load configuration
        self.config = self._load_config()
        
        # Load or initialize state
        self.state = self._load_state()
        
        # Initialize sub-engines if available
        if has_market_modules:
            self.entropy_tracker = get_entropy_tracker(debug_mode=debug_mode)
            self.conviction_engine = get_conviction_engine(debug_mode=debug_mode)
            self.sentiment_engine = get_sentiment_engine(debug_mode=debug_mode)
        else:
            self.entropy_tracker = None
            self.conviction_engine = None
            self.sentiment_engine = None
            print("Warning: Market modules not available. Investment Engine running in limited mode.")
        
        # Initialize ledger connection
        self.ledger_connected = False
        if self.config.get("use_ledger", False):
            self.ledger_connected = self._initialize_ledger()
        
        # Initialize broker connection
        self.broker_connected = False
        if self.config.get("use_broker", False):
            self.broker_connected = self._initialize_broker()
        
        print(f"InvestmentEngine initialized in {self.state['current_mode']} mode")
    
    def _load_config(self) -> Dict:
        """Load engine configuration."""
        default_config = {
            "use_ledger": False,
            "use_broker": False,
            "default_mode": "balanced",
            "auto_save_state": True,
            "save_interval_minutes": 10,
            "auto_track_assets": True,
            "tracked_assets": [
                {"symbol": "BTC", "asset_class": "crypto", "description": "Bitcoin"},
                {"symbol": "ETH", "asset_class": "crypto", "description": "Ethereum"},
                {"symbol": "NVDA", "asset_class": "equity", "description": "NVIDIA Corporation"}
            ],
            "enjoyment_mode": {
                "min_conviction_ratio": 0.60,
                "max_conviction_ratio": 0.85,
                "adjustment_interval_days": 7,
                "performance_sensitive": True
            },
            "ledger_config": {
                "use_webhook": False,
                "api_endpoint": "",
                "query_interval_minutes": 60
            },
            "broker_config": {
                "api_endpoint": "",
                "api_version": "v1",
                "use_paper_trading": True
            },
            "execution_limits": {
                "max_daily_trades": 5,
                "max_position_value": 0.20,  # Maximum position as % of portfolio
                "min_trade_interval_hours": 24
            }
        }
        
        # Try to load from file
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults, keeping loaded values
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(default_config[key], dict) and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                print(f"Error loading config: {e}")
        else:
            # Save default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config
    
    def _load_state(self) -> Dict:
        """Load or initialize engine state."""
        default_state = {
            "current_mode": self.config.get("default_mode", "balanced"),
            "enjoyment_mode_active": False,
            "enjoyment_mode_toggled_at": None,
            "current_quantum_ratio": self.QUANTUM_RATIO,
            "last_state_save": datetime.now().isoformat(),
            "last_trade": None,
            "portfolio_snapshot": {},
            "last_recommendations": {},
            "execution_stats": {
                "trades_today": 0,
                "trades_week": 0,
                "trades_month": 0,
                "last_trade_date": None
            },
            "last_ledger_sync": None,
            "prediction_bias_log": []
        }
        
        # Try to load from file
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    
                    # If loading succeeded, use loaded state but ensure all keys exist
                    for key, value in default_state.items():
                        if key not in state:
                            state[key] = value
                    
                    return state
            except Exception as e:
                print(f"Error loading state: {e}")
        
        # Save default state
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path, 'w', encoding='utf-8') as f:
            json.dump(default_state, f, indent=2)
        
        return default_state
    
    def _save_state(self) -> bool:
        """Save current engine state."""
        try:
            # Update last save timestamp
            self.state["last_state_save"] = datetime.now().isoformat()
            
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def _initialize_ledger(self) -> bool:
        """Initialize connection to Ledger."""
        # This is a placeholder for actual Ledger integration
        ledger_config = self.config.get("ledger_config", {})
        
        if ledger_config.get("use_webhook", False):
            # Webhook mode - Ledger will send updates to us
            print("Ledger webhook mode enabled")
            return True
        else:
            # API mode - we'll query Ledger periodically
            api_endpoint = ledger_config.get("api_endpoint", "")
            if not api_endpoint:
                print("Ledger API endpoint not configured")
                return False
            
            # In a real implementation, validate API connection here
            print(f"Ledger API mode enabled with endpoint: {api_endpoint}")
            return True
    
    def _initialize_broker(self) -> bool:
        """Initialize connection to trading broker."""
        # This is a placeholder for actual broker integration
        broker_config = self.config.get("broker_config", {})
        api_endpoint = broker_config.get("api_endpoint", "")
        
        if not api_endpoint:
            print("Broker API endpoint not configured")
            return False
        
        # In a real implementation, validate broker API connection here
        use_paper = broker_config.get("use_paper_trading", True)
        mode = "paper trading" if use_paper else "live trading"
        print(f"Broker connection initialized in {mode} mode")
        return True
    
    def get_status(self) -> Dict:
        """
        Get current engine status.
        
        Returns:
            Dict with engine status details
        """
        # Determine sub-engine status
        entropy_status = "active" if self.entropy_tracker else "unavailable"
        conviction_status = "active" if self.conviction_engine else "unavailable"
        sentiment_status = "active" if self.sentiment_engine else "unavailable"
        
        # Get connected assets count
        tracked_assets = 0
        if self.conviction_engine:
            tracked_assets = len(self.conviction_engine.conviction_trackers)
        
        # Calculate uptime
        if "startup_time" not in self.state:
            self.state["startup_time"] = datetime.now().isoformat()
            self._save_state()
        
        startup = datetime.fromisoformat(self.state["startup_time"])
        uptime_seconds = (datetime.now() - startup).total_seconds()
        
        # Format uptime
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(days)}d {int(hours)}h {int(minutes)}m"
        
        return {
            "status": "running",
            "mode": self.state["current_mode"],
            "enjoyment_mode_active": self.state["enjoyment_mode_active"],
            "quantum_ratio": self.state["current_quantum_ratio"],
            "tracked_assets": tracked_assets,
            "ledger_connected": self.ledger_connected,
            "broker_connected": self.broker_connected,
            "sub_engines": {
                "entropy_tracker": entropy_status,
                "conviction_engine": conviction_status,
                "sentiment_engine": sentiment_status
            },
            "execution_stats": self.state["execution_stats"],
            "uptime": uptime_str,
            "last_state_save": self.state["last_state_save"]
        }
    
    def toggle_enjoyment_mode(self, activate: bool) -> Dict:
        """
        Toggle enjoyment mode on or off.
        
        Args:
            activate: Whether to activate enjoyment mode
            
        Returns:
            Dict with updated engine status
        """
        # Don't do anything if already in desired state
        if self.state["enjoyment_mode_active"] == activate:
            return {
                "status": "unchanged",
                "message": f"Enjoyment mode already {'active' if activate else 'inactive'}",
                "mode": self.state["current_mode"],
                "enjoyment_mode_active": self.state["enjoyment_mode_active"]
            }
        
        # Update enjoyment mode state
        self.state["enjoyment_mode_active"] = activate
        self.state["enjoyment_mode_toggled_at"] = datetime.now().isoformat()
        
        if activate:
            # Set mode to enjoyment_mode
            self.state["current_mode"] = "enjoyment_mode"
            
            # Record prediction bias before activation
            self._record_prediction_bias("pre_enjoyment")
            
            # Log in quantum diary if available
            if has_market_modules:
                try:
                    register_insight(
                        label="Enjoyment Mode Activated",
                        summary="Investment engine toggled to Enjoyment Mode, enabling dynamic conviction/exploration ratio adjustment.",
                        phi_impact=0.08,
                        tags=["investment_engine", "enjoyment_mode", "toggle", "execution_singularity"]
                    )
                except Exception as e:
                    print(f"Error logging enjoyment mode activation: {e}")
        else:
            # Set mode back to balanced
            self.state["current_mode"] = "balanced"
            self.state["current_quantum_ratio"] = self.QUANTUM_RATIO
            
            # Record prediction bias after deactivation
            self._record_prediction_bias("post_enjoyment")
            
            # Log in quantum diary if available
            if has_market_modules:
                try:
                    register_insight(
                        label="Enjoyment Mode Deactivated",
                        summary="Investment engine switched back from Enjoyment Mode to Balanced mode.",
                        phi_impact=0.04,
                        tags=["investment_engine", "enjoyment_mode", "toggle", "execution_recalibration"]
                    )
                except Exception as e:
                    print(f"Error logging enjoyment mode deactivation: {e}")
        
        # Save updated state
        self._save_state()
        
        print(f"Enjoyment mode {'activated' if activate else 'deactivated'}")
        
        return {
            "status": "changed",
            "message": f"Enjoyment mode {'activated' if activate else 'deactivated'}",
            "mode": self.state["current_mode"],
            "enjoyment_mode_active": self.state["enjoyment_mode_active"],
            "toggled_at": self.state["enjoyment_mode_toggled_at"]
        }
    
    def get_investment_recommendations(self, 
                                      asset_classes: Optional[List[str]] = None) -> Dict:
        """
        Get comprehensive investment recommendations combining conviction and sentiment.
        
        Args:
            asset_classes: Optional list of asset classes to filter by
            
        Returns:
            Dict with comprehensive investment recommendations
        """
        if not has_market_modules:
            return {
                "status": "error",
                "message": "Market modules not available",
                "recommendations": []
            }
        
        # Get recommendation parts
        try:
            # Get long-term conviction recommendations
            long_term = self.conviction_engine.get_portfolio_recommendations(asset_classes)
            
            # Get active short-term signals
            short_term_signals = self.sentiment_engine.get_active_signals(min_strength=0.6)
            
            # Group short-term signals by symbol
            short_term_by_symbol = {}
            for signal in short_term_signals:
                symbol = signal["symbol"]
                if symbol not in short_term_by_symbol:
                    short_term_by_symbol[symbol] = []
                short_term_by_symbol[symbol].append(signal)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error getting recommendations: {str(e)}",
                "recommendations": []
            }
        
        # Get current mode and quantum ratio
        mode = self.state["current_mode"]
        quantum_ratio = self.state["current_quantum_ratio"]
        
        # Create integrated recommendations
        recommendations = []
        for allocation in long_term["allocations"]:
            symbol = allocation["symbol"]
            conviction = allocation["conviction"]
            
            # Get short-term signals for this symbol
            signals = short_term_by_symbol.get(symbol, [])
            
            # Determine short-term sentiment direction and strength
            sentiment_direction = "neutral"
            sentiment_strength = 0.0
            
            if signals:
                # Use strongest signal as primary indicator
                strongest = max(signals, key=lambda s: s["strength_value"])
                sentiment_direction = strongest["direction"]
                sentiment_strength = strongest["strength_value"]
            
            # Calculate adjustment based on quantum ratio
            conviction_weight = quantum_ratio
            sentiment_weight = 1.0 - quantum_ratio
            
            # Compute tactical allocation adjustment
            base_allocation = allocation["allocation"]
            
            # Determine directional alignment
            aligned = False
            if (sentiment_direction in ["positive", "acceleration", "bullish"] and conviction > 0.5) or \
               (sentiment_direction in ["negative", "deceleration", "bearish"] and conviction < 0.5):
                aligned = True
            
            # Calculate tactical adjustment factor
            tactical_adjustment = 0.0
            if aligned:
                # Positive adjustment when signals align
                tactical_adjustment = sentiment_strength * 0.2  # Up to 20% increase
            elif sentiment_direction != "neutral" and sentiment_strength > 0.6:
                # Negative adjustment when strong signals contradict
                tactical_adjustment = -sentiment_strength * 0.15  # Up to 15% decrease
            
            # Apply tactical adjustment within quantum ratio bounds
            tactical_allocation = base_allocation * (1 + tactical_adjustment)
            
            # Blend based on quantum ratio
            final_allocation = (base_allocation * conviction_weight) + (tactical_allocation * sentiment_weight)
            
            # Create recommendation
            recommendation = {
                "symbol": symbol,
                "asset_class": allocation["asset_class"],
                "conviction": conviction,
                "conviction_level": allocation["conviction_level"],
                "time_horizon": allocation["time_horizon"],
                "base_allocation": base_allocation,
                "tactical_adjustment": tactical_adjustment,
                "final_allocation": final_allocation,
                "sentiment": {
                    "direction": sentiment_direction,
                    "strength": sentiment_strength,
                    "signals_count": len(signals),
                    "signals": [{"type": s["type"], "strength": s["strength"], "description": s["description"]} 
                              for s in signals[:3]]  # Include top 3 signals
                },
                "alignment": "aligned" if aligned else "divergent"
            }
            
            recommendations.append(recommendation)
        
        # Sort by final allocation descending
        recommendations = sorted(recommendations, key=lambda x: x["final_allocation"], reverse=True)
        
        # Calculate allocations by asset class
        asset_class_allocations = {}
        for rec in recommendations:
            asset_class = rec["asset_class"]
            if asset_class not in asset_class_allocations:
                asset_class_allocations[asset_class] = 0
            asset_class_allocations[asset_class] += rec["final_allocation"]
        
        # Calculate remaining cash allocation
        total_allocated = sum(rec["final_allocation"] for rec in recommendations)
        cash_allocation = max(0, 1.0 - total_allocated)
        
        # Create result
        result = {
            "status": "success",
            "quantum_ratio": quantum_ratio,
            "mode": mode,
            "recommendations": recommendations,
            "asset_class_allocations": asset_class_allocations,
            "cash_allocation": cash_allocation,
            "generated_at": datetime.now().isoformat()
        }
        
        # Store last recommendations
        self.state["last_recommendations"] = result
        self._save_state()
        
        return result
    
    def set_investment_mode(self, mode: str) -> Dict:
        """
        Set the investment mode.
        
        Args:
            mode: Mode to set (balanced, conviction_heavy, exploration_heavy, enjoyment_mode, pause)
            
        Returns:
            Dict with status and updated mode
        """
        if mode not in self.INVESTMENT_MODES:
            return {
                "status": "error",
                "message": f"Invalid mode: {mode}. Valid modes are: {', '.join(self.INVESTMENT_MODES.keys())}",
                "current_mode": self.state["current_mode"]
            }
        
        old_mode = self.state["current_mode"]
        
        # Special handling for enjoyment mode
        if mode == "enjoyment_mode":
            toggle_result = self.toggle_enjoyment_mode(True)
            return {
                "status": "success",
                "message": f"Mode changed from {old_mode} to {mode}",
                "enjoyment_toggle_result": toggle_result
            }
        
        # Turn off enjoyment mode if switching to another mode
        if self.state["enjoyment_mode_active"]:
            self.toggle_enjoyment_mode(False)
        
        # Set mode
        self.state["current_mode"] = mode
        
        # Update quantum ratio based on mode
        if mode == "balanced":
            self.state["current_quantum_ratio"] = self.QUANTUM_RATIO
        elif mode == "conviction_heavy":
            self.state["current_quantum_ratio"] = 0.85
        elif mode == "exploration_heavy":
            self.state["current_quantum_ratio"] = 0.60
        elif mode == "pause":
            # Don't change quantum ratio when paused
            pass
        
        # Save state
        self._save_state()
        
        return {
            "status": "success",
            "message": f"Mode changed from {old_mode} to {mode}",
            "current_mode": mode,
            "quantum_ratio": self.state["current_quantum_ratio"]
        }
    
    def track_asset(self,
                  symbol: str,
                  asset_class: str,
                  description: Optional[str] = None,
                  initial_conviction: Optional[float] = None,
                  initial_thesis: Optional[str] = None) -> Dict:
        """
        Start tracking a new asset.
        
        Args:
            symbol: Asset symbol
            asset_class: Asset class (equity, crypto, commodity, etc.)
            description: Asset description
            initial_conviction: Initial conviction score
            initial_thesis: Initial investment thesis
            
        Returns:
            Dict with tracking status
        """
        if not has_market_modules:
            return {
                "status": "error",
                "message": "Market modules not available",
                "tracked": False
            }
        
        symbol = symbol.upper()
        
        try:
            # Start entropy tracking
            self.entropy_tracker.track_symbol(symbol, description)
            
            # Start conviction tracking
            self.conviction_engine.track_asset(
                symbol=symbol,
                asset_class=asset_class,
                description=description,
                initial_conviction=initial_conviction,
                thesis=initial_thesis,
                tags=[asset_class]
            )
            
            # Start sentiment tracking
            self.sentiment_engine.track_asset(
                symbol=symbol,
                description=description,
                tags=[asset_class]
            )
            
            return {
                "status": "success",
                "message": f"Started tracking {symbol}",
                "tracked": True,
                "symbol": symbol,
                "asset_class": asset_class
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error tracking asset: {str(e)}",
                "tracked": False
            }
    
    def update_portfolio_snapshot(self, portfolio_data: Dict[str, Any]) -> Dict:
        """
        Update portfolio snapshot with current holdings and values.
        
        Args:
            portfolio_data: Dict with portfolio holdings and values
            
        Returns:
            Dict with update status
        """
        try:
            # Store snapshot with timestamp
            portfolio_data["timestamp"] = datetime.now().isoformat()
            self.state["portfolio_snapshot"] = portfolio_data
            self._save_state()
            
            return {
                "status": "success",
                "message": "Portfolio snapshot updated",
                "timestamp": portfolio_data["timestamp"]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error updating portfolio snapshot: {str(e)}"
            }
    
    def get_portfolio_snapshot(self) -> Dict:
        """
        Get the current portfolio snapshot.
        
        Returns:
            Dict with portfolio snapshot
        """
        snapshot = self.state.get("portfolio_snapshot", {})
        
        if not snapshot:
            return {
                "status": "empty",
                "message": "No portfolio snapshot available"
            }
        
        return {
            "status": "success",
            "portfolio": snapshot,
            "timestamp": snapshot.get("timestamp")
        }
    
    def _record_prediction_bias(self, phase: str) -> None:
        """Record prediction bias for analysis."""
        # This is a placeholder for actual prediction bias tracking
        bias_entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "mode": self.state["current_mode"],
            "quantum_ratio": self.state["current_quantum_ratio"]
        }
        
        # Add market-related bias metrics if available
        if has_market_modules and self.conviction_engine:
            try:
                # Get accuracy metrics
                assets = list(self.conviction_engine.conviction_trackers.keys())
                
                if assets:
                    # Just record first asset's metrics as example
                    symbol = assets[0]
                    conviction = self.conviction_engine.conviction_trackers[symbol]["current_conviction"]
                    
                    # Get sentiment signals if available
                    sentiment_accuracy = 0.0
                    if self.sentiment_engine:
                        try:
                            signals = self.sentiment_engine.get_active_signals(symbol)
                            if signals:
                                # Just use first signal's confidence
                                sentiment_confidence = signals[0]["confidence"]
                                bias_entry["sentiment_confidence"] = sentiment_confidence
                        except Exception:
                            pass
                    
                    bias_entry["conviction"] = conviction
            except Exception:
                pass
        
        # Add to prediction bias log
        self.state["prediction_bias_log"].append(bias_entry)
        
        # Limit log size to 100 entries
        if len(self.state["prediction_bias_log"]) > 100:
            self.state["prediction_bias_log"] = self.state["prediction_bias_log"][-100:]
        
        # Save state
        self._save_state()
    
    def create_middleware_spec(self) -> Dict:
        """
        Create API middleware specification for external services.
        
        Returns:
            Dict with middleware API specification
        """
        # Create MonteBravo middleware spec
        montebravo_spec = {
            "version": "1.0.0",
            "endpoints": {
                "portfolio": "/api/portfolio",
                "recommendations": "/api/recommendations",
                "track_asset": "/api/track",
                "toggle_enjoyment": "/api/toggle_enjoyment",
                "status": "/api/status"
            },
            "auth": {
                "type": "bearer",
                "header": "Authorization"
            },
            "formats": {
                "request": "application/json",
                "response": "application/json"
            }
        }
        
        # Create Ledger bridge spec
        ledger_spec = {
            "version": "1.0.0",
            "endpoints": {
                "webhook": "/api/ledger_webhook",
                "query_balance": "/api/ledger/balance",
                "broadcast_tx": "/api/ledger/broadcast"
            },
            "auth": {
                "type": "bearer",
                "header": "Authorization"
            },
            "supported_chains": [
                "bitcoin",
                "ethereum"
            ],
            "formats": {
                "request": "application/json",
                "response": "application/json"
            }
        }
        
        return {
            "montebravo": montebravo_spec,
            "ledger": ledger_spec,
            "generated_at": datetime.now().isoformat(),
            "base_url": "[BASE_URL_PLACEHOLDER]",
            "documentation_url": "[DOCS_URL_PLACEHOLDER]"
        }

# Singleton pattern
_investment_engine_instance = None

def get_investment_engine(config_path: Optional[str] = None,
                        state_path: Optional[str] = None,
                        debug_mode: bool = False) -> InvestmentEngine:
    """
    Get or create singleton instance of the investment engine.
    
    Args:
        config_path: Path to configuration file
        state_path: Path to engine state file
        debug_mode: Enable debug logging
        
    Returns:
        InvestmentEngine instance
    """
    global _investment_engine_instance
    
    if _investment_engine_instance is None:
        _investment_engine_instance = InvestmentEngine(
            config_path=config_path,
            state_path=state_path,
            debug_mode=debug_mode
        )
    
    return _investment_engine_instance

if __name__ == "__main__":
    # Example usage
    engine = get_investment_engine(debug_mode=True)
    
    # Get engine status
    status = engine.get_status()
    print(f"Engine Status: {status['status']}")
    print(f"Mode: {status['mode']}")
    print(f"Quantum Ratio: {status['quantum_ratio']}")
    print(f"Sub-engines: {status['sub_engines']}")
    
    # Track some assets if not tracked already
    if status['tracked_assets'] == 0:
        engine.track_asset(
            symbol="BTC",
            asset_class="crypto",
            description="Bitcoin",
            initial_conviction=0.7,
            initial_thesis="Long-term store of value with institutional adoption increasing"
        )
        
        engine.track_asset(
            symbol="ETH",
            asset_class="crypto",
            description="Ethereum",
            initial_conviction=0.75,
            initial_thesis="Leading smart contract platform with strong developer ecosystem"
        )
        
        engine.track_asset(
            symbol="NVDA",
            asset_class="equity",
            description="NVIDIA Corporation",
            initial_conviction=0.8,
            initial_thesis="AI chip leader with strong moat and growing data center business"
        )
    
    # Get investment recommendations
    recs = engine.get_investment_recommendations()
    print("\nInvestment Recommendations:")
    print(f"Mode: {recs['mode']}, Quantum Ratio: {recs['quantum_ratio']:.2f}")
    
    for rec in recs["recommendations"]:
        print(f"\n{rec['symbol']} ({rec['asset_class']}):")
        print(f"  Conviction: {rec['conviction']:.2f} ({rec['conviction_level']})")
        print(f"  Time Horizon: {rec['time_horizon']}")
        print(f"  Base Allocation: {rec['base_allocation']*100:.1f}%")
        print(f"  Tactical Adjustment: {rec['tactical_adjustment']*100:+.1f}%")
        print(f"  Final Allocation: {rec['final_allocation']*100:.1f}%")
        print(f"  Sentiment: {rec['sentiment']['direction']} ({rec['sentiment']['strength']:.2f})")
        print(f"  Signal Count: {rec['sentiment']['signals_count']}")
        
        if rec['sentiment']['signals']:
            print("  Top Signals:")
            for signal in rec['sentiment']['signals']:
                print(f"    - {signal['type']} ({signal['strength']}): {signal['description']}")
    
    print(f"\nAsset Class Allocations:")
    for asset_class, allocation in recs["asset_class_allocations"].items():
        print(f"  {asset_class}: {allocation*100:.1f}%")
    
    print(f"Cash: {recs['cash_allocation']*100:.1f}%")
    
    # Test toggling enjoyment mode
    print("\nToggling Enjoyment Mode ON:")
    toggle_result = engine.toggle_enjoyment_mode(True)
    print(f"Result: {toggle_result['message']}")
    
    # Get updated recommendations in enjoyment mode
    enjoyment_recs = engine.get_investment_recommendations()
    print(f"\nEnjoyment Mode Quantum Ratio: {enjoyment_recs['quantum_ratio']:.2f}")
    
    # Create middleware spec
    middleware = engine.create_middleware_spec()
    print("\nMiddleware API Spec:")
    print(f"MonteBravo Endpoints: {middleware['montebravo']['endpoints']}")
    print(f"Ledger Endpoints: {middleware['ledger']['endpoints']}")
    
    # Toggle back off
    print("\nToggling Enjoyment Mode OFF:")
    toggle_result = engine.toggle_enjoyment_mode(False)
    print(f"Result: {toggle_result['message']}")