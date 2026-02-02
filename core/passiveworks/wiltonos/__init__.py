"""
WiltonOS - Quantum Middleware Ecosystem
A revolutionary system exploring the intersection of consciousness, computation, and creative technological interfaces
"""

import os
import logging
import json
from typing import Dict, Any, Optional
import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
os.makedirs("state", exist_ok=True)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/wiltonos.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("wiltonos")

# WiltonOS version
__version__ = "1.5.1"

# Module initialization tracking
_initialized_modules = {}

def _register_module(module_name: str, version: str):
    """Register an initialized module"""
    _initialized_modules[module_name] = {
        "version": version,
        "initialized_at": datetime.datetime.now().isoformat()
    }
    logger.info(f"Module initialized: {module_name} v{version}")

# Meta-information about the system
_system_info = {
    "name": "WiltonOS",
    "version": __version__,
    "description": "Quantum Middleware Ecosystem",
    "started_at": datetime.datetime.now().isoformat(),
    "environment": os.getenv("NODE_ENV", "development"),
    "modules": _initialized_modules
}

# Import core modules
try:
    from .agent_bus import agent_bus, AgentProfile
    _register_module("agent_bus", "1.0.0")
except ImportError:
    logger.warning("Failed to import agent_bus module")

try:
    from .signal_mesh import signal_mesh_server, create_signal_node
    _register_module("signal_mesh", "1.0.0")
except ImportError:
    logger.warning("Failed to import signal_mesh module")

try:
    from .function_calling import function_registry, function_caller, openai_function
    _register_module("function_calling", "1.0.0")
except ImportError:
    logger.warning("Failed to import function_calling module")

try:
    from .fractal_visualizer import fractal_visualizer, FractalParams
    _register_module("fractal_visualizer", "1.0.0")
except ImportError:
    logger.warning("Failed to import fractal_visualizer module")

try:
    from .streamlit_enhancements import apply_wiltonos_theme, card, metric_card, timeline, glow_text
    _register_module("streamlit_enhancements", "1.0.0")
except ImportError:
    logger.warning("Failed to import streamlit_enhancements module")

try:
    from .db_controller import db
    _register_module("db_controller", "1.0.0")
except ImportError:
    logger.warning("Failed to import db_controller module")

def get_system_info() -> Dict[str, Any]:
    """Get information about the WiltonOS system"""
    # Update modules info
    _system_info["modules"] = _initialized_modules
    return _system_info

def save_system_state(filepath: str = "state/wiltonos_state.json") -> bool:
    """Save the current system state to a file"""
    try:
        state = {
            "system_info": get_system_info(),
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Add module-specific state information
        if 'agent_bus' in _initialized_modules:
            try:
                from .agent_bus import agent_bus
                state["agent_bus"] = {
                    "agents": [a["profile"].to_dict() for a in agent_bus.agents.values()]
                }
            except Exception as e:
                logger.error(f"Error saving agent_bus state: {str(e)}")
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"System state saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving system state: {str(e)}")
        return False

def load_system_state(filepath: str = "state/wiltonos_state.json") -> Optional[Dict[str, Any]]:
    """Load the system state from a file"""
    if not os.path.exists(filepath):
        logger.warning(f"System state file {filepath} not found")
        return None
    
    try:
        with open(filepath, 'r') as f:
            state = json.load(f)
        logger.info(f"System state loaded from {filepath}")
        return state
    except Exception as e:
        logger.error(f"Error loading system state: {str(e)}")
        return None

# Print initialization banner
logger.info(f"""
╭──────────────────────────────────────────────╮
│                                              │
│  WiltonOS v{__version__}                              │
│  Quantum Middleware Ecosystem                │
│                                              │
│  Initialized modules: {len(_initialized_modules)}/{6}              │
│                                              │
╰──────────────────────────────────────────────╯
""")