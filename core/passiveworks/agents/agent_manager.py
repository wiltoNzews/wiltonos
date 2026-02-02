"""
WiltonOS Agent Manager
=====================

Manages the activation and coordination of agents within the WiltonOS ecosystem.
This module serves as the central orchestration point for all cognitive agents.

Core functions:
1. Initializes and manages agent lifecycle
2. Routes data and instructions to appropriate agents
3. Maintains quantum coherence balance (3:1 ratio)
"""

import os
import logging
import importlib
import threading
import time
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [AGENT-MGR] %(message)s',
)

logger = logging.getLogger('agent_manager')

# Default agents to load
DEFAULT_AGENTS = [
    "core_agent",
    "lemniscate_mode"
]

# Global variables
active_agents = {}
agent_threads = {}
manager_running = False
coherence_value = 0.75  # Target: 3:1 ratio (75% coherence, 25% exploration)

def run_agents(manifest):
    """
    Main function to initialize and run all agents
    """
    global manager_running
    
    if manager_running:
        logger.warning("Agent manager already running")
        return
    
    manager_running = True
    logger.info("🧠 Starting WiltonOS Agent Manager...")
    
    # Initialize all default agents
    for agent_name in DEFAULT_AGENTS:
        load_agent(agent_name, manifest)
    
    # Start continuous monitoring thread
    monitor_thread = threading.Thread(
        target=monitor_agents,
        args=(manifest,),
        daemon=True
    )
    monitor_thread.start()
    
    logger.info(f"✅ Agent manager started with {len(active_agents)} agents")

def load_agent(agent_name, manifest):
    """
    Dynamically load and initialize an agent
    """
    if agent_name in active_agents:
        logger.info(f"Agent '{agent_name}' already loaded")
        return active_agents[agent_name]
    
    try:
        # Try to import the agent module
        agent_module = importlib.import_module(f"agents.{agent_name}")
        
        # Get the agent class
        # Most agent modules should have an 'Agent' class
        if hasattr(agent_module, 'Agent'):
            agent_class = getattr(agent_module, 'Agent')
            agent = agent_class()
            active_agents[agent_name] = agent
            
            # Initialize the agent
            if hasattr(agent, 'initialize'):
                agent.initialize(manifest)
            
            logger.info(f"Loaded agent: {agent_name}")
            
            # Start agent in a thread if it has a run method
            if hasattr(agent, 'run'):
                thread = threading.Thread(
                    target=agent_run_wrapper,
                    args=(agent, agent_name, manifest),
                    daemon=True
                )
                thread.start()
                agent_threads[agent_name] = thread
                logger.info(f"Started agent thread: {agent_name}")
            
            return agent
        else:
            logger.error(f"No Agent class found in module: {agent_name}")
            return None
    
    except ImportError as e:
        logger.error(f"Failed to import agent '{agent_name}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading agent '{agent_name}': {str(e)}")
        return None

def agent_run_wrapper(agent, agent_name, manifest):
    """
    Wrapper to safely run an agent and handle exceptions
    """
    try:
        logger.info(f"Agent thread starting: {agent_name}")
        agent.run(manifest)
    except Exception as e:
        logger.error(f"Error in agent '{agent_name}': {str(e)}")

def monitor_agents(manifest):
    """
    Continuously monitor agents and maintain system coherence
    """
    logger.info("Starting agent monitoring thread")
    
    while manager_running:
        try:
            # Check each agent's status
            for agent_name, agent in list(active_agents.items()):
                if agent_name in agent_threads:
                    thread = agent_threads[agent_name]
                    if not thread.is_alive():
                        logger.warning(f"Agent thread '{agent_name}' died, restarting...")
                        # Restart the agent
                        load_agent(agent_name, manifest)
                
                # Check agent coherence if method exists
                if hasattr(agent, 'get_coherence'):
                    agent_coherence = agent.get_coherence()
                    if agent_coherence < 0.5:  # Arbitrary threshold
                        logger.warning(f"Agent '{agent_name}' has low coherence: {agent_coherence}")
            
            # Sleep before next check
            time.sleep(30)
        
        except Exception as e:
            logger.error(f"Error in agent monitor: {str(e)}")
            time.sleep(60)  # Sleep longer on error

def get_agent(agent_name):
    """
    Get a reference to an active agent
    """
    return active_agents.get(agent_name)

def calculate_system_coherence():
    """
    Calculate overall system coherence from all agents
    """
    if not active_agents:
        return coherence_value
    
    total = 0.0
    count = 0
    
    for agent_name, agent in active_agents.items():
        if hasattr(agent, 'get_coherence'):
            try:
                agent_coherence = agent.get_coherence()
                total += agent_coherence
                count += 1
            except Exception as e:
                logger.error(f"Error getting coherence from {agent_name}: {str(e)}")
    
    # Return average if we have values, otherwise use default
    return total / count if count > 0 else coherence_value

def shutdown():
    """
    Gracefully shutdown all agents
    """
    global manager_running
    logger.info("Shutting down agent manager...")
    
    manager_running = False
    
    # Shutdown each agent
    for agent_name, agent in active_agents.items():
        if hasattr(agent, 'shutdown'):
            try:
                agent.shutdown()
                logger.info(f"Shutdown agent: {agent_name}")
            except Exception as e:
                logger.error(f"Error shutting down {agent_name}: {str(e)}")
    
    logger.info("Agent manager shutdown complete")

if __name__ == "__main__":
    # This allows running the agent manager directly for testing
    import json
    
    logger.info("Starting WiltonOS Agent Manager in standalone mode...")
    
    try:
        # Load manifest
        if os.path.exists("../memory/manifest.json"):
            with open("../memory/manifest.json", "r") as f:
                manifest = json.load(f)
        else:
            logger.warning("No manifest found, using empty manifest")
            manifest = {}
        
        # Run agents
        run_agents(manifest)
        
        # Keep running for testing
        while True:
            time.sleep(10)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        shutdown()
    except Exception as e:
        logger.error(f"Error in standalone agent manager: {str(e)}")