"""
WiltonOS Agents Package
=====================

This package contains the various cognitive agents that make up the WiltonOS system.
Agents are responsible for processing data, maintaining coherence, and executing tasks.
"""

from .agent_manager import run_agents, load_agent, get_agent, calculate_system_coherence, shutdown

__all__ = ['run_agents', 'load_agent', 'get_agent', 'calculate_system_coherence', 'shutdown']